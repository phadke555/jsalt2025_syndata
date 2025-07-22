# f5_tac.model.cfm.py
from __future__ import annotations

from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default,
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)

class CFMDD(nn.Module):
    """
    Conditional Flow Matching (CFM) Model adapted for a two-speaker architecture
    using a Transformer with Transform-Average-Concatenate (TAC) layers (DiTWithTAC).
    """
    def __init__(
        self,
        transformer: nn.Module, # Expects DiTWithTAC
        sigma=0.0,
        odeint_kwargs: dict = dict(
            method="euler"
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # Mel spectrogram module
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # Classifier-free guidance probabilities
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # The core transformer model (DiTWithTAC)
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # Conditional flow related
        self.sigma = sigma

        # Sampling related
        self.odeint_kwargs = odeint_kwargs

        # Vocab map for tokenization
        self.vocab_char_map = vocab_char_map

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample_joint(
        self,
        texts: list[str],            # [text_A, text_B]
        conds: torch.Tensor,         # shape [2, n_mels, T]
        durations: list[int] | None = None,
        lens: torch.Tensor | None = None,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()
        device = self.device

        # Prepare conds → mel channels last: [2, T, D]
        conds = conds.to(device)
        if conds.ndim == 2:
            conds = self.mel_spec(conds)
        conds = conds.permute(0, 2, 1).to(next(self.parameters()).dtype)  # [2, T, D]

        batch, cond_seq_len, D = conds.shape
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        text_A, text_B = texts[0:1], texts[1:2]  
        # Convert text inputs
        if isinstance(text_A, list):
            if exists(self.vocab_char_map):
                text_A = list_str_to_idx(text_A, self.vocab_char_map).to(device)
                text_B = list_str_to_idx(text_B, self.vocab_char_map).to(device)
            else:
                texts = list_str_to_tensor(texts).to(device)
            # assert texts.shape[0] == batch  # Should be 2

        # Duration logic
        if isinstance(durations, int):
            durations = torch.full((batch,), durations, device=device, dtype=torch.long)
        durations = torch.tensor(durations, device=device, dtype=torch.long)
        # durations = torch.maximum(
        #     torch.maximum((texts != -1).sum(dim=-1), lens) + 1, durations
        # ).clamp(max=max_duration)
        max_duration = durations.amax()

        # Build masks and step_cond per speaker
        cond_mask = lens_to_mask(lens)  # [2, T]
        conds = F.pad(conds, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        if no_ref_audio:
            conds = torch.zeros_like(conds)

        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)  # [2, T, 1]
        step_cond = torch.where(cond_mask, conds, torch.zeros_like(conds))

        # Split inputs per speaker
        cond_A, cond_B = step_cond[0:1], step_cond[1:2]  # [1, T, D]
        mask_A, mask_B = cond_mask[0:1], cond_mask[1:2]  # [1, T, 1]
        dur_A, dur_B = durations[0], durations[1]       # scalars


        # ODE function with CFG
        def fn(t_cur, x_pair):
            x_A, x_B = x_pair.chunk(2, dim=0)  # split along batch dim
            if cfg_strength < 1e-5:
                pred_A, pred_B = self.transformer(
                    x_A, cond_A, text_A, None,
                    x_B, cond_B, text_B, None,
                    time=t_cur,
                    drop_audio_cond=False,
                    drop_text=False,
                    cfg_infer=False,
                    cache=True,
                )
                return torch.cat([pred_A, pred_B], dim=0)

            # Classifier-free guidance path
            pred_cfg_A, pred_cfg_B = self.transformer(
                x_A, cond_A, text_A, None,
                x_B, cond_B, text_B, None,
                time=t_cur,
                drop_audio_cond=False,
                drop_text=False,
                cfg_infer=True,
                cache=True,
            )
            pred_A, null_pred_A = pred_cfg_A.chunk(2, dim=0)
            pred_B, null_pred_B = pred_cfg_B.chunk(2, dim=0)
            guided_A = pred_A + (pred_A - null_pred_A) * cfg_strength
            guided_B = pred_B + (pred_B - null_pred_B) * cfg_strength
            return torch.cat([guided_A, guided_B], dim=0)

        # Noise initialization per speaker
        torch.manual_seed(seed) if seed is not None else None
        y0_list_A = []
        y0_list_B = []
        y0_list_A.append(torch.randn(max_duration, self.num_channels, device=device, dtype=step_cond.dtype))

        y0_list_B.append(torch.randn(max_duration, self.num_channels, device=device, dtype=step_cond.dtype))

        y0_A = pad_sequence(y0_list_A, padding_value=0.0, batch_first=True)  # [1, T_A, D]
        y0_B = pad_sequence(y0_list_B, padding_value=0.0, batch_first=True)  # [1, T_B, D]

        # # Pad to max_duration for batching
        # y0_A = F.pad(y0_A, (0, 0, 0, max_duration - dur_A), value=0.0)
        # y0_B = F.pad(y0_B, (0, 0, 0, max_duration - dur_B), value=0.0)

        # Timestep schedule
        if use_epss:
            t = get_epss_timesteps(steps, device=device, dtype=step_cond.dtype)
        else:
            t = torch.linspace(0, 1, steps + 1, device=device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        # Combine noise for integration
        y0 = torch.cat([y0_A, y0_B], dim=0)  # [2, T_out, D]

        # Integrate ODE
        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        # Extract final and restore prompt frames
        sampled = trajectory[-1]  # [2, T_out, D]
        out = torch.where(cond_mask, conds, sampled)

        # # Optional vocoding
        # if vocoder is not None:
        #     out = out.permute(0, 2, 1).contiguous()  # [2, D, T_out]
        #     out = vocoder(out)

        return out, trajectory


    def forward(
        self,
        mel_A: float["b n d"],
        text_A: int["b nt"] | list[str],
        mel_lengths_A: int["b"],
        mel_B: float["b n d"],
        text_B: int["b nt"] | list[str],
        mel_lengths_B: int["b"],
        noise_scheduler: str | None = None,
    ):
        """
        Adapted forward pass for dual-speaker training. Stacks speaker data before
        passing to the DiTWithTAC transformer.

        Returns:
            loss_A (Tensor): The loss for speaker A.
            loss_B (Tensor): The loss for speaker B.
            cond_A (Tensor): The conditional input for speaker A (for debugging).
            pred_A (Tensor): The prediction for speaker A (for debugging).
            cond_B (Tensor): The conditional input for speaker B (for debugging).
            pred_B (Tensor): The prediction for speaker B (for debugging).
        """
        # --- Pre-process Speaker A ---
        if mel_A.ndim == 2: mel_A = self.mel_spec(mel_A).permute(0, 2, 1)
        # if isinstance(text_A, list): text_A = list_str_to_tensor(text_A).to(self.device)
        
        # --- Pre-process Speaker B ---
        if mel_B.ndim == 2: mel_B = self.mel_spec(mel_B).permute(0, 2, 1)
        # if isinstance(text_B, list): text_B = list_str_to_tensor(text_B).to(self.device)

        batch, _, dtype, device = *mel_A.shape[:2], mel_A.dtype, self.device
        if isinstance(text_A, list) and isinstance(text_B, list):
            text_A = list_str_to_idx(text_A, self.vocab_char_map).to(device)
            text_B = list_str_to_idx(text_B, self.vocab_char_map).to(device)
        else:
            # If one side is already tensor (e.g. resumed), ensure it’s on device
            if isinstance(text_A, torch.Tensor):
                text_A = text_A.to(device)
            if isinstance(text_B, torch.Tensor):
                text_B = text_B.to(device)
        
        # --- Flow Matching Logic (performed independently) ---
        x1_A, x0_A = mel_A, torch.randn_like(mel_A)
        x1_B, x0_B = mel_B, torch.randn_like(mel_B)
        
        time = torch.rand((batch,), dtype=dtype, device=device)
        t = time.view(-1, 1, 1)

        phi_A, flow_A = (1 - t) * x0_A + t * x1_A, x1_A - x0_A
        phi_B, flow_B = (1 - t) * x0_B + t * x1_B, x1_B - x0_B
        flow_mix = flow_A + flow_B

        # --- Create masks and conditional inputs ---
        mask_A = lens_to_mask(mel_lengths_A)
        frac_lengths_A = torch.zeros((batch,), device=device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask_A = mask_from_frac_lengths(mel_lengths_A, frac_lengths_A) & mask_A
        cond_A = torch.where(rand_span_mask_A[..., None], torch.zeros_like(x1_A), x1_A)

        mask_B = lens_to_mask(mel_lengths_B)
        frac_lengths_B = torch.zeros((batch,), device=device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask_B = mask_from_frac_lengths(mel_lengths_B, frac_lengths_B) & mask_B
        cond_B = torch.where(rand_span_mask_B[..., None], torch.zeros_like(x1_B), x1_B)

        # --- CFG Training Drops --- from f5 repo
        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # --- Call the Double DiT transformer ---
        pred_A, pred_B = self.transformer(
            phi_A, cond_A, text_A, mask_A,
            phi_B, cond_B, text_B, mask_B,
            time,
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text
        )

        pred_mix = pred_A + pred_B

        loss_A = F.mse_loss(pred_A, flow_A, reduction="none")
        loss_A = loss_A[rand_span_mask_A].mean()

        loss_B = F.mse_loss(pred_B, flow_B, reduction="none")
        loss_B = loss_B[rand_span_mask_B].mean()

        mask_mix = rand_span_mask_A | rand_span_mask_B
        mix_loss = F.mse_loss(pred_mix, flow_mix, reduction = "none")
        mix_loss = mix_loss[mask_mix].mean()

        return loss_A, loss_B, mix_loss, cond_A, pred_A, cond_B, pred_B