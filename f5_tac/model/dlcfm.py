# f5_tac.model.cfm.py
from __future__ import annotations

from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default,
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_tensor,
    mask_from_frac_lengths,
)

from f5_tac.model.utils import list_str_to_idx

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

        # Convert text inputs
        if isinstance(texts, list):
            if exists(self.vocab_char_map):
                texts = list_str_to_idx(texts, self.vocab_char_map).to(device)
            else:
                texts = list_str_to_tensor(texts).to(device)
            assert texts.shape[0] == batch  # Should be 2

        # Duration logic
        if isinstance(durations, int):
            durations = torch.full((batch,), durations, device=device, dtype=torch.long)
        durations = torch.tensor(durations, device=device, dtype=torch.long)
        durations = torch.maximum(
            torch.maximum((texts != -1).sum(dim=-1), lens) + 1, durations
        ).clamp(max=max_duration)
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
        text_A, text_B = texts[0:1], texts[1:2]          # [1, L]
        mask_A, mask_B = cond_mask[0:1], cond_mask[1:2]  # [1, T, 1]
        dur_A, dur_B = durations[0], durations[1]       # scalars

        # Noise initialization per speaker
        torch.manual_seed(seed) if seed is not None else None
        y0_A = torch.randn((1, dur_A, self.num_channels), device=device, dtype=step_cond.dtype)
        y0_B = torch.randn((1, dur_B, self.num_channels), device=device, dtype=step_cond.dtype)

        # Pad to max_duration for batching
        y0_A = F.pad(y0_A, (0, 0, 0, max_duration - dur_A), value=0.0)
        y0_B = F.pad(y0_B, (0, 0, 0, 0, max_duration - dur_B), value=0.0)

        # Timestep schedule
        if use_epss:
            t = get_epss_timesteps(steps, device=device, dtype=step_cond.dtype)
        else:
            t = torch.linspace(0, 1, steps + 1, device=device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        # ODE function with CFG
        def fn(t_cur, x_pair):
            x_A, x_B = x_pair.chunk(2, dim=0)  # split along batch dim
            if cfg_strength < 1e-5:
                pred_A, pred_B = self.transformer(
                    x_A, cond_A, text_A, mask_A,
                    x_B, cond_B, text_B, mask_B,
                    time=t_cur,
                    drop_audio_cond=False,
                    drop_text=False,
                    cfg_infer=False,
                    cache=True,
                )
                return torch.cat([pred_A, pred_B], dim=0)

            # Classifier-free guidance path
            pred_cfg_A, pred_cfg_B = self.transformer(
                x_A, cond_A, text_A, mask_A,
                x_B, cond_B, text_B, mask_B,
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

        # Combine noise for integration
        y0 = torch.cat([y0_A, y0_B], dim=0)  # [2, T_out, D]

        # Integrate ODE
        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        # Extract final and restore prompt frames
        sampled = trajectory[-1]  # [2, T_out, D]
        out = torch.where(cond_mask, conds, sampled)

        # Optional vocoding
        if vocoder is not None:
            out = out.permute(0, 2, 1).contiguous()  # [2, D, T_out]
            out = vocoder(out)

        return out, trajectory



    @torch.no_grad()
    def sample(
        self,
        text: list[str],
        speaker_id: str, # 'A' or 'B'
        cond: float["b n d"] | float["b nw"] | None = None,
        duration: int | int["b"] | None = None,
        *,
        lens: int["b"] | None = None, # noqa: F821
        steps=32,
        cfg_strength=1.0,
        # ... other sampling params from original CFM
    ):
        """
        Adapted sampling method for the dual-speaker model.
        Generates audio for a specific speaker by creating a dummy channel for the other.
        """
        self.eval()
        
        is_speaker_A = speaker_id.upper() == 'A'
        batch_size = len(text)
        device = self.device

        # --- Text processing (for the active speaker) ---
        if isinstance(text, list):
            text = list_str_to_tensor(text).to(device)
        
        # --- Reference audio processing (for the active speaker) ---
        if exists(cond) and cond.ndim == 2:
            cond = self.mel_spec(cond).permute(0, 2, 1)

        # --- Duration processing ---
        if not exists(duration):
            duration = (text != -1).sum(dim=-1) + 10 # Heuristic
        if isinstance(duration, int):
            duration = torch.full((batch_size,), duration, device=device, dtype=torch.long)
        
        max_duration = duration.max()
        
        # --- ODE Integration ---
        def fn(t, x):
            # 'x' is the noisy input for the active speaker, shape [b, n, d]
            
            # Create placeholders for the inactive speaker
            dummy_phi = torch.zeros_like(x)
            dummy_cond = torch.zeros_like(x) if not exists(cond) else torch.zeros_like(cond)
            dummy_text = torch.zeros_like(text)
            
            # Assign inputs based on which speaker is active
            phi_A, cond_A, text_A = (x, cond, text) if is_speaker_A else (dummy_phi, dummy_cond, dummy_text)
            phi_B, cond_B, text_B = (dummy_phi, dummy_cond, dummy_text) if is_speaker_A else (x, cond, text)
            
            # Stack inputs for the transformer: [b*s, n, d]
            phi_stacked = torch.cat([phi_A, phi_B], dim=0)
            cond_stacked = torch.cat([cond_A, cond_B], dim=0) if exists(cond) else None
            text_stacked = torch.cat([text_A, text_B], dim=0)

            # Create speaker mask: [b, s]
            spk_mask = torch.tensor([[is_speaker_A, not is_speaker_A]], device=device).repeat(batch_size, 1)
            
            # Predict flow using the transformer with CFG
            pred_stacked = self.transformer(
                x=phi_stacked,
                cond=cond_stacked,
                text=text_stacked,
                time=t,
                cfg_infer=True,
                spk_mask=spk_mask,
            )
            
            # Extract the prediction for the active speaker
            # The CFG logic in DiT returns a stacked [pred, null_pred] of shape [2 * b*s, n, d]
            # We take the first chunk (the conditional prediction)
            pred_cond, _ = torch.chunk(pred_stacked, 2, dim=0)
            
            # Un-stack speakers and get the active one
            pred_A, pred_B = torch.chunk(pred_cond, 2, dim=0)
            
            return pred_A if is_speaker_A else pred_B

        # Initial noise for the active speaker
        y0 = torch.randn(batch_size, max_duration, self.num_channels, device=device)
        
        t = torch.linspace(0, 1, steps + 1, device=device)
        
        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()
        
        sampled = trajectory[-1]
        return sampled, trajectory

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
            # 1) merge into one list of 2*batch utterances
            merged_texts = text_A + text_B
            # 2) convert all 2*batch sequences → ids and pad to the same length L
            merged_texts = list_str_to_idx(merged_texts, self.vocab_char_map).to(device)
            # 3) split back into two [batch, L] tensors
            text_A, text_B = torch.chunk(merged_texts, 2, dim=0)
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

        # --- Stack inputs for the DiTWithTAC Transformer ---
        # Shape becomes [2*b, n, d]
        # TODO: Can be done with a loop for n generations...
        # phi_stacked = torch.cat([phi_A, phi_B], dim=0)
        # cond_stacked = torch.cat([cond_A, cond_B], dim=0)
        # text_stacked = torch.cat([text_A, text_B], dim=0)
        # mask_stacked = torch.cat([mask_A, mask_B], dim=0)


        # --- CFG Training Drops --- from f5 repo
        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # --- Call the DiTWithTAC Transformer ---
        # The transformer expects stacked inputs. spk_mask is not needed for training
        # as the default behavior in DiTBlockWithTAC assumes all speakers are active.
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
