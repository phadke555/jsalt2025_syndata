from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    AdaLayerNorm_Final,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiT,
    TimestepEmbedding,
    get_pos_embed_indices,
    precompute_freqs_cis,
)

from f5_tac.model.modules import DiTBlockWithTAC


class DiTWithTAC(DiT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.transformer_blocks = nn.ModuleList([
            DiTBlockWithTAC (
                dim          = self.dim,
                heads        = blk.attn.heads,
                dim_head     = blk.attn.inner_dim // blk.attn.heads,
                ff_mult      = blk.ff.ff[0].out_features  / blk.ff.ff[0].in_features,
                dropout      = blk.ff.ff[1].p,
                qk_norm      = blk.attn.q_norm      is not None,
                pe_attn_head = blk.attn.processor.pe_attn_head,
                attn_backend = blk.attn.processor.attn_backend,
                attn_mask_enabled = blk.attn.processor.attn_mask_enabled,
            )
            for blk in self.transformer_blocks
        ])
    
    def forward(
        self,
        x: float["b n d"],  # nosied input audio  # noqa: F722
        cond: float["b n d"],  # masked cond audio  # noqa: F722
        # x: float["b s n d"],  # nosied input audio  # noqa: F722
        # cond: float["b s n d"],  # masked cond audio  # noqa: F722
        text: int["b nt"],  # text  # noqa: F722
        time: float["b"] | float[""],  # time step  # noqa: F821 F722
        mask: bool["b n"] | None = None,  # noqa: F722
        drop_audio_cond: bool = False,  # cfg for cond audio
        drop_text: bool = False,  # cfg for text
        cfg_infer: bool = False,  # cfg inference, pack cond & uncond forward
        cache: bool = False,
        spk_mask: bool["b s"] | None = None,  # NEW: (b, s) boolean mask
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, text: text, x: noised audio + cond audio + text
        t = self.time_embed(time)
        if cfg_infer:  # pack cond & uncond forward: b n d -> 2b n d
            x_cond = self.get_input_embed(x, cond, text, drop_audio_cond=False, drop_text=False, cache=cache)
            x_uncond = self.get_input_embed(x, cond, text, drop_audio_cond=True, drop_text=True, cache=cache)
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
        else:
            x = self.get_input_embed(x, cond, text, drop_audio_cond=drop_audio_cond, drop_text=drop_text, cache=cache)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, mask, rope, spk_mask use_reentrant=False)
            else:
                x = block(x, t, mask=mask, rope=rope, spk_mask=spk_mask)

            # reshape
            # espnet: TAC

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output