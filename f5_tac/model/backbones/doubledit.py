# f5_tac.model.backbones.dittac.py

from __future__ import annotations
import copy
import torch
import torch.nn.functional as F
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    AdaLayerNorm_Final,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    TimestepEmbedding,
    get_pos_embed_indices,
    precompute_freqs_cis,
)

from f5_tts.model.backbones.dit import DiT
from f5_tac.model.tac import TAC

from f5_tac.model.modules import DiTBlockWithTAC


class DoubleDiT(DiT):
    def __init__(self, **dit_kwargs):
        super().__init__(**dit_kwargs)

        self.blocks_A = copy.deepcopy(self.transformer_blocks)
        self.blocks_B = copy.deepcopy(self.transformer_blocks)
        delattr(self, "transformer_blocks")

        self.cross_tacs = nn.ModuleList([
            TAC(in_channels=self.dim, expansion_f=3, dropout=0.1)
            for _ in range(self.depth)
        ])
        # self.cross_tac = TAC(in_channels=self.dim, expansion_f=3, dropout=0.1)

    def forward(self, x_A, cond_A, text_A, mask_A,
                      x_B, cond_B, text_B, mask_B, time,
                      drop_audio_cond=False, drop_text=False,
                      cfg_infer=False, cache=False):
        
        batch, seq_len = x_A.shape[0], x_A.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        tA = self.time_embed(time)
        tB = self.time_embed(time)

        if cfg_infer:  # pack cond & uncond forward: b n d -> 2b n d
            x_cond_A = self.get_input_embed(x_A, cond_A, text_A, drop_audio_cond=False, drop_text=False, cache=cache)
            x_uncond_A = self.get_input_embed(x_A, cond_A, text_A, drop_audio_cond=True, drop_text=True, cache=cache)
            x_A = torch.cat((x_cond_A, x_uncond_A), dim=0)
            tA = torch.cat((tA, tA), dim=0)
            mask_A = torch.cat((mask_A, mask_A), dim=0) if mask_A is not None else None

            x_cond_B = self.get_input_embed(x_B, cond_B, text_B, drop_audio_cond=False, drop_text=False, cache=cache)
            x_uncond_B = self.get_input_embed(x_B, cond_B, text_B, drop_audio_cond=True, drop_text=True, cache=cache)
            x_B = torch.cat((x_cond_B, x_uncond_B), dim=0)
            tB = torch.cat((tB, tB), dim=0)
            mask_B = torch.cat((mask_B, mask_B), dim=0) if mask_B is not None else None

        else:
            x_A = self.get_input_embed(x_A, cond_A, text_A, drop_audio_cond, drop_text, cache)
            x_B = self.get_input_embed(x_B, cond_B, text_B, drop_audio_cond, drop_text, cache)

        ropeA = self.rotary_embed.forward_from_seq_len(seq_len)
        ropeB = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection:
            res_A = x_A; res_B = x_B

        # 2) Loop over two block sequences + interleave cross-TAC
        for i, (block_A, block_B) in enumerate(zip(self.blocks_A, self.blocks_B)):
            x_A = block_A(x_A, tA, mask=mask_A, rope=ropeA)
            x_B = block_B(x_B, tB, mask=mask_B, rope=ropeB)

            # stack, fuse, unstack
            B, T, D = x_A.shape
            x = torch.stack([x_A, x_B], dim=2)      # [B, T, 2, D]
            x = x.reshape(B*T, 2, D)
            mask = torch.ones(B*T, 2, device=x_A.device, dtype=torch.bool)
            tac = self.cross_tacs[i]
            x = tac(x, mask=mask) # [B*T, 2, D]
            x = x.reshape(B, T, 2, D)
            x_A, x_B = x[:, :, 0, :], x[:, :, 1, :]

        if self.long_skip_connection:
            x_A = self.long_skip(torch.cat([x_A, res_A], dim=-1))
            x_B = self.long_skip(torch.cat([x_B, res_B], dim=-1))
        
        x_A = self.norm_out(x_A, tA); out_A = self.proj_out(x_A)
        x_B = self.norm_out(x_B, tB); out_B = self.proj_out(x_B)
        return out_A, out_B