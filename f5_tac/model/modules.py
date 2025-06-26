import torch
import torch.nn.functional as F
import torch.nn as nn
import difflib

from f5_tts.model.backbones.dit import DiT
from f5_tts.model.modules import DiTBlock


class DiTBlockWithTAC(DiTBlock):
    def __init__(self, *args, dim, **kwargs):
        super().__init__(*args, dim=dim, **kwargs)
        self.tac = TAC(in_channels=dim, expansion_f=3, dropout=0.1)
    
    def forward(self, x, t, mask=None, rope=None, spk_mask=None):
        x = super().forward(x, t, mask=mask, rope=rope) # -> (B⋅S, T, D)

        B2, T, D = x.shape
        S = 2 # TODO: infer from speaker count
        B = B2 // S

        x_spk = x.view(B, S, T, D) # -> (B, S, T, D)

        # per speaker average internally
        x_spk = self.tac(x_spk, mask=spk_mask)

        x = x_spk.view(B2, T, D)   # -> (B⋅S, T, D)
        return x