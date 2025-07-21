# f5_tac.model.modules.py

import torch
import torch.nn.functional as F
import torch.nn as nn
import difflib

from f5_tts.model.backbones.dit import DiT
from f5_tts.model.modules import DiTBlock

from f5_tac.model.tac import TAC


class DiTBlockWithTAC(DiTBlock):
    def __init__(self, *args, dim, num_speakers: int = 2, **kwargs):
        super().__init__(*args, dim=dim, **kwargs)
        self.num_speakers = num_speakers
        self.tac = TAC(in_channels=dim, expansion_f=3, dropout=0.1)
    
    def forward(self, x, t, mask=None, rope=None, spk_mask=None):
        x = super().forward(x, t, mask=mask, rope=rope) # -> (B⋅S, T, D)

        B2, T, D = x.shape
        if spk_mask is None:
            S = self.num_speakers # TODO: infer from speaker count
            B = B2 // S
            spk_mask = x.new_ones((B, S), dtype=torch.bool)
        else:
            S = spk_mask.shape[1]
            B = B2 // S

        # per active speaker fusion
        # x = x.view(B, S, T, D) # -> (B, S, T, D)
            
        # Convert (B*S, T, D) → (B, S, T, D) using chunk
        x = torch.stack(torch.chunk(x, S, dim=0), dim=1)

        
        x = x.permute(0, 2, 1, 3) # -> (B, T, S, D)
        x = x.reshape(B*T, S, D) 

        spk_mask = spk_mask.unsqueeze(1).expand(-1, T, -1)
        spk_mask = spk_mask.reshape(B*T, S)
        
        # per speaker average internally
        x = self.tac(x, mask=spk_mask)
        x = x.reshape(B, T, S, D).permute(0, 2, 1, 3)

        # x = x.reshape(B*S, T, D)   # -> (B⋅S, T, D)

        blocks = [x[:, i] for i in range(S)]    # each (B, T, D)
        x = torch.cat(blocks, dim=0)            # → (B⋅S, T, D)

        return x
