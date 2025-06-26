# f5_tac.model.tac.py

import torch

import torch.nn
import difflib

def get_layer(l_name, library=torch.nn):
    """Return layer object handler from library e.g. from torch.nn

    E.g. if l_name=="elu", returns torch.nn.ELU.

    Args:
        l_name (string): Case insensitive name for layer in library (e.g. .'elu').
        library (module): Name of library/module where to search for object handler
        with l_name e.g. "torch.nn".

    Returns:
        layer_handler (object): handler for the requested layer e.g. (torch.nn.ELU)

    """

    all_torch_layers = [x for x in dir(torch.nn)]
    match = [x for x in all_torch_layers if l_name.lower() == x.lower()]
    if len(match) == 0:
        close_matches = difflib.get_close_matches(
            l_name, [x.lower() for x in all_torch_layers]
        )
        raise NotImplementedError(
            "Layer with name {} not found in {}.\n Closest matches: {}".format(
                l_name, str(library), close_matches
            )
        )
    elif len(match) > 1:
        close_matches = difflib.get_close_matches(
            l_name, [x.lower() for x in all_torch_layers]
        )
        raise NotImplementedError(
            "Multiple matchs for layer with name {} not found in {}.\n "
            "All matches: {}".format(l_name, str(library), close_matches)
        )
    else:
        # valid
        layer_handler = getattr(library, match[0])
        return layer_handler



class TAC(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        expansion_f=3,
        dropout=0.0,
        norm_type="LayerNorm",
        activation="relu",
    ):
        super(TAC, self).__init__()
        hid_channels = int(in_channels * expansion_f)
        self.hid_channels = hid_channels
        self.transform_shared = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hid_channels, 1),
            get_layer(activation)(),
            torch.nn.Dropout(dropout),
        )
        self.transform_avg = torch.nn.Sequential(
            torch.nn.Linear(hid_channels, hid_channels, 1),
            get_layer(activation)(),
            torch.nn.Dropout(dropout),
        )
        self.transform_final = torch.nn.Sequential(
            torch.nn.Linear(2 * hid_channels, in_channels, 1),
            get_layer(activation)(),
            torch.nn.Dropout(dropout),
        )
        self.norm = get_layer(norm_type)(in_channels)

    def forward(self, inp, mask=None):
        B, S, T, C = inp.shape

        flat = inp.view(B*S*T, C)
        shared = self.transform_shared(flat).view(B, S, T, self.hid_channels)

        if mask is not None:
            # mask: (B, S) -> (B, S, 1, 1) to broadcast over time
            m = mask.view(B, S, 1, 1).float()
            total = m.sum(dim=1, keepdim=True)
            avg_spk = (shared * m).sum(dim=1) / total.squeeze(1)
        else:
            avg_spk = shared.mean(dim=1)
        
        avg_flat = avg_spk.contiguous().view(B * T, self.hid_channels)
        avg_hid = self.transform_avg(avg_flat).view(B, T, self.hid_channels)
        avg_rep = avg_hid.unsqueeze(1).expand(-1, S, -1, -1)

        concat   = torch.cat([shared, avg_rep], dim=-1) 
        cat_flat = concat.view(B * S * T, 2 * self.hid_channels)
        out_flat = self.transform_final(cat_flat)
        out = out_flat.view(B, S, T, C)

        return self.norm(out) + inp