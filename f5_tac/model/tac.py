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

        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.uniform_(module.weight, a=-0.01, b=0.01)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, inp, mask=None):
        # bsz, mics, frames, channels
        bsz, mics, channels = inp.shape
        transformed = self.transform_shared(inp.reshape(bsz * mics, channels)).reshape(
            bsz, mics, self.hid_channels
        )
        if mask is not None:
            # average = transformed.masked_fill(mask, 0.0)
            average = (
                (transformed * mask.unsqueeze(-1))           # broadcast mask: (B,mics,1)→(B,mics,hid)
                .sum(1)                                      # sum over speakers → (B,hid)
                / mask.sum(1, keepdim=True).float()          # divide by #active speakers (B,1)→broadcast (B,hid)
            )
            # average = average.masked_fill(mask, 0.0)
        else:
            average = transformed.mean(1)

        average = self.transform_avg(average.unsqueeze(1).repeat(1, mics, 1))
        transformed = torch.cat((transformed, average), -1).reshape(
            bsz * mics,
            2 * self.hid_channels,
        )
        out = self.norm(self.transform_final(transformed)) + inp.reshape(
            bsz * mics, channels
        )
        return out.reshape(bsz, mics, channels)