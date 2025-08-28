import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn
import difflib

### espnet channel attention and related implementations used here
### from: https://github.com/espnet/espnet/blob/04f57dbc1d852531fd33e30f7e870eb163070c22/espnet2/enh/layers/uses.py#L290

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
    
class LayerNormalization(nn.Module):
    def __init__(self, input_dim, dim=1, total_dim=4, eps=1e-5):
        super().__init__()
        self.dim = dim if dim >= 0 else total_dim + dim
        param_size = [1 if ii != self.dim else input_dim for ii in range(total_dim)]
        self.gamma = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        self.eps = eps

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, x):
        if x.ndim - 1 < self.dim:
            raise ValueError(
                f"Expect x to have {self.dim + 1} dimensions, but got {x.ndim}"
            )
        mu_ = x.mean(dim=self.dim, keepdim=True)
        std_ = torch.sqrt(x.var(dim=self.dim, unbiased=False, keepdim=True) + self.eps)
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat
    
class ChannelAttention(nn.Module):
    def __init__(
        self, input_dim, att_heads=4, att_dim=256, activation="relu", eps=1e-5, init_scale=1.0
    ):
        """Channel Attention module.

        Args:
            input_dim (int): dimension of the input feature.
            att_heads (int): number of attention heads in self-attention.
            att_dim (int): projection dimension for query and key before self-attention.
            activation (str): non-linear activation function.
            eps (float): epsilon for layer normalization.
        """
        super().__init__()
        self.att_heads = att_heads
        self.att_dim = att_dim
        self.activation = activation
        self.init_scale = init_scale
        assert input_dim % att_heads == 0, (input_dim, att_heads)
        self.attn_conv_Q = nn.Sequential(
            nn.Linear(input_dim, att_dim),
            get_layer(activation)(),
            LayerNormalization(att_dim, dim=-1, total_dim=5, eps=eps),
        )
        self.attn_conv_K = nn.Sequential(
            nn.Linear(input_dim, att_dim),
            get_layer(activation)(),
            LayerNormalization(att_dim, dim=-1, total_dim=5, eps=eps),
        )
        self.attn_conv_V = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            get_layer(activation)(),
            LayerNormalization(input_dim, dim=-1, total_dim=5, eps=eps),
        )
        self.attn_concat_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            get_layer(activation)(),
            LayerNormalization(input_dim, dim=-1, total_dim=5, eps=eps),
        )

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            # choose initializer based on activation
            if self.activation == 'relu':
                # He (Kaiming) initialization for ReLU
                torch.nn.init.kaiming_uniform_(m.weight, a=0, nonlinearity='relu')
            else:
                # Xavier/Glorot init with gain for other nonlinearities
                gain = torch.nn.init.calculate_gain(self.activation)
                torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            # zero biases
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

            if self.init_scale != 1.0:
                m.weight.data.mul_(self.init_scale)

    def __getitem__(self, key):
        return getattr(self, key)

    def forward(self, x, ref_channel=None):
        """ChannelAttention Forward.

        Args:
            x (torch.Tensor): input feature (batch, C, N, freq, time)
            ref_channel (None or int): index of the reference channel.
        Returns:
            output (torch.Tensor): output feature (batch, C, N, freq, time)
        """
        B, C, N, F, T = x.shape
        batch = x.permute(0, 4, 1, 3, 2)  # [B, T, C, F, N]

        Q = (
            self.attn_conv_Q(batch)
            .reshape(B, T, C, F, -1, self.att_heads)
            .permute(0, 5, 1, 2, 3, 4)
            .contiguous()
        )  # [B, head, T, C, F, D]
        K = (
            self.attn_conv_K(batch)
            .reshape(B, T, C, F, -1, self.att_heads)
            .permute(0, 5, 1, 2, 3, 4)
            .contiguous()
        )  # [B, head, T, C, F, D]
        V = (
            self.attn_conv_V(batch)
            .reshape(B, T, C, F, -1, self.att_heads)
            .permute(0, 5, 1, 2, 3, 4)
            .contiguous()
        )  # [B, head, T, C, F, D']

        emb_dim = V.size(-2) * V.size(-1)
        attn_mat = torch.einsum("bhtcfn,bhtefn->bhce", Q / T, K / emb_dim**0.5)
        attn_mat = nn.functional.softmax(attn_mat, dim=-1)  # [B, head, C, C]
        V = torch.einsum("bhce,bhtefn->bhtcfn", attn_mat, V)  # [B, head, T, C, F, D']

        batch = torch.cat(V.unbind(dim=1), dim=-1)  # [B, T, C, F, D]
        batch = self["attn_concat_proj"](batch)  # [B, T, C, F, N]

        return batch.permute(0, 2, 4, 3, 1) + x