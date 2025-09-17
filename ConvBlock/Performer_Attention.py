import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class LinearPerformerAttention(nn.Module):
    def __init__(self, dim, heads=8, feature_dim=256, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.feature_dim = feature_dim
        self.head_dim = dim // heads

        self.proj_matrix = nn.Parameter(torch.randn(heads, self.head_dim, feature_dim))
        nn.init.orthogonal_(self.proj_matrix)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x, mask: torch.Tensor = None):
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            x = x * mask.to(x.dtype)

        b, n, d = x.shape
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)

        q_proj = torch.einsum('bhnd,hdf->bhnf', q, self.proj_matrix)
        k_proj = torch.einsum('bhnd,hdf->bhnf', k, self.proj_matrix)

        q_proj = F.elu(q_proj) + 1
        k_proj = F.elu(k_proj) + 1

        k_v = torch.einsum('bhnf,bhnd->bhfd', k_proj, v)
        attention_out = torch.einsum('bhnf,bhfd->bhnd', q_proj, k_v)

        k_proj_sum = k_proj.sum(dim=2, keepdim=True)
        z = 1.0 / (torch.einsum('bhnf,bhf->bhn', q_proj, k_proj_sum.squeeze(2)) + 1e-8)
        attention_out = attention_out * z.unsqueeze(-1)

        attention_out = attention_out.transpose(1, 2).reshape(b, n, -1)
        out = self.to_out(attention_out)

        if 'mask' in locals() and mask is not None:
            if mask.shape[0] == out.shape[0] and mask.shape[1] == out.shape[1]:
                out = out * mask.to(out.dtype)
        # Causality is enforced upstream via masks and local attention windows

        return out
