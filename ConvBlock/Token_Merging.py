import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class TokenMerging(nn.Module):
    def __init__(self, dim, reduction_ratio=2):
        super().__init__()
        self.reduction_ratio = reduction_ratio
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim * reduction_ratio, dim)

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

        b, s, d = x.shape
        if s % self.reduction_ratio != 0:
            pad_len = self.reduction_ratio - (s % self.reduction_ratio)
            x = F.pad(x, (0, 0, 0, pad_len, 0, 0))
            s = s + pad_len

        x = x.view(b, s // self.reduction_ratio, self.reduction_ratio * d)
        x = self.linear(x)
        return self.norm(x)