import torch
import torch.nn as nn
from torch.nn import init


class FastKernelCompression(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.compression = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid()
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

        b, s, c = x.size()
        se = self.global_pool(x.transpose(1, 2)).view(b, c)
        se = self.compression(se).view(b, 1, c)
        return x * se