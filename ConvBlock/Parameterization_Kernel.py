import torch
import torch.nn as nn
from torch.nn import init


class LinearParameterizationKernel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, feature_dim=32):
        super().__init__()
        self.feature_dim = feature_dim

        self.U = nn.Linear(in_channels, feature_dim, bias=False)
        self.V = nn.Linear(feature_dim, out_channels, bias=False)
        self.activation = nn.GELU()

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

        x = self.U(x)
        x = self.activation(x)
        x = self.V(x)
        return x
