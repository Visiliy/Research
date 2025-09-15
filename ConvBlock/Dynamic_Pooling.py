import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicPooling(nn.Module):
    def __init__(self, output_size=None):
        super().__init__()
        self.output_size = output_size
        self.pool = nn.AdaptiveAvgPool1d(output_size)

    def forward(self, x, mask: torch.Tensor = None):
        x = x.transpose(1, 2)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            x = x * mask.transpose(1, 2).to(x.dtype)
        if self.output_size is None:
            seq_len = x.size(2)
            pool_size = max(8, seq_len // 4)
            pooled = F.adaptive_avg_pool1d(x, pool_size)
        else:
            pooled = self.pool(x)

        return pooled.transpose(1, 2)
