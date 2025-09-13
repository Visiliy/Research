import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicPooling(nn.Module):
    def __init__(self, output_size=None, mode='adaptive'):
        super().__init__()
        self.output_size = output_size
        self.mode = mode

        if mode == 'adaptive':
            self.pool = nn.AdaptiveAvgPool1d(output_size)
        elif mode == 'learnable':
            self.alpha = nn.Parameter(torch.tensor(0.5))
        elif mode == 'identity':
            pass

    def forward(self, x, mask: torch.Tensor = None):
        if self.mode == 'identity':
            return x

        x = x.transpose(1, 2)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            x = x * mask.transpose(1, 2).to(x.dtype)
        if self.mode == 'adaptive':
            if self.output_size is None:
                seq_len = x.size(2)
                pool_size = max(8, seq_len // 4)
                pooled = F.adaptive_avg_pool1d(x, pool_size)
            else:
                pooled = self.pool(x)
        elif self.mode == 'learnable':
            if self.output_size is None:
                seq_len = x.size(2)
                pool_size = max(8, seq_len // 4)
            else:
                pool_size = self.output_size
            avg_pool = F.adaptive_avg_pool1d(x, pool_size)
            max_pool = F.adaptive_max_pool1d(x, pool_size)
            pooled = self.alpha * avg_pool + (1 - self.alpha) * max_pool
        else:
            seq_len = x.size(2)
            if self.output_size is None:
                pool_size = max(8, seq_len // 4)
                pooled = F.adaptive_avg_pool1d(x, pool_size)
            else:
                pooled = F.adaptive_avg_pool1d(x, self.output_size)

        return pooled.transpose(1, 2)
