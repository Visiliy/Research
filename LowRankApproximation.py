import torch
import torch.nn as nn
from torch.nn import init


class AdaptiveLowRank(nn.Module):
    def __init__(self, embedding_dim, min_rank=1, max_rank=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.min_rank = min_rank
        self.max_rank = max_rank or embedding_dim

        self.rank_weights = nn.Parameter(torch.zeros(self.max_rank - self.min_rank + 1))
        self.basis_matrices = nn.ModuleList([
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
            for _ in range(self.min_rank, self.max_rank + 1)
        ])
        self.output_norm = nn.LayerNorm(self.embedding_dim)

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

        batch_size, seq_len, emb_dim = x.shape

        probs = torch.softmax(self.rank_weights, dim=0)
        expected_rank = torch.sum(probs * torch.arange(self.min_rank, self.max_rank + 1).float().to(x.device))

        reconstructions = []
        for i, module in enumerate(self.basis_matrices):
            rank = self.min_rank + i
            recon = module(x)
            reconstructions.append(recon.unsqueeze(-1))

        all_recon = torch.cat(reconstructions, dim=-1)
        output = torch.einsum('bsed,d->bse', all_recon, probs)

        output = self.output_norm(output)

        rank_entropy = -torch.sum(probs * torch.log(probs + 1e-8))

        return output, expected_rank, rank_entropy
