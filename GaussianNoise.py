import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianNoiseBlock(nn.Module):

    def __init__(self, embedding_dim: int, init_beta: float = 0.5, init_n: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.model_direction = nn.Parameter(torch.randn(embedding_dim))

        self.beta_param = nn.Parameter(torch.tensor(float(init_beta)))

        self.n_param = nn.Parameter(torch.tensor(float(init_n)))

        self.eps = 1e-8

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        return x / (norm + self.eps)

    def _broadcast_external(self, external_direction: torch.Tensor, target_shape, device):
        if external_direction is None:
            return None

        ext = external_direction.to(device)
        if ext.dim() == 1 and ext.size(-1) == self.embedding_dim:
            ext = ext.view(1, 1, -1)
        elif ext.dim() == 2 and ext.size(-1) == self.embedding_dim:
            ext = ext.unsqueeze(1)
        elif ext.dim() == 3 and ext.size(-1) == self.embedding_dim:
            pass
        else:
            raise ValueError(f"external_direction must have last dim == {self.embedding_dim}")

        ext = ext.expand(target_shape)
        return ext

    def forward(self, embeddings: torch.Tensor, external_direction: torch.Tensor = None,
                return_stats: bool = False, apply_noise: bool = True, mask: torch.Tensor = None):
        if embeddings.dim() != 3 or embeddings.size(-1) != self.embedding_dim:
            raise ValueError(f"embeddings must be (batch, seq_len, {self.embedding_dim})")

        batch, seq_len, dim = embeddings.shape
        device = embeddings.device

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            embeddings = embeddings * mask.to(embeddings.dtype)

        ext_dir = self._broadcast_external(external_direction, (batch, seq_len, dim), device)
        if ext_dir is not None and mask is not None:
            ext_dir = ext_dir * mask.to(ext_dir.dtype)

        model_dir = self._normalize(self.model_direction)
        model_dir = model_dir.view(1, 1, dim).expand(batch, seq_len, dim)

        if ext_dir is None:
            mixed = model_dir
        else:
            ext_norm = self._normalize(ext_dir)
            beta = torch.sigmoid(self.beta_param)
            mixed = beta * model_dir + (1.0 - beta) * ext_norm

        mixed_unit = self._normalize(mixed)

        n = F.softplus(self.n_param)

        if not apply_noise:
            return embeddings

        magnitudes = torch.randn(batch, seq_len, 1, device=device) * n
        noise = magnitudes * mixed_unit
        out = embeddings + noise
        return out

