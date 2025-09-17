import torch
import torch.nn as nn
from ConvBlock import DynamicPooling, LinearPerformerAttention, LinearParameterizationKernel, FastKernelCompression, LinearBlockSparseAttention, OptimizedDilatedResidual, TokenMerging, LinearLocalAttention, LinearDynamicInceptionBlock


class ConvBlock(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.dynamic_pooling = DynamicPooling(output_size=None)
        self.performer_attention = LinearPerformerAttention(dim, heads, feature_dim=256, dropout=dropout)
        self.parameterization_kernel = LinearParameterizationKernel(dim, dim, feature_dim=32)
        self.kernel_compression = FastKernelCompression(dim, reduction_ratio=4)
        self.block_sparse_attention = LinearBlockSparseAttention(dim, block_size=32, heads=heads, feature_dim=128, dropout=dropout)
        self.dilated_residual_cnn = OptimizedDilatedResidual(dim, dilations=[1, 2, 4, 8], dropout=dropout, use_glu=True)
        self.token_merging = TokenMerging(dim, reduction_ratio=1)
        self.local_attention = LinearLocalAttention(dim, window_size=7, heads=heads, feature_dim=64, dropout=dropout)
        self.dynamic_inception_block = LinearDynamicInceptionBlock(dim, kernel_sizes=[1, 3, 5], feature_dims=[32, 64, 32], dropout=dropout)

        self.inception_proj = nn.Linear((dim // 3) * 4, dim)

        self.final_norm = nn.LayerNorm(dim)
        
    def forward(self, x, mask: torch.Tensor = None):
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.squeeze(-1)
            mask = mask.to(torch.bool)

        x = self.dynamic_inception_block(x, mask=mask)
        x = self.inception_proj(x)
        x = self.local_attention(x, mask=mask)

        mask_padded = None
        reduced_mask = None
        if mask is not None:
            b, s = mask.shape
            r = self.token_merging.reduction_ratio
            pad_len = (r - (s % r)) % r
            if pad_len > 0:
                pad = torch.zeros((b, pad_len), dtype=torch.bool, device=mask.device)
                mask_padded = torch.cat([mask, pad], dim=1)
            else:
                mask_padded = mask
            new_s = mask_padded.shape[1] // r
            reduced_mask = mask_padded.view(b, new_s, r).any(dim=-1)

        x = self.token_merging(x, mask=mask_padded if mask_padded is not None else None)

        mask = reduced_mask

        x = self.dilated_residual_cnn(x, mask=mask if mask is None or mask.dim() == 3 else mask.unsqueeze(-1))
        x = self.block_sparse_attention(x, mask=mask if mask is None or mask.dim() == 3 else mask.unsqueeze(-1))
        x = self.kernel_compression(x, mask=mask if mask is None or mask.dim() == 3 else mask.unsqueeze(-1))
        x = self.parameterization_kernel(x, mask=mask if mask is None or mask.dim() == 3 else mask.unsqueeze(-1))
        x = self.performer_attention(x, mask=mask if mask is None or mask.dim() == 3 else mask.unsqueeze(-1))
        x = self.dynamic_pooling(x, mask=mask if mask is None or mask.dim() == 3 else mask.unsqueeze(-1))

        return self.final_norm(x)