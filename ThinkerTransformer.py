import torch
import torch.nn as nn
from torch.nn import init

from ConvBlockModel import ConvBlock
from FFTBlock import SPECTREFFTBlock
from LowRankApproximation import AdaptiveLowRank
from GaussianNoise import GaussianNoiseBlock
from UnificationBlock import UnificationBlock
from VectorTransformerBlock import VectorTransformerBlock
from PatternsOfThinking import PatternsOfThinking


class ThinkerTransformer(nn.Module):

    def __init__(self, seq_d, seq_q, seq_k, seq_v, d_model: int = 512, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.seq_d = seq_d
        self.seq_q = seq_q
        self.seq_k = seq_k
        self.seq_v = seq_v

        self.d1 = nn.Dropout(0.1)
        self.d2 = nn.Dropout(0.1)
        self.d3 = nn.Dropout(0.1)

        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

        self.ffn2 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

        self.ffn3 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

        self.ffn4 = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

        self.conv_block1 = ConvBlock(d_model, heads=heads, dropout=dropout)
        self.conv_block2 = ConvBlock(d_model, heads=heads, dropout=dropout)
        self.conv_block3 = ConvBlock(d_model, heads=heads, dropout=dropout)
        self.conv_block4 = ConvBlock(d_model, heads=heads, dropout=dropout)

        self.fft_block1 = SPECTREFFTBlock(d_model=d_model, num_heads=heads, dropout=dropout)
        self.fft_block2 = SPECTREFFTBlock(d_model=d_model, num_heads=heads, dropout=dropout)
        self.fft_block3 = SPECTREFFTBlock(d_model=d_model, num_heads=heads, dropout=dropout)
        self.fft_block4 = SPECTREFFTBlock(d_model=d_model, num_heads=heads, dropout=dropout)
        self.fft_block5 = SPECTREFFTBlock(d_model=d_model, num_heads=heads, dropout=dropout)

        self.lowrank_block1 = AdaptiveLowRank(embedding_dim=d_model)
        self.lowrank_block2 = AdaptiveLowRank(embedding_dim=d_model)
        self.lowrank_block3 = AdaptiveLowRank(embedding_dim=d_model)
        self.lowrank_block4 = AdaptiveLowRank(embedding_dim=d_model)
        self.lowrank_block5 = AdaptiveLowRank(embedding_dim=d_model)

        self.gaussian_block1 = GaussianNoiseBlock(embedding_dim=d_model)
        self.gaussian_block2 = GaussianNoiseBlock(embedding_dim=d_model)
        self.gaussian_block3 = GaussianNoiseBlock(embedding_dim=d_model)
        self.gaussian_block4 = GaussianNoiseBlock(embedding_dim=d_model)
        self.gaussian_block5 = GaussianNoiseBlock(embedding_dim=d_model)

        self.vector_transformer = VectorTransformerBlock(embedding_dim=d_model, seq_d=seq_d, seq_k=seq_k, seq_v=seq_v,
                                                         seq_q=seq_q)
        self.unification_block1 = UnificationBlock(d_model=d_model, heads=heads)
        self.unification_block2 = UnificationBlock(d_model=d_model, heads=heads)

        self.patterns_of_thinking1 = PatternsOfThinking(heads=heads, embedding_dim=d_model, seq_len=seq_q)
        self.patterns_of_thinking2 = PatternsOfThinking(heads=heads, embedding_dim=d_model, seq_len=seq_k)
        self.patterns_of_thinking3 = PatternsOfThinking(heads=heads, embedding_dim=d_model, seq_len=seq_v)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, Q, K, V, D):
        batch = Q.size(0)
        device = Q.device

        k_mask = torch.ones(batch, K.size(1), device=device, dtype=torch.bool)
        v_mask = torch.ones(batch, V.size(1), device=device, dtype=torch.bool)

        mask2 = torch.tril(torch.ones(self.seq_k, self.seq_k, device=device))
        mask2 = mask2.masked_fill(mask2 == 0, float('-inf')).masked_fill(mask2 == 1, 0.0)

        mask3 = torch.tril(torch.ones(self.seq_v, self.seq_v, device=device))
        mask3 = mask3.masked_fill(mask3 == 0, float('-inf')).masked_fill(mask3 == 1, 0.0)

        Q = self.conv_block1(Q)
        K = self.conv_block2(K, mask=k_mask)
        k_mask = torch.ones(batch, K.size(1), device=device, dtype=torch.bool)

        V = self.conv_block3(V, mask=v_mask)
        v_mask = torch.ones(batch, V.size(1), device=device, dtype=torch.bool)
        D = self.conv_block4(D)

        Q = self.fft_block1(Q)
        K = self.fft_block2(K, mask=k_mask)
        V = self.fft_block3(V, mask=v_mask)

        Q, _, _ = self.lowrank_block1(Q)
        K, _, _ = self.lowrank_block2(K, mask=k_mask)
        V, _, _ = self.lowrank_block3(V, mask=v_mask)

        external_directions = self.vector_transformer(Q, K, V, D, mask1=mask2, mask2=mask3)

        Q = self.gaussian_block1(Q, external_direction=external_directions[0], apply_noise=True)
        K = self.gaussian_block2(K, external_direction=external_directions[1], apply_noise=True, mask=k_mask)
        V = self.gaussian_block3(V, external_direction=external_directions[2], apply_noise=True, mask=v_mask)

        Q = self.patterns_of_thinking1(Q)
        K = self.patterns_of_thinking2(K, mask=mask2)
        V = self.patterns_of_thinking3(V, mask=mask3)

        K = self.unification_block1(K, Q, mask=mask2)
        K = self.fft_block4(K, mask=k_mask)
        K, _, _ = self.lowrank_block4(K, mask=k_mask)
        K = self.gaussian_block4(K, mask=k_mask)

        V = self.unification_block2(V, K, mask=mask3)
        V = self.fft_block5(V, mask=v_mask)
        V, _, _ = self.lowrank_block5(V, mask=v_mask)
        V = self.gaussian_block5(V, mask=v_mask)

        Q = self.ffn1(self.d1(Q))
        K = self.ffn2(self.d2(K))
        V = self.ffn3(self.d3(V))
        D = self.ffn4(D)

        return Q, K, V, D
