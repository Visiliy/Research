import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import DataLoader, random_split
from amps_dataset import AmpsTextDataset

from ThinkerTransformer import ThinkerTransformer
from Adapter1 import Adapter1
from Adapter2 import Adapter2


class MainModel(nn.Module):

    def __init__(self, seq_d, seq_q, seq_k, seq_v, d_model, heads, dropout, output_dim):
        super().__init__()

        self.layer1 = nn.Linear(d_model, output_dim)
        self.layer2 = nn.Linear(d_model, output_dim)

        self.layer3 = nn.Linear(d_model, output_dim)
        self.layer4 = nn.Linear(d_model, output_dim)

        self.block_list = nn.ModuleList(
            ThinkerTransformer(seq_d=seq_d, seq_q=seq_q, seq_k=seq_k, seq_v=seq_v, d_model=d_model, heads=heads,
                               dropout=dropout) for _ in range(12))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, Q, K, V, D):
        for block in self.block_list:
            Q, K, V, D = block(Q, K, V, D)
        return self.layer1(Q), self.layer2(K), self.layer3(V), self.layer4(D)


class UnifiedModel(nn.Module):

    def __init__(self, seq_len: int, d_model: int, heads: int, dropout: float = 0.1, output_dim: int | None = None):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.heads = heads
        self.dropout = dropout
        self.output_dim = output_dim if output_dim is not None else d_model

        self.adapter1 = Adapter1(d_model=d_model, heads=heads, seq_len=seq_len)

        self.out_layer = nn.Linear(d_model, output_dim)

        self.layer1 = nn.Linear(output_dim, d_model)
        self.layer2 = nn.Linear(output_dim, d_model)
        self.layer3 = nn.Linear(output_dim, d_model)
        self.layer4 = nn.Linear(output_dim, d_model)

        self.main_model = None
        self.adapter2 = None

    def _lazy_build(self, segments):
        q_seg, k_seg, v_seg, d_seg = segments
        d_len = d_seg.size(1)
        k_len = k_seg.size(1)
        v_len = v_seg.size(1)
        q_len = q_seg.size(1)

        self.main_model = MainModel(
            seq_d=d_len,
            seq_q=q_len,
            seq_k=k_len,
            seq_v=v_len,
            d_model=self.d_model,
            heads=self.heads,
            dropout=self.dropout,
            output_dim=self.output_dim,
        )

        total_concat = d_len + k_len + v_len + q_len
        self.adapter2 = Adapter2(self.d_model, self.seq_len, total_concat)

    def forward(self, x, mask=None):
        seg0, seg1, seg2, seg3 = self.adapter1(x, mask)
        D, K, V, Q = seg0, seg1, seg2, seg3
        if self.main_model is None or self.adapter2 is None:
            self._lazy_build((Q, K, V, V))
        q_out, k_out, v_out, d_out = self.main_model(Q, K, V, D)
        y = self.adapter2([self.layer1(q_out), self.layer2(k_out), self.layer3(v_out), self.layer4(d_out)])
        return self.out_layer(y)


if __name__ == '__main__':
    pass