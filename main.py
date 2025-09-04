import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from ThinkerTransformer import ThinkerTransformer


class MainModel(nn.Module):

    def __init__(self, seq_d, seq_q, seq_k, seq_v, d_model, heads, dropout):
        super().__init__()

        self.layer1 = nn.Linear(d_model, d_model)
        self.layer2 = nn.Linear(d_model, d_model)

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
        return Q, self.layer1(K), self.layer2(V), D


if __name__ == '__main__':
    model = MainModel(seq_d=4, seq_q=10, seq_k=12, seq_v=18, d_model=128, heads=8, dropout=0.1)
    Q = torch.randn(1, 10, 128)
    K = torch.randn(1, 12, 128)
    V = torch.randn(1, 18, 128)
    D = torch.randn(1, 4, 128)
    Q, K, V, D = model(Q, K, V, D)
    print(Q.shape)
    print(K.shape)
    print(V.shape)
    print(D.shape)
