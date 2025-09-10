import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as ffn


class Adapter2(nn.Module):

    def __init__(self, embedding_dim, seq_len, n):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len

        self.layer = nn.Linear(n, seq_len)
        
        self.seq_ffn = nn.Sequential(
            nn.Linear(seq_len, seq_len * 4),
            nn.GELU(),
            nn.Linear(seq_len * 4, seq_len)
        )

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4), 
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )

        self.norm = nn.LayerNorm(embedding_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, data_tensor):
        data_tensor = [x.transpose(-2, -1) for x in data_tensor]
        data = torch.cat(data_tensor, dim=-1)

        new_data = self.layer(data)
        new_data = self.seq_ffn(new_data).transpose(-2, -1)
        new_data = self.norm(new_data)

        return self.ffn(new_data)


if __name__ == "__main__":
    model = Adapter2(132, 16, 18)

    data1 = torch.randn(2, 6, 132)
    data2 = torch.randn(2, 4, 132)
    data3 = torch.randn(2, 4, 132)
    data4 = torch.randn(2, 4, 132)

    print(model([data1, data2, data3, data4]).shape)