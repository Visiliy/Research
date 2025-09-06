import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class PatternsOfThinkingBlock(nn.Module):

    def __init__(self, seq_len):
        super().__init__()

        self.seq_len = seq_len

        self.layer = nn.Linear(seq_len, seq_len)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        x2 = F.softmax(x, dim=-1)
        array = []
        for i in range(self.seq_len):
            max_num = x[:, :, i].max()
            array.append(max_num)
        array = torch.stack(array)
        print(array.shape)


class DiagonalBlock(nn.Module):

    def __init__(self, embedding_dim, heads, seq_len):
        super().__init__()

        self.head_dim = embedding_dim // heads
        self.heads = heads

        self.d = nn.Dropout(0.1)

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )

        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.layer1 = nn.Linear(embedding_dim, embedding_dim)
        self.layer2 = nn.Linear(embedding_dim, embedding_dim)
        self.layer3 = nn.Linear(embedding_dim, embedding_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x, mask=None):
        batch_size, seq_len, embedding_dim = x.shape

        Q = self.layer1(x)
        K = self.layer2(x)
        V = self.layer3(x)

        Q = Q.view(batch_size, seq_len, self.heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        result = torch.matmul(Q, K.transpose(-2, -1))
        if mask is not None:
            result += mask
        result = result / (embedding_dim ** 0.5)
        result = F.softmax(result, dim=-1)
        result = self.diagonal_training(result)
        result = torch.matmul(result, V)

        result = result.transpose(1, 2)
        result = result.reshape(batch_size, seq_len, self.head_dim * self.heads)

        result = self.d(result)
        result = self.ffn(result)
        return self.layer_norm(result)


if __name__ == "__main__":
    model = DiagonalBlock(seq_len=10, heads=12, embedding_dim=132)
    x = torch.randn((8, 10, 132))
    output = model(x)
    print("Output shape:", output.shape)
