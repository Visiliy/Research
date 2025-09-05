import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attention_scores = attention_scores + mask
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = torch.matmul(attention_scores, v)
        attention_scores = attention_scores.transpose(1, 2).reshape(batch_size, seq_len, self.head_dim * self.n_heads)
        return self.out(attention_scores)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.attention = Attention(d_model, n_heads)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        d = self.attention(x, mask)
        d = self.dropout(d)
        d = self.layer_norm1(x + d)
        ffn = self.ffn(d)
        ffn = self.layer_norm2(ffn + d)
        return ffn
        


class Transformer(nn.Module):
    def __init__(self, layer, d_model, n_heads):
        super().__init__()
        self.layer = layer
        self.d_model = d_model
        self.n_heads = n_heads

        self.attention_layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads) for _ in range(n_heads)
        ])

        self.layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        for attention in self.attention_layers:
            x = attention(x, mask=mask)
        return self.layer(x)


if __name__ == "__main__":
    model = Transformer(layer=12, d_model=132, n_heads=12)
    array = torch.randn((8, 10, 132))
    print(model(array).shape)