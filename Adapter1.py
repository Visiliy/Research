import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, embeddings_dim, heads):
        self.heads = heads
        self.embeddings_dim = embeddings_dim
        self.head_dim = embeddings_dim // heads

        self.layer1 = nn.Linear(embeddings_dim, embeddings_dim)
        self.layer2 = nn.Linear(embeddings_dim, embeddings_dim)
        self.layer3 = nn.Linear(embeddings_dim, embeddings_dim)


        self.ffn = nn.Sequential(
            nn.Linear(embeddings_dim, embeddings_dim * 4),
            nn.GELU(),
            nn.Linear(embeddings_dim * 4, embeddings_dim)
        )

        self.norm1 = LayerNorm(embeddings_dim)
        self.norm2 = LayerNorm(embeddings_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        batch_size, seq_len, embeddings_dim = x.shape
        Q = self.layer1(x)
        K = self.layer2(x)
        V = self.layer3(x)

        Q = Q.view(batch_size, seq_len, self.heads, self.head_dim).tarnspose(1, 2)
        K = K.view(batch_size, seq_len, self.heads, self.head_dim).tarnspose(1, 2)
        V = V.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)

        score = torch.matmul(Q, K.tarnspose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            score += mask
        score = torch.matmul(F.softmax(score, dim=-1), V)
        score = score.transpose(1, 2).reshape(batch_size, seq_len, self.head_dim * self.heads)
        score = self.dropout(score)

        result = self.norm1(score + x)
        result2 = self.ffn(result)

        return self.norm2(result2 + result)


class Adapter1(nn.Module):

    def __init__(self, d_model, heads, layers):
        self.d_model = d_model
        self.heads = heads
        self.layers = layers

        self.layer = nn.Linear(d_model, d_model)