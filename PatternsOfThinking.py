import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class PatternsOfThinkingBlock(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.layer = nn.Linear(seq_len, seq_len)
        self.gelu = nn.GELU()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        max_indices = torch.argmax(x, dim=-1, keepdim=True)

        max_values = torch.gather(x, -1, max_indices)

        transformed = self.layer(max_values.squeeze(-1))
        transformed = self.gelu(transformed)

        mask = F.one_hot(max_indices.squeeze(-1), num_classes=self.seq_len).float()
        inverted_mask = 1 - mask

        x = x * inverted_mask

        result = x + mask * transformed.unsqueeze(-1)
        return result


class PatternsOfThinking(nn.Module):
    def __init__(self, embedding_dim, heads, seq_len):
        super().__init__()
        self.head_dim = embedding_dim // heads
        self.heads = heads
        self.seq_len = seq_len

        self.d = nn.Dropout(0.1)
        self.patterns_of_thinking_block = PatternsOfThinkingBlock(seq_len)

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

        Q = Q.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            scores += mask

        scores = self.patterns_of_thinking_block(scores)

        attention = F.softmax(scores, dim=-1)
        result = torch.matmul(attention, V)

        result = result.transpose(1, 2).reshape(batch_size, seq_len, embedding_dim)
        result = self.d(result)

        result = self.ffn(result)
        result = self.layer_norm(result + x)

        return result


if __name__ == "__main__":
    model = PatternsOfThinking(seq_len=10, heads=12, embedding_dim=132)
    x = torch.randn((8, 10, 132))

    output = model(x)
    loss = output.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"Нет градиентов для {name}")
        else:
            print(f"Градиенты для {name}: {param.grad.abs().sum().item()}")

    print(output.shape)
