import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class DiagonalTraining(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.w_array1 = nn.ModuleList()
        self.w_array2 = nn.ModuleList()

        for i in range(self.seq_len):
            diagonal_length = i + 1
            self.w_array1.append(nn.Linear(diagonal_length, diagonal_length))

        for i in range(1, self.seq_len):
            diagonal_length = self.seq_len - i
            self.w_array2.append(nn.Linear(diagonal_length, diagonal_length))
    
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        batch_size = x.shape[0]

        for i in range(self.seq_len):
            array_w = []
            l = i
            r = 0
            while r <= i and l >= 0:
                array_w.append(x[:, r, l])
                r += 1
                l -= 1
            
            if len(array_w) == 0:
                continue
                
            array_w = torch.stack(array_w, dim=1)
            array_w = self.w_array1[i](array_w)
            
            l = i
            r = 0
            idx = 0
            while r <= i and l >= 0 and idx < array_w.shape[1]:
                x[:, r, l] = array_w[:, idx]
                r += 1
                l -= 1
                idx += 1

        for i in range(1, self.seq_len):
            array_w = []
            l = self.seq_len - 1
            r = i
            while r < self.seq_len and l >= i:
                array_w.append(x[:, r, l])
                r += 1
                l -= 1
            
            if len(array_w) == 0:
                continue
                
            array_w = torch.stack(array_w, dim=1)
            array_w = self.w_array2[i-1](array_w)
            
            l = self.seq_len - 1
            r = i
            idx = 0
            while r < self.seq_len and l >= i and idx < array_w.shape[1]:
                x[:, r, l] = array_w[:, idx]
                r += 1
                l -= 1
                idx += 1

        return x
    

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

        self.layer_norm = nn.LayerNorm(embedding_dim, embedding_dim)

        self.layer1 = nn.Linear(embedding_dim, embedding_dim)
        self.layer2 = nn.Linear(embedding_dim, embedding_dim)
        self.layer3 = nn.Linear(embedding_dim, embedding_dim)

        self.diagonal_training = DiagonalTraining(seq_len=seq_len)

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
        result = F.softmax(result)
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