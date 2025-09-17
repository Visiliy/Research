import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from mask_utils import build_causal_mask


class UnificationBlock(nn.Module):

    def __init__(self, d_model, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = d_model // self.heads
        self.transformation1 = nn.Linear(d_model, d_model)
        self.transformation2 = nn.Linear(d_model, d_model)
        self.transformation3 = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

        self.dropout = nn.Dropout(0.1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)


    def forward(self, Q, K, mask=None):
        Q1 = self.transformation1(Q)
        Q2 = self.transformation2(Q)
        K = self.transformation3(K)

        batch_size1, seq_q, _ = Q1.shape
        batch_size2, seq_k, _ = K.shape

        Q1 = Q1.view(batch_size1, seq_q, self.heads, self.head_dim)
        Q2 = Q2.view(batch_size1, seq_q, self.heads, self.head_dim)
        K = K.view(batch_size2, seq_k, self.heads, self.head_dim)

        Q1 = Q1.transpose(1, 2)
        Q2 = Q2.transpose(1, 2)
        K = K.transpose(1, 2)

        result = torch.matmul(Q1, K.transpose(-2, -1))
        seq_q = result.size(-2)
        seq_k = result.size(-1)
        causal = build_causal_mask(seq_q, seq_k, result.device)
        result = result + causal
        result = torch.matmul(result, result.transpose(-2, -1))

        if mask is not None:
            m = mask
            if m.dim() == 2:
                m = m.unsqueeze(0).unsqueeze(0)
            elif m.dim() == 3:
                m = m.unsqueeze(1)
            result = result + m.to(result.dtype)

        result = F.softmax(result, dim=-1)
        result = torch.matmul(result, Q2)

        result = result.transpose(1, 2)

        result = result.reshape(batch_size1, seq_q, self.head_dim * self.heads)

        result = self.dropout(result)
        result = self.ffn(result)

        return result
    

if __name__ == "__main__":
    model = UnificationBlock(d_model=132, heads=12)
    Q = torch.randn((2, 20, 132))
    K = torch.randn((2, 15, 132))
    
    res = model(K, Q)
    print(res.shape)