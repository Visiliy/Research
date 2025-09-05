import torch
import torch.nn as nn
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
            self.w_array2.append(nn.Linear(diagonal_length, diagonal_length))

        del self.w_array2[-1]

        
    def forward(self, x):
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
        return x


if __name__ == "__main__":
    model = DiagonalTraining(seq_len=10)
    x = torch.randn((8, 10, 10))
    print(model(x).shape)
