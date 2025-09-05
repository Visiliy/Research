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
            self.w_array1.append(nn.Linear(i+1, i+1))
            self.w_array2.append(nn.Linear(i+1, i+1))

        del self.w_array2[-1]

        
    def forward(self, x):
        for i in range(self.seq_len):
            array_w = []
            l = i
            r = 0
            while r < i:
                array_w.append(x[:, r, l])
                r += 1
                l -= 1
            array_w = torch.tensor(array_w)
            array_w= self.w_array1[i](array_w)
            array_w = array_w.tolist()[::-1]

            l = i
            r = 0
            while r < i:
                x[:, r, l] = array_w[r]
                r += 1
                l -= 1
        return x


if __name__ == "__main__":
    model = DiagonalTraining(seq_len=10)
    x = torch.randn((1, 10, 10))
    print(model(x).shape)
