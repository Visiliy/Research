import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class VectorTransformerBlock(nn.Module):

    def __init__(self, seq_q, seq_k, seq_v, seq_d, embedding_dim):
        super().__init__()
        self.matrix1 = nn.Parameter(torch.randn((seq_q, 1)))
        self.matrix2 = nn.Parameter(torch.randn((seq_k, 1)))
        self.matrix3 = nn.Parameter(torch.randn((seq_v, 1)))
        self.matrix4 = nn.Parameter(torch.randn((seq_d, 1)))

        self.layer1 = nn.Linear(embedding_dim, embedding_dim)
        self.layer2 = nn.Linear(embedding_dim, embedding_dim)
        self.layer3 = nn.Linear(embedding_dim, embedding_dim)
        self.layer4 = nn.Linear(embedding_dim, embedding_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, Q, K, V, D, mask1, mask2):
        Q = torch.matmul(F.softmax(torch.matmul(Q, Q.transpose(-2, -1))), self.layer1(Q))
        D = torch.matmul(F.softmax(torch.matmul(D, D.transpose(-2, -1))), self.layer4(D))

        q_vec = torch.matmul(Q.transpose(-2, -1), self.matrix1)
        d_vec = torch.matmul(D.transpose(-2, -1), self.matrix4)

        K = torch.matmul(F.softmax(torch.matmul(K, K.transpose(-2, -1)) + mask1), self.layer2(K))
        V = torch.matmul(F.softmax(torch.matmul(V, V.transpose(-2, -1)) + mask2), self.layer3(V))

        k_vec = torch.matmul(K.transpose(-2, -1), self.matrix2)
        v_vec = torch.matmul(V.transpose(-2, -1), self.matrix3)

        vector_to_gaussian_q = (d_vec - q_vec) / ((d_vec - q_vec).norm() + 1e-8)
        vector_to_gaussian_k = (d_vec - k_vec) / ((d_vec - k_vec).norm() + 1e-8)
        vector_to_gaussian_v = (d_vec - v_vec) / ((d_vec - v_vec).norm() + 1e-8)

        return vector_to_gaussian_q.transpose(-2, -1), vector_to_gaussian_k.transpose(-2,
                                                                                      -1), vector_to_gaussian_v.transpose(
            -2, -1)


if __name__ == "__main__":
    model = VectorTransformerBlock(seq_q=10, seq_k=12, seq_v=18, seq_d=4, embedding_dim=132)
    Q = torch.randn(1, 10, 132)
    K = torch.randn(1, 12, 132)
    V = torch.randn(1, 18, 132)
    D = torch.randn(1, 4, 132)
    mask2 = torch.tril(torch.ones(12, 12, device="cpu"))
    mask2 = mask2.masked_fill(mask2 == 0, float('-inf')).masked_fill(mask2 == 1, 0.0)

    mask3 = torch.tril(torch.ones(18, 18, device="cpu"))
    mask3 = mask3.masked_fill(mask3 == 0, float('-inf')).masked_fill(mask3 == 1, 0.0)
    vec1, vec2, vec3 = model(Q, K, V, D, mask2, mask3)
    print(vec1.shape)
    print(vec2.shape)
    print(vec3.shape)
