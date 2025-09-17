import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class VectorTransformerBlock(nn.Module):

    def __init__(self, seq_q, seq_k, seq_v, seq_d, embedding_dim):
        super().__init__()
        # Learn per-token importance scores that are independent of sequence length
        self.score_q = nn.Linear(embedding_dim, 1)
        self.score_k = nn.Linear(embedding_dim, 1)
        self.score_v = nn.Linear(embedding_dim, 1)
        self.score_d = nn.Linear(embedding_dim, 1)

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
        # Self-similarity weighting for Q and D
        Q = torch.matmul(F.softmax(torch.matmul(Q, Q.transpose(-2, -1)), dim=-1), self.layer1(Q))
        D = torch.matmul(F.softmax(torch.matmul(D, D.transpose(-2, -1)), dim=-1), self.layer4(D))

        # Content-based token weights (shape: [batch, seq, 1])
        w_q = F.softmax(self.score_q(Q).squeeze(-1), dim=-1).unsqueeze(-1)
        w_d = F.softmax(self.score_d(D).squeeze(-1), dim=-1).unsqueeze(-1)

        # Weighted sums into vectors of shape [batch, embedding_dim, 1]
        q_vec = torch.matmul(Q.transpose(-2, -1), w_q)
        d_vec = torch.matmul(D.transpose(-2, -1), w_d)

        # Masked self-similarity for K and V then linear projection
        K = torch.matmul(F.softmax(torch.matmul(K, K.transpose(-2, -1)) + mask1, dim=-1), self.layer2(K))
        V = torch.matmul(F.softmax(torch.matmul(V, V.transpose(-2, -1)) + mask2, dim=-1), self.layer3(V))

        # Content-based weights for K and V
        w_k = F.softmax(self.score_k(K).squeeze(-1), dim=-1).unsqueeze(-1)
        w_v = F.softmax(self.score_v(V).squeeze(-1), dim=-1).unsqueeze(-1)

        k_vec = torch.matmul(K.transpose(-2, -1), w_k)
        v_vec = torch.matmul(V.transpose(-2, -1), w_v)

        # Direction vectors, normalized; keep shape [batch, 1, embedding_dim]
        vector_to_gaussian_q = (d_vec - q_vec)
        vector_to_gaussian_k = (d_vec - k_vec)
        vector_to_gaussian_v = (d_vec - v_vec)

        # Normalize with numerical stability across embedding dimension
        eps = 1e-8
        vector_to_gaussian_q = vector_to_gaussian_q / (vector_to_gaussian_q.norm(dim=-2, keepdim=True) + eps)
        vector_to_gaussian_k = vector_to_gaussian_k / (vector_to_gaussian_k.norm(dim=-2, keepdim=True) + eps)
        vector_to_gaussian_v = vector_to_gaussian_v / (vector_to_gaussian_v.norm(dim=-2, keepdim=True) + eps)

        return vector_to_gaussian_q.transpose(-2, -1), vector_to_gaussian_k.transpose(-2, -1), vector_to_gaussian_v.transpose(-2, -1)


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
