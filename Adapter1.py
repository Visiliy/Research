import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftSplit(nn.Module):

    def __init__(self, seq_len, n, embedding_dim):
        super().__init__()
        self.n = n
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        
        self.output_sizes = nn.Parameter(torch.ones(n) * (seq_len / n))
        
        self.segment_layers = nn.ModuleList()
        for i in range(n):
            output_size = max(1, round(self.output_sizes[i].item()))
            self.segment_layers.append(
                nn.Sequential(
                    nn.Linear(seq_len, output_size * 2),
                    nn.GELU(),
                    nn.Linear(output_size * 2, output_size)
                )
            )
        
        self.importance_weights = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, n),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        batch_size, seq_len, embedding_dim = x.shape
        
        importance = self.importance_weights(x.mean(dim=1))
        
        x_t = x.transpose(1, 2) 
        
        segments = []
        for i, layer in enumerate(self.segment_layers):
            compressed = layer(x_t)
            
            scale = importance[:, i].view(batch_size, 1, 1)
            compressed = compressed * scale
            
            segments.append(compressed.transpose(1, 2))
        
        return segments


class Attention(nn.Module):

    def __init__(self, embeddings_dim, heads):

        super().__init__()
        self.heads = heads
        self.embeddings_dim = embeddings_dim
        self.head_dim = embeddings_dim // heads

        self.query = nn.Linear(embeddings_dim, embeddings_dim)
        self.key = nn.Linear(embeddings_dim, embeddings_dim)
        self.value = nn.Linear(embeddings_dim, embeddings_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embeddings_dim, embeddings_dim * 4),
            nn.GELU(),
            nn.Linear(embeddings_dim * 4, embeddings_dim)
        )

        self.norm1 = nn.LayerNorm(embeddings_dim)
        self.norm2 = nn.LayerNorm(embeddings_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        batch_size, seq_len, embeddings_dim = x.shape
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)

        score = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            score =+ mask
        attention = F.softmax(score, dim=-1)
        
        output = torch.matmul(attention, V)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, embeddings_dim)
        output = self.dropout(output)

        output = self.norm1(output + x)
        output = self.ffn(output)
        return self.norm2(output + output)

class Adapter1(nn.Module):

    def __init__(self, d_model, heads, seq_len):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.seq_len = seq_len
        
        self.attention = Attention(d_model, heads)
        self.splitter = SoftSplit(seq_len, 4, d_model)

    def forward(self, x, mask=None):
        x = self.attention(x, mask)
        
        segments = self.splitter(x)
        
        return segments[0], segments[1], segments[2], segments[3]


if __name__ == "__main__":
    model = Adapter1(d_model=132, seq_len=100, heads=12)
    data = torch.randn((8, 100, 132))
    
    data_segment, reasoning, answer, query  = model(data)
    
    print("Original shape:", data.shape)
    print("Query shape:", query.shape)
    print("Data shape:", data_segment.shape)
    print("Reasoning shape:", reasoning.shape)
    print("Answer shape:", answer.shape)