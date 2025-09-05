import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
import pandas as pd
import torch
from torch.nn import init
from torch.utils.data import DataLoader
import tiktoken
import torch.optim as optim
from make_dataset import PositionalEncoding, GPTDataset


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

        self._initialize_weights()

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

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)


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

        self._initialize_weights()

    def forward(self, x, mask=None):
        d = self.attention(x, mask)
        d = self.dropout(d)
        d = self.layer_norm1(x + d)
        ffn = self.ffn(d)
        ffn = self.layer_norm2(ffn + d)
        return ffn

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)


class Transformer(nn.Module):
    def __init__(self, layers, d_model, n_heads, output_dim):
        super().__init__()
        self.layers = layers
        self.d_model = d_model
        self.n_heads = n_heads

        self.attention_layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads) for _ in range(layers)
        ])

        self.layer = nn.Linear(d_model, output_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x, mask=None):
        for attention in self.attention_layers:
            x = attention(x, mask=mask)
        return self.layer(x)


if __name__ == "__main__":
    ds = load_dataset("codesignal/sms-spam-collection")
    df = pd.DataFrame(ds['train'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    texts = df["message"].tolist()

    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab
    embedding_dim = 512
    block_size = 256
    batch_size = 32

    mask = torch.tril(torch.ones(block_size, block_size, device=device))
    mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
    mask = mask.view(1, 1, batch_size, block_size)

    embedding_layer = nn.Embedding(vocab_size, embedding_dim).to(device)
    pos_encoder = PositionalEncoding(embedding_dim).to(device)

    dataset = GPTDataset(texts, tokenizer, block_size=block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Transformer(layers=12, d_model=512, n_heads=16, output_dim=vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10


    def prepare_batch(batch):
        x, y = batch
        token_embeddings = embedding_layer(x)
        embeddings_with_pos = pos_encoder(token_embeddings)

        return embeddings_with_pos, y


    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            x, y = prepare_batch(batch)
            x = x.to(device)
            y = y.to(device)

            predict = model(x, mask=mask)
            predict = predict.view(-1, vocab_size)
            y = y.view(-1)

            loss = F.cross_entropy(predict, y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, loss: {total_loss / len(dataloader)}")
