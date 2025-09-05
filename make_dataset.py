import torch
from torch.utils.data import Dataset
import tiktoken
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class GPTDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = self._prepare_data()

    def _prepare_data(self):
        all_text = " ".join([str(text) for text in self.texts])
        tokenizer = tiktoken.get_encoding("gpt2")
        tokens = tokenizer.encode(all_text)
        return tokens

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
