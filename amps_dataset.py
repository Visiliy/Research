import os
import glob
import math
from typing import List, Tuple
import torch
import torch.nn as nn


class AmpsTextDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, seq_len: int, d_model: int, file_glob: str = "**/*.txt"):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.d_model = d_model
        self.file_paths = sorted(glob.glob(os.path.join(self.root_dir, file_glob), recursive=True))
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.stoi = {self.pad_token: 0, self.unk_token: 1}
        self.itos = [self.pad_token, self.unk_token]
        self.token_sequences = []
        for p in self.file_paths:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            tokens = self._tokenize(text)
            self._extend_vocab(tokens)
            self.token_sequences.append(tokens)
        self.vocab_size = len(self.itos)
        self.word_embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=0)
        self.word_embedding.weight.data.normal_(mean=0.0, std=0.02)

    def _tokenize(self, text: str) -> List[str]:
        return text.strip().split()

    def _extend_vocab(self, tokens: List[str]) -> None:
        for t in tokens:
            if t not in self.stoi:
                self.stoi[t] = len(self.itos)
                self.itos.append(t)

    def __len__(self) -> int:
        return len(self.token_sequences)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.token_sequences[index]
        token_ids = [self.stoi.get(t, self.stoi[self.unk_token]) for t in tokens]
        if len(token_ids) >= self.seq_len:
            token_ids = token_ids[: self.seq_len]
        else:
            token_ids = token_ids + [self.stoi[self.pad_token]] * (self.seq_len - len(token_ids))
        target_ids = token_ids[1:] + [self.stoi[self.pad_token]]
        token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
        target_ids_tensor = torch.tensor(target_ids, dtype=torch.long)
        with torch.no_grad():
            word_emb = self.word_embedding(token_ids_tensor)
            pos_emb = self._sinusoidal_positional_encoding(self.seq_len, self.d_model, word_emb.device)
            x = word_emb + pos_emb
        return x, target_ids_tensor

    def _sinusoidal_positional_encoding(self, seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
        position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model, dtype=torch.float32, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


def create_amps_dataloader(root_dir: str, seq_len: int, d_model: int, batch_size: int, shuffle: bool = True) -> Tuple[AmpsTextDataset, torch.utils.data.DataLoader]:
    dataset = AmpsTextDataset(root_dir=root_dir, seq_len=seq_len, d_model=d_model)
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = [b[0] for b in batch]
        ys = [b[1] for b in batch]
        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataset, loader


