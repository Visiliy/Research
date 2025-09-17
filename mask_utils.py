import torch


def build_causal_mask(seq_q: int, seq_k: int, device) -> torch.Tensor:
    i = torch.arange(seq_q, device=device).unsqueeze(1)
    j = torch.arange(seq_k, device=device).unsqueeze(0)
    m = (j <= i).to(torch.float32)
    return m.masked_fill(m == 0, float('-inf')).masked_fill(m == 1, 0.0)


