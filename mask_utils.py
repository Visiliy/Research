import torch


def build_causal_mask(seq_q: int, seq_k: int, device) -> torch.Tensor:
    """Return lower-triangular causal mask shaped [seq_q, seq_k].

    Positions can only attend to keys at positions <= query index.
    The mask uses 0.0 for allowed and -inf for disallowed positions.
    """
    i = torch.arange(seq_q, device=device).unsqueeze(1)
    j = torch.arange(seq_k, device=device).unsqueeze(0)
    m = (j <= i).to(torch.float32)
    return m.masked_fill(m == 0, float('-inf')).masked_fill(m == 1, 0.0)


