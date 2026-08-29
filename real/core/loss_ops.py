from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def causal_ce_per_sample(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, T, V]
    labels: [B, T] with -100 ignored
    returns: [B] mean CE over non-ignored tokens
    """
    B, T, V = logits.shape
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    flat = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )
    token_loss = flat.view(B, T - 1)
    token_mask = (shift_labels != -100).float()
    denom = token_mask.sum(dim=1).clamp(min=1.0)
    return (token_loss * token_mask).sum(dim=1) / denom


def apply_answer_loss_window(labels: torch.Tensor, answer_loss_tokens: Optional[int]) -> torch.Tensor:
    """Keep only the first N supervised (non -100) label tokens per sample.

    This is a masking helper for callers that keep full sequences but want a
    consistent "answer-start loss window" interpretation.

    Note: for denoise training we *also* truncate packed sequences to the window
    (see build_sft_example in train_real_v1_3.py). This helper is for cases
    where truncation is not possible/desirable.
    """
    if answer_loss_tokens is None:
        return labels
    keep_n = max(1, int(answer_loss_tokens))

    supervised = labels != -100
    sup_count = supervised.long().cumsum(dim=1)
    drop = supervised & (sup_count > keep_n)
    if not drop.any():
        return labels
    out = labels.clone()
    out[drop] = -100
    return out
