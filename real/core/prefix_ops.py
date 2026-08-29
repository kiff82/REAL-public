from __future__ import annotations

from typing import Optional, Tuple

import torch


def clamp_prefix_norm(prefix: torch.Tensor, max_prefix_norm: float, *, eps: float = 1e-6) -> torch.Tensor:
    """Clamp per-prefix-vector L2 norm (no-op if max_prefix_norm<=0).

    Semantics match existing call sites in:
    - train_real_v1_3.py (training wrapper safety clamp)
    - scripts/bakeoff.py (optional clamp flag)
    """
    if max_prefix_norm is None or float(max_prefix_norm) <= 0:
        return prefix
    pn = prefix.norm(dim=-1, keepdim=True)
    scale = (float(max_prefix_norm) / (pn + eps)).clamp(max=1.0)
    return prefix * scale


def build_inputs_embeds_with_prefix(
    *,
    emb: torch.Tensor,  # [B,L,D]
    attention_mask: torch.Tensor,  # [B,L]
    labels: Optional[torch.Tensor] = None,  # [B,L]
    prefix: Optional[torch.Tensor] = None,  # [B,P,D]
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Concatenate a prefix onto token embeddings and extend masks/labels.

    - If `prefix` is None (or has P==0), returns inputs unchanged.
    - Prefix tokens are always fully attended (mask=1).
    - Prefix labels are always ignored (-100) when `labels` is provided.
    """
    if prefix is None or prefix.size(1) == 0:
        return emb, attention_mask, labels

    B, P, _D = prefix.shape
    prefix_attn = torch.ones((B, P), device=emb.device, dtype=attention_mask.dtype)

    inputs_embeds = torch.cat([prefix, emb], dim=1)
    attn2 = torch.cat([prefix_attn, attention_mask], dim=1)

    if labels is None:
        return inputs_embeds, attn2, None

    prefix_labels = torch.full((B, P), -100, device=emb.device, dtype=labels.dtype)
    labels2 = torch.cat([prefix_labels, labels], dim=1)
    return inputs_embeds, attn2, labels2
