from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from real.core.prefix_ops import build_inputs_embeds_with_prefix


def average_candidate_logprob(
    *,
    backbone,
    prompt_input_ids: torch.Tensor,
    prefix: torch.Tensor,
    candidate_ids: torch.Tensor,
    model_dtype: Optional[torch.dtype] = None,
    checkpoint_backbone: bool = False,
) -> torch.Tensor:
    """Average log probability of a candidate continuation under a REAL prefix.

    This is the differentiable counterpart to the trace-probe candidate scorer:
    the backbone stays frozen, but gradients can flow through `prefix` to the
    REAL head.
    """
    if prompt_input_ids.ndim != 2 or candidate_ids.ndim != 2:
        raise ValueError("prompt_input_ids and candidate_ids must be rank-2 tensors")
    if prompt_input_ids.size(0) != candidate_ids.size(0):
        raise ValueError("prompt and candidate batch sizes must match")
    if candidate_ids.size(1) <= 0:
        raise ValueError("candidate_ids must contain at least one token")

    emb_weight = backbone.get_input_embeddings().weight
    dev = emb_weight.device
    if model_dtype is None:
        model_dtype = getattr(backbone, "dtype", emb_weight.dtype)

    prompt_input_ids = prompt_input_ids.to(dev)
    candidate_ids = candidate_ids.to(dev)
    full_ids = torch.cat([prompt_input_ids, candidate_ids], dim=1)
    attention_mask = torch.ones_like(full_ids, dtype=torch.long, device=dev)

    emb = backbone.get_input_embeddings()(full_ids).to(model_dtype)
    inputs_embeds, attn, _labels_unused = build_inputs_embeds_with_prefix(
        emb=emb,
        attention_mask=attention_mask,
        labels=None,
        prefix=prefix.to(model_dtype),
    )
    if checkpoint_backbone and torch.is_grad_enabled() and inputs_embeds.requires_grad:
        def _forward_logits(inputs_embeds_arg: torch.Tensor, attention_mask_arg: torch.Tensor) -> torch.Tensor:
            return backbone(
                inputs_embeds=inputs_embeds_arg,
                attention_mask=attention_mask_arg,
                use_cache=False,
                return_dict=True,
            ).logits

        logits = checkpoint(
            _forward_logits,
            inputs_embeds,
            attn,
            use_reentrant=False,
            preserve_rng_state=False,
        )
    else:
        out = backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            use_cache=False,
            return_dict=True,
        )
        logits = out.logits

    prefix_len = int(prefix.size(1))
    prompt_len = int(prompt_input_ids.size(1))
    candidate_len = int(candidate_ids.size(1))
    if prompt_len <= 0:
        raise ValueError("prompt_input_ids must contain at least one token")

    positions = torch.arange(candidate_len, device=dev, dtype=torch.long) + prefix_len + prompt_len - 1
    step_logits = logits[:, positions, :].float()
    log_probs = F.log_softmax(step_logits, dim=-1)
    selected = log_probs.gather(dim=-1, index=candidate_ids.unsqueeze(-1)).squeeze(-1)
    return selected.mean(dim=1)
