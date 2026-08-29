from __future__ import annotations

import torch


def cuda_prefers_bf16() -> bool:
    """Return True when the active CUDA device should default to bf16.

    Newer CUDA/PyTorch stacks can report `is_bf16_supported()` on pre-Ampere
    GPUs such as T4 even though REAL should still prefer fp16 there. Treat
    Ampere+ (SM80+) as the default bf16 line and keep older cards on fp16.
    """
    if not torch.cuda.is_available():
        return False
    try:
        major, _minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    except Exception:
        return False
    return bool(torch.cuda.is_bf16_supported() and major >= 8)


def auto_dtype() -> torch.dtype:
    """Return REAL's default mixed-precision compute dtype.

    REAL prefers bf16 only on Ampere+ CUDA GPUs and keeps pre-Ampere cards
    such as T4 on fp16 by default.
    """
    if cuda_prefers_bf16():
        return torch.bfloat16
    return torch.float16


def resolve_dtype_mode(mode: str) -> torch.dtype:
    """Resolve a dtype mode string to a torch.dtype.

    Supported:
    - "auto" -> auto_dtype()
    - "bf16" -> torch.bfloat16
    - "fp16"/"float16" -> torch.float16
    """
    m = (mode or "auto").strip().lower()
    if m == "auto":
        return auto_dtype()
    if m in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if m in {"fp16", "float16", "f16"}:
        return torch.float16
    raise ValueError(f"Unknown dtype mode: {mode!r} (expected auto|bf16|fp16)")
