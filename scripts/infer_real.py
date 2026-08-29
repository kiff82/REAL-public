#!/usr/bin/env python3
"""Single-example inference and loss harness for REAL checkpoints.

This script compares base, StaticPrefix, and REAL on one prompt/target pair or
demo file. It is the canonical reusable entrypoint for Phase A probes and for
small manual checkpoint inspections.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts._runtime_bootstrap import load_module


DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
torch = None
real = None
build_inputs_embeds_with_prefix = None
AutoModelForCausalLM = None
AutoTokenizer = None
BitsAndBytesConfig = None


def _ensure_runtime() -> None:
    """Load heavy runtime modules only when execution needs them."""
    global torch, real, build_inputs_embeds_with_prefix
    global AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if (
        torch is not None
        and real is not None
        and build_inputs_embeds_with_prefix is not None
        and AutoModelForCausalLM is not None
        and AutoTokenizer is not None
    ):
        return
    torch = load_module("torch", context="scripts/infer_real.py")
    real = load_module("train_real_v1_3", context="scripts/infer_real.py")
    prefix_ops = load_module("real.core.prefix_ops", context="scripts/infer_real.py")
    transformers = load_module("transformers", context="scripts/infer_real.py")
    build_inputs_embeds_with_prefix = prefix_ops.build_inputs_embeds_with_prefix
    AutoModelForCausalLM = transformers.AutoModelForCausalLM
    AutoTokenizer = transformers.AutoTokenizer
    BitsAndBytesConfig = getattr(transformers, "BitsAndBytesConfig", None)


@dataclass(frozen=True)
class DemoInput:
    """Normalized single-example input, regardless of source file format."""
    prompt: str
    target: Optional[str] = None
    answer_loss_tokens: Optional[int] = 64
    max_length: int = 256
    rubric: Optional[Dict[str, str]] = None


def _load_text(path: Path) -> str:
    """Read a text or JSON demo file using UTF-8."""
    return path.read_text(encoding="utf-8")


def load_demo_input(arg: str) -> DemoInput:
    """Load a demo spec from JSON/TXT or treat the argument as a raw prompt."""
    p = Path(arg)
    if p.exists() and p.is_file():
        if p.suffix.lower() == ".json":
            data = json.loads(_load_text(p))
            prompt = data.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Missing/invalid 'prompt' in {p}")
            target = data.get("target")
            if target is not None and not isinstance(target, str):
                raise ValueError(f"Invalid 'target' in {p} (must be string if present)")
            answer_loss_tokens = data.get("answer_loss_tokens", 64)
            if answer_loss_tokens is not None:
                answer_loss_tokens = int(answer_loss_tokens)
            max_length = int(data.get("max_length", 256))
            rubric = data.get("rubric")
            if rubric is not None and not isinstance(rubric, dict):
                raise ValueError(f"Invalid 'rubric' in {p} (must be object if present)")
            return DemoInput(
                prompt=prompt,
                target=target,
                answer_loss_tokens=answer_loss_tokens,
                max_length=max_length,
                rubric=rubric,
            )
        if p.suffix.lower() == ".txt":
            return DemoInput(prompt=_load_text(p))
        return DemoInput(prompt=_load_text(p))
    return DemoInput(prompt=arg)


def _check_rubric(text: str, rubric: Optional[Dict[str, str]]) -> Optional[bool]:
    """Evaluate the optional exact/contains/regex rubric for a generated string."""
    if rubric is None:
        return None
    rtype = (rubric.get("type") or "").strip().lower()
    value = rubric.get("value")
    if not isinstance(value, str):
        raise ValueError("rubric.value must be a string")

    if rtype == "exact_match":
        return text.strip() == value.strip()
    if rtype == "contains":
        return value in text
    if rtype == "regex":
        return re.search(value, text, flags=re.DOTALL) is not None
    raise ValueError(f"Unknown rubric.type={rtype!r} (expected exact_match|contains|regex)")


def _safe_float(x: Any) -> Optional[float]:
    """Best-effort float conversion for optional loss/metric fields."""
    try:
        return float(x)
    except Exception:
        return None


def _ensure_bnb_available():
    """Fail early when 4-bit loading dependencies are unavailable."""
    _ensure_runtime()
    if BitsAndBytesConfig is None:
        raise RuntimeError(
            "BitsAndBytesConfig is unavailable; install bitsandbytes/transformers to use 4-bit loading."
        )


def load_backbone_and_tokenizer(model_name: str):
    """Load the frozen backbone/tokenizer using the repo's quantized defaults."""
    _ensure_runtime()
    _ensure_bnb_available()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb,
        torch_dtype=torch.float16,
    )
    backbone.config.use_cache = False
    backbone.eval()
    real.ensure_backbone_frozen(backbone)
    return tokenizer, backbone


def _read_head_ckpt_config(path: Path) -> Dict[str, Any]:
    """Read the optional config payload embedded inside a saved head checkpoint."""
    _ensure_runtime()
    state = torch.load(path, map_location="cpu")
    cfg = state.get("config") or {}
    if not isinstance(cfg, dict):
        cfg = {}
    return cfg


def load_head(backbone: AutoModelForCausalLM, head_ckpt: Path) -> Tuple[real.REALHead, Dict[str, Any], int]:
    """Instantiate and load a REAL head checkpoint onto the backbone device."""
    _ensure_runtime()
    cfg = _read_head_ckpt_config(head_ckpt)

    emb_dev = backbone.get_input_embeddings().weight.device
    d_model = backbone.get_input_embeddings().embedding_dim

    head = real.REALHead(
        d_model=d_model,
        latent_dim=int(cfg.get("latent_dim", 256)),
        prefix_len=int(cfg.get("prefix_len", 16)),
        num_steps=int(cfg.get("num_steps", 4)),
        num_particles=int(cfg.get("num_particles", 4)),
    ).to(emb_dev)
    step_in_ckpt, _seed_in_ckpt = real.load_head_checkpoint(head, opt=None, path=head_ckpt)
    head.eval()
    for p in head.parameters():
        p.requires_grad = False
    return head, cfg, int(step_in_ckpt or 0)


def load_static_prefix(
    backbone: AutoModelForCausalLM,
    static_ckpt: Path,
) -> Tuple[real.StaticPrefix, real.StaticPrefixWrapper]:
    """Load a StaticPrefix checkpoint and wrap it around the frozen backbone."""
    _ensure_runtime()
    state = torch.load(static_ckpt, map_location="cpu")
    sd = state.get("static_prefix") if isinstance(state, dict) else None
    if not isinstance(sd, dict) or "prefix" not in sd:
        raise ValueError(f"Unrecognized static prefix checkpoint format at {static_ckpt}")

    prefix = sd["prefix"]
    if not torch.is_tensor(prefix) or prefix.ndim != 2:
        raise ValueError(f"Unrecognized static prefix tensor at {static_ckpt}")

    prefix_len, d_model_ckpt = int(prefix.shape[0]), int(prefix.shape[1])

    emb_dev = backbone.get_input_embeddings().weight.device
    d_model = backbone.get_input_embeddings().embedding_dim
    if d_model_ckpt != d_model:
        raise ValueError(f"StaticPrefix d_model mismatch: ckpt={d_model_ckpt}, backbone={d_model}")

    static_prefix = real.StaticPrefix(prefix_len=prefix_len, d_model=d_model).to(emb_dev)
    real.load_static_checkpoint(static_prefix, static_ckpt)
    static_prefix.eval()
    for p in static_prefix.parameters():
        p.requires_grad = False

    static_model = real.StaticPrefixWrapper(backbone, static_prefix).to(emb_dev)
    static_model.eval()
    return static_prefix, static_model


def build_batch(tokenizer, demo: DemoInput) -> Optional[real.Batch]:
    """Build a single-example supervised batch when the demo has a target."""
    _ensure_runtime()
    if demo.target is None:
        return None
    ex = real.build_sft_example(
        tokenizer=tokenizer,
        prompt=demo.prompt,
        answer=demo.target,
        max_length=demo.max_length,
        answer_loss_tokens=demo.answer_loss_tokens,
        prompt_head_ratio=0.50,
        add_bos=True,
        add_eos=True,
    )
    return real.collate_batch([ex], pad_id=tokenizer.pad_token_id)


def compute_real_prefix_and_trace(
    backbone: AutoModelForCausalLM,
    head: real.REALHead,
    batch: real.Batch,
    *,
    refine_policy: str,
    refine_max_steps: Optional[int],
    energy_delta_tol: float,
    patience: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the final REAL prefix, predicted energy trace, and steps used for a batch."""
    _ensure_runtime()
    emb_dev = backbone.get_input_embeddings().weight.device
    input_ids = batch.input_ids.to(emb_dev)
    attention_mask = batch.attention_mask.to(emb_dev)
    prompt_len = batch.prompt_len.to(emb_dev)

    _, L = input_ids.shape
    pos = torch.arange(L, device=emb_dev).unsqueeze(0)
    prompt_mask = (pos < prompt_len.unsqueeze(1)) & attention_mask.bool()
    trace = compute_real_prefix_trace_from_ids(
        backbone,
        head,
        input_ids,
        attention_mask,
        pool_mask=prompt_mask,
        refine_policy=refine_policy,
        refine_max_steps=refine_max_steps,
        energy_delta_tol=energy_delta_tol,
        patience=patience,
    )
    return trace["prefix_final"], trace["energy_trace"], trace["steps_used"]


def _normalize_pool_mask(
    attention_mask: torch.Tensor,
    pool_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Normalize an optional pool mask and ensure each row pools at least one visible token."""
    visible_mask = attention_mask.bool()
    if pool_mask is None:
        return visible_mask
    pool_mask_t = pool_mask.to(attention_mask.device).bool() & visible_mask
    empty_rows = pool_mask_t.sum(dim=1) <= 0
    if bool(empty_rows.any()):
        pool_mask_t = torch.where(empty_rows.unsqueeze(1), visible_mask, pool_mask_t)
    return pool_mask_t


def compute_real_prefix_trace_from_ids(
    backbone: AutoModelForCausalLM,
    head: real.REALHead,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    pool_mask: Optional[torch.Tensor] = None,
    refine_policy: str,
    refine_max_steps: Optional[int],
    energy_delta_tol: float,
    patience: int,
) -> Dict[str, torch.Tensor]:
    """Compute a REAL prefix trace from explicit ids/attention with optional pooled subspan selection.

    The head still pools a masked mean over token embeddings. This helper keeps
    that interpretation explicit while exposing the full live-state trace:
    pooled state, energy trace, latent trajectory, and prefix trajectory.
    """
    _ensure_runtime()
    with torch.no_grad():
        emb_dev = backbone.get_input_embeddings().weight.device

        input_ids = input_ids.to(emb_dev)
        attention_mask = attention_mask.to(emb_dev)
        pool_mask_t = _normalize_pool_mask(attention_mask, pool_mask)

        emb = backbone.get_input_embeddings()(input_ids)
        h = real.masked_mean(emb, pool_mask_t, dim=1)

        refine_policy_norm = (refine_policy or "forward").strip().lower()
        if refine_policy_norm in {"forward", "fixed", "fixed_steps"}:
            prefix_final, energies_t, z_t = head(h)
            steps_used = torch.full((input_ids.size(0),), int(head.num_steps), device=emb_dev, dtype=torch.long)
        else:
            prefix_final, energies_t, z_t, steps_used = head.refine_adaptive(
                h,
                max_steps=refine_max_steps,
                energy_delta_tol=energy_delta_tol,
                patience=patience,
            )

        batch_size, trace_steps, latent_dim = z_t.shape
        prefix_traj = head.prefix_proj(z_t.reshape(batch_size * trace_steps, latent_dim)).view(
            batch_size,
            trace_steps,
            head.prefix_len,
            head.d_model,
        )
        return {
            "pooled_state": h,
            "pool_mask": pool_mask_t,
            "prefix_final": prefix_final,
            "energy_trace": energies_t,
            "steps_used": steps_used,
            "z_traj": z_t,
            "prefix_traj": prefix_traj,
        }


def compute_real_prefix_and_trace_from_prompt(
    backbone: AutoModelForCausalLM,
    head: real.REALHead,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    *,
    refine_policy: str,
    refine_max_steps: Optional[int],
    energy_delta_tol: float,
    patience: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Variant of REAL prefix tracing for prompt-only inputs without supervised targets."""
    trace = compute_real_prefix_trace_from_ids(
        backbone,
        head,
        prompt_input_ids,
        prompt_attention_mask,
        pool_mask=None,
        refine_policy=refine_policy,
        refine_max_steps=refine_max_steps,
        energy_delta_tol=energy_delta_tol,
        patience=patience,
    )
    return trace["prefix_final"], trace["energy_trace"], trace["steps_used"]


def loss_with_prefix(
    backbone: AutoModelForCausalLM,
    batch: real.Batch,
    prefix_final: torch.Tensor,
) -> float:
    """Compute the answer-window CE for a batch after applying a prefix embedding."""
    _ensure_runtime()
    with torch.no_grad():
        emb_dev = backbone.get_input_embeddings().weight.device
        model_dtype = getattr(backbone, "dtype", torch.float16)

        input_ids = batch.input_ids.to(emb_dev)
        attention_mask = batch.attention_mask.to(emb_dev)
        labels = batch.labels.to(emb_dev)

        emb = backbone.get_input_embeddings()(input_ids).to(model_dtype)

        prefix_final = prefix_final.to(model_dtype)

        inputs_embeds, attn, labels2 = build_inputs_embeds_with_prefix(
            emb=emb,
            attention_mask=attention_mask,
            labels=labels,
            prefix=prefix_final,
        )

        out = backbone(inputs_embeds=inputs_embeds, attention_mask=attn, use_cache=False, return_dict=True)
        loss_per_sample = real.causal_ce_per_sample(out.logits, labels2)
        return float(loss_per_sample.mean().item())


def generate_with_prefix(
    tokenizer: AutoTokenizer,
    backbone: AutoModelForCausalLM,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    *,
    prefix_final: Optional[torch.Tensor],
    max_new_tokens: int,
    min_new_tokens: int,
) -> str:
    """Greedily generate continuation text with an optional prefix embedding applied."""
    _ensure_runtime()
    with torch.no_grad():
        emb_dev = backbone.get_input_embeddings().weight.device
        model_dtype = getattr(backbone, "dtype", torch.float16)
        pad = tokenizer.pad_token_id

        if prefix_final is None:
            out_ids = backbone.generate(
                input_ids=prompt_input_ids.to(emb_dev),
                attention_mask=prompt_attention_mask.to(emb_dev),
                do_sample=False,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                pad_token_id=pad,
            )
            return tokenizer.decode(out_ids[0][prompt_input_ids.size(1):], skip_special_tokens=True)

        emb = backbone.get_input_embeddings()(prompt_input_ids.to(emb_dev)).to(model_dtype)
        prefix_final = prefix_final.to(model_dtype)

        inputs_embeds, attn, _labels_unused = build_inputs_embeds_with_prefix(
            emb=emb,
            attention_mask=prompt_attention_mask.to(emb_dev),
            labels=None,
            prefix=prefix_final,
        )

        out_ids = backbone.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            pad_token_id=pad,
        )
        # When using inputs_embeds, some transformer versions return ONLY the newly generated tokens
        # (i.e., without the prompt/prefix tokens). Be robust and only drop if they are present.
        drop = int(prefix_final.size(1) + prompt_input_ids.size(1))
        start = drop if out_ids.size(1) >= drop else 0
        gen_part = out_ids[0, start:]
        return tokenizer.decode(gen_part, skip_special_tokens=True)


def run_inference(
    demo: DemoInput,
    *,
    tokenizer: AutoTokenizer,
    backbone: AutoModelForCausalLM,
    head: Optional[real.REALHead],
    static_model: Optional[real.StaticPrefixWrapper],
    mode: str,
    refine_policy: str,
    refine_max_steps: Optional[int],
    refine_energy_delta_tol: float,
    refine_patience: int,
    max_new_tokens: int,
    min_new_tokens: int,
    trace_energy: bool,
) -> Dict[str, Any]:
    """Run base/static/real generation and optional answer-window losses."""
    _ensure_runtime()
    mode = (mode or "all").strip().lower()
    if mode not in {"base", "static", "real", "all"}:
        raise ValueError(f"Unknown --mode {mode!r}")

    batch = build_batch(tokenizer, demo)
    if batch is not None:
        prompt_len = int(batch.prompt_len[0].item())
        prompt_input_ids = batch.input_ids[:, :prompt_len]
        prompt_attention_mask = batch.attention_mask[:, :prompt_len]
    else:
        prompt_ids = tokenizer(demo.prompt, add_special_tokens=False).input_ids
        max_len = int(demo.max_length or 256)
        bos = getattr(tokenizer, "bos_token_id", None)
        if max_len <= 0:
            raise ValueError(f"max_length must be positive (got {max_len})")
        budget = max_len - (1 if bos is not None else 0)
        if budget < 0:
            raise ValueError(f"max_length={max_len} too small to fit BOS token")
        if len(prompt_ids) > budget:
            prompt_ids = [] if budget == 0 else real._truncate_tokens_head_tail(prompt_ids, budget, head_ratio=0.50)
        if bos is not None:
            prompt_ids = [bos] + prompt_ids
        prompt_input_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
        prompt_attention_mask = torch.ones_like(prompt_input_ids)

    res: Dict[str, Any] = {
        "prompt": demo.prompt,
        "target": demo.target,
        "answer_loss_tokens": demo.answer_loss_tokens,
        "max_length": demo.max_length,
        "mode": mode,
        "refine_policy": refine_policy,
        "refine_max_steps": refine_max_steps,
        "refine_energy_delta_tol": refine_energy_delta_tol,
        "refine_patience": refine_patience,
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "rubric": demo.rubric,
    }

    want_base = mode in {"base", "all"}
    want_static = mode in {"static", "all"} and static_model is not None
    want_real = mode in {"real", "all"} and head is not None

    if mode in {"static", "all"} and static_model is None:
        res["static_warning"] = "static_ckpt not provided; static mode skipped"
    if mode in {"real", "all"} and head is None:
        raise ValueError("--head_ckpt is required for modes including real")

    losses: Dict[str, Optional[float]] = {"base": None, "static": None, "real": None, "cond": None}
    gens: Dict[str, Optional[str]] = {"base": None, "static": None, "real": None}
    rubric_ok: Dict[str, Optional[bool]] = {"base": None, "static": None, "real": None}

    prefix_real = None
    energy_trace = None
    steps_used = None
    if want_real:
        if batch is not None:
            prefix_real, energy_trace_t, steps_used_t = compute_real_prefix_and_trace(
                backbone,
                head,
                batch,
                refine_policy=refine_policy,
                refine_max_steps=refine_max_steps,
                energy_delta_tol=refine_energy_delta_tol,
                patience=refine_patience,
            )
        else:
            prefix_real, energy_trace_t, steps_used_t = compute_real_prefix_and_trace_from_prompt(
                backbone,
                head,
                prompt_input_ids,
                prompt_attention_mask,
                refine_policy=refine_policy,
                refine_max_steps=refine_max_steps,
                energy_delta_tol=refine_energy_delta_tol,
                patience=refine_patience,
            )
        steps_used = int(steps_used_t[0].item())
        if trace_energy:
            energy_trace = [float(x) for x in energy_trace_t[0].detach().cpu().tolist()]

    if want_base:
        gens["base"] = generate_with_prefix(
            tokenizer,
            backbone,
            prompt_input_ids,
            prompt_attention_mask,
            prefix_final=None,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
        )
        rubric_ok["base"] = _check_rubric(gens["base"], demo.rubric)

    if want_static:
        prefix_static = static_model.static_prefix(1)
        gens["static"] = generate_with_prefix(
            tokenizer,
            backbone,
            prompt_input_ids,
            prompt_attention_mask,
            prefix_final=prefix_static,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
        )
        rubric_ok["static"] = _check_rubric(gens["static"], demo.rubric)

    if want_real:
        gens["real"] = generate_with_prefix(
            tokenizer,
            backbone,
            prompt_input_ids,
            prompt_attention_mask,
            prefix_final=prefix_real,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
        )
        rubric_ok["real"] = _check_rubric(gens["real"], demo.rubric)

    if batch is not None:
        if want_base:
            losses["base"] = _safe_float(real.baseline_loss_mean(backbone, batch).item())
        if want_static:
            losses["static"] = _safe_float(static_model(batch).item())
        if want_real:
            losses["real"] = loss_with_prefix(backbone, batch, prefix_real)
        if losses["static"] is not None and losses["real"] is not None:
            losses["cond"] = losses["static"] - losses["real"]

    if demo.target is None:
        res["loss_warning"] = "no target provided; skipping loss computation"

    res.update(
        {
            "loss": losses,
            "gen": gens,
            "rubric_ok": rubric_ok,
            "steps_used_real": steps_used,
        }
    )
    res.update(
        {
            "loss_base": losses["base"],
            "loss_static": losses["static"],
            "loss_real": losses["real"],
            "cond": losses["cond"],
        }
    )
    if trace_energy:
        res["pred_energy_real"] = energy_trace
    return res


def main():
    """CLI entrypoint for single-example REAL inference."""
    parser = argparse.ArgumentParser(description="Single-example REAL inference harness (base/static/real)")
    parser.add_argument("--head_ckpt", type=str, default=None, help="Path to head_stepXXXXX.pt (required for real)")
    parser.add_argument("--static_ckpt", type=str, default=None, help="Path to static_prefix.pt (optional)")
    parser.add_argument("--model_name", type=str, default=None, help="HF model name override (defaults from head_ckpt or Qwen3-4B)")
    parser.add_argument("--mode", type=str, default="all", choices=["base", "static", "real", "all"])
    parser.add_argument("--refine_policy", type=str, default="accept_reject_v1", choices=["forward", "accept_reject_v1"])
    parser.add_argument("--refine_max_steps", type=int, default=None)
    parser.add_argument("--refine_energy_delta_tol", type=float, default=1e-3)
    parser.add_argument("--refine_patience", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--min_new_tokens", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input", type=str, required=True, help="Path to demo JSON/.txt or a literal prompt string")
    parser.add_argument("--out_json", type=str, default=None, help="Optional path to write JSON output")
    parser.add_argument("--trace_energy", action="store_true", help="Include predicted energy trace in JSON/stdout")
    args = parser.parse_args()

    _ensure_runtime()
    real.set_seed(args.seed)

    head_ckpt = Path(args.head_ckpt) if args.head_ckpt else None
    static_ckpt = Path(args.static_ckpt) if args.static_ckpt else None

    cfg = {}
    if head_ckpt is not None:
        cfg = _read_head_ckpt_config(head_ckpt)
    model_name = args.model_name or cfg.get("model_name") or DEFAULT_MODEL_NAME

    tokenizer, backbone = load_backbone_and_tokenizer(model_name)

    head = None
    head_step = None
    if head_ckpt is not None:
        head, _cfg2, head_step = load_head(backbone, head_ckpt)

    static_model = None
    if static_ckpt is not None:
        _static_prefix, static_model = load_static_prefix(backbone, static_ckpt)

    demo = load_demo_input(args.input)
    t0 = time.time()
    out = run_inference(
        demo,
        tokenizer=tokenizer,
        backbone=backbone,
        head=head,
        static_model=static_model,
        mode=args.mode,
        refine_policy=args.refine_policy,
        refine_max_steps=args.refine_max_steps,
        refine_energy_delta_tol=args.refine_energy_delta_tol,
        refine_patience=args.refine_patience,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        trace_energy=args.trace_energy,
    )
    out.update(
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_name": model_name,
            "head_ckpt": str(head_ckpt) if head_ckpt else None,
            "static_ckpt": str(static_ckpt) if static_ckpt else None,
            "head_step": head_step,
            "seed": args.seed,
            "runtime_s": round(time.time() - t0, 3),
        }
    )

    loss = out.get("loss") or {}
    l_base = loss.get("base")
    l_static = loss.get("static")
    l_real = loss.get("real")
    cond = loss.get("cond")
    print(f"[infer] model={model_name}")
    if head_ckpt:
        print(f"[infer] head_ckpt={head_ckpt} (step={head_step})")
    if static_ckpt:
        print(f"[infer] static_ckpt={static_ckpt}")
    if l_base is not None or l_real is not None or l_static is not None:
        print(f"[loss] base={l_base} static={l_static} real={l_real} cond={cond}")
    if out.get("loss_warning"):
        print(f"[warn] {out.get('loss_warning')}")
    if out.get("steps_used_real") is not None:
        print(f"[real] steps_used={out.get('steps_used_real')}")
    if args.trace_energy and out.get("pred_energy_real") is not None:
        print(f"[real] pred_energy={out.get('pred_energy_real')}")

    gen = out.get("gen") or {}
    if gen.get("base") is not None:
        print("\n[base]\n" + gen["base"])
    if gen.get("static") is not None:
        print("\n[static]\n" + gen["static"])
    if gen.get("real") is not None:
        print("\n[real]\n" + gen["real"])

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"\n[wrote] {out_path}")


if __name__ == "__main__":
    main()
