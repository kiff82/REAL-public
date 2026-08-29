#!/usr/bin/env python3
"""Dedicated uncertainty-aware reasoning evaluation for REAL checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import numbers
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pathlib

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from real.core.path_ops import basename_or_relpath, maybe_localize_outputs_path, portable_path_str
from real.core.protocol import EvalProtocol
from scripts._runtime_bootstrap import load_module
from utils.hotpot_reasoning_contracts import (
    HOTPOT_REASONING_OUTPUT_CONTRACTS,
    HOTPOT_REASONING_PROMPT_VARIANTS,
    HOTPOT_REASONING_RESPONSE_CUE_VARIANTS,
    classify_hotpot_keyfacts_structure,
    extract_hotpot_reasoning_field_char_spans,
    hotpot_contract_field_names,
    hotpot_reasoning_effective_generated_text,
    hotpot_reasoning_response_cue,
    hotpot_reasoning_target_continuation_text,
    hotpot_reasoning_target_text,
    normalize_hotpot_reasoning_output_contract,
    normalize_hotpot_reasoning_response_cue_variant,
    parse_hotpot_reasoning_fields,
    rebase_hotpot_char_spans_from_effective_text,
)


DEFAULT_TASKS = ["squad2_reasoning", "fever_reasoning"]
TASK_CHOICES = tuple(DEFAULT_TASKS + ["hotpotqa_reasoning"])
GENERATION_TRACE_TOPK_DEFAULT = 5
HOTPOT_ENTRY_BOUNDARY_NAMES: Tuple[str, ...] = (
    "key_fact_1_to_key_fact_2",
    "key_fact_2_to_answer",
)
torch = None
infer = None
real = None
build_inputs_embeds_with_prefix = None
compute_context_pull_probe = None
compute_pressure_trace = None
summarize_pressure_trace = None
compute_masked_replay_probe = None
make_token_mask = None
token_span_from_char_span = None
compute_entry_transition_probe_from_scores = None
compute_generation_trace_from_scores = None
summarize_real_trace = None


def _ensure_runtime() -> None:
    """Load heavy runtime modules only after CLI parsing."""
    global torch, infer, real, build_inputs_embeds_with_prefix
    global compute_context_pull_probe, compute_pressure_trace, summarize_pressure_trace
    global compute_masked_replay_probe, make_token_mask, token_span_from_char_span
    global compute_entry_transition_probe_from_scores, compute_generation_trace_from_scores, summarize_real_trace
    if (
        torch is not None
        and infer is not None
        and real is not None
        and build_inputs_embeds_with_prefix is not None
        and compute_context_pull_probe is not None
        and compute_pressure_trace is not None
        and summarize_pressure_trace is not None
        and compute_masked_replay_probe is not None
        and make_token_mask is not None
        and token_span_from_char_span is not None
        and compute_entry_transition_probe_from_scores is not None
        and compute_generation_trace_from_scores is not None
        and summarize_real_trace is not None
    ):
        return
    torch = load_module("torch", context="scripts/reasoning_eval.py")
    infer = load_module("infer_real", context="scripts/reasoning_eval.py")
    real = load_module("train_real_v1_3", context="scripts/reasoning_eval.py")
    prefix_ops = load_module("real.core.prefix_ops", context="scripts/reasoning_eval.py")
    context_pull_mod = load_module("utils.context_pull_probe", context="scripts/reasoning_eval.py")
    generation_trace_mod = load_module("utils.generation_trace", context="scripts/reasoning_eval.py")
    pressure_probe_mod = load_module("utils.pressure_probe", context="scripts/reasoning_eval.py")
    replay_probe_mod = load_module("utils.replay_probe", context="scripts/reasoning_eval.py")
    real_trace_probe_mod = load_module("utils.real_trace_probe", context="scripts/reasoning_eval.py")
    build_inputs_embeds_with_prefix = prefix_ops.build_inputs_embeds_with_prefix
    compute_context_pull_probe = context_pull_mod.compute_context_pull_probe
    compute_entry_transition_probe_from_scores = generation_trace_mod.compute_entry_transition_probe_from_scores
    compute_generation_trace_from_scores = generation_trace_mod.compute_generation_trace_from_scores
    compute_pressure_trace = pressure_probe_mod.compute_pressure_trace
    summarize_pressure_trace = pressure_probe_mod.summarize_pressure_trace
    compute_masked_replay_probe = replay_probe_mod.compute_masked_replay_probe
    make_token_mask = replay_probe_mod.make_token_mask
    token_span_from_char_span = replay_probe_mod.token_span_from_char_span
    summarize_real_trace = real_trace_probe_mod.summarize_real_trace


def _lazy_no_grad(fn):
    """Wrap a helper so it loads torch lazily before entering no_grad."""
    def wrapped(*args, **kwargs):
        _ensure_runtime()
        with torch.no_grad():
            return fn(*args, **kwargs)

    return wrapped


@dataclass(frozen=True)
class TaskBundle:
    """Resolved dataset/config bundle for one reasoning-eval task."""
    name: str
    dataset: Any
    dataset_name: str
    dataset_config_name: Optional[str]
    split: str
    max_length: int
    answer_loss_tokens: int
    benchmark_role: str
    reasoning_eval_mode: bool
    allow_abstain: bool
    abstain_mode: str
    abstain_text: Optional[str]
    abstain_option_letter: Optional[str]
    primary_metric_name: str
    output_contract: Optional[str] = None
    prompt_variant: Optional[str] = None
    response_cue_variant: Optional[str] = None


def _load_json(path: Path) -> Dict[str, Any]:
    """Read a JSON object from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL rows from disk."""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    """Write a JSON object with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_portable_json_value(obj), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write JSONL rows for per-sample inspection/debugging."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_portable_json_value(row), sort_keys=True) + "\n")


def _safe_float(x: Any) -> Optional[float]:
    """Best-effort float conversion used for optional metrics."""
    try:
        return float(x)
    except Exception:
        return None


def _portable_json_value(x: Any) -> Any:
    """Replace NaN/inf with JSON-portable nulls inside nested structures."""
    if isinstance(x, dict):
        return {k: _portable_json_value(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_portable_json_value(v) for v in x]
    if isinstance(x, tuple):
        return [_portable_json_value(v) for v in x]
    if isinstance(x, numbers.Real) and not isinstance(x, bool):
        x_float = float(x)
        if math.isnan(x_float) or math.isinf(x_float):
            return None
    return x


def _fmt_metric(x: Optional[float]) -> str:
    """Format a metric for concise CLI logging."""
    if x is None:
        return "nan"
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return f"{x:.6f}"


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    """Drop duplicates without disturbing the original order."""
    seen = set()
    out: List[str] = []
    for raw_value in values:
        value = str(raw_value).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _load_sample_ids(path: Path) -> List[str]:
    """Load sample ids from plain text, JSON, or JSONL."""
    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise SystemExit(f"--sample_ids_path is empty: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(raw_text)
        if not isinstance(data, list):
            raise SystemExit(f"--sample_ids_path JSON must be a list of strings: {path}")
        return _dedupe_preserve_order([str(item).strip() for item in data])

    if path.suffix.lower() == ".jsonl":
        rows = _load_jsonl(path)
        sample_ids: List[str] = []
        for idx, row in enumerate(rows, start=1):
            if not isinstance(row, dict) or "sample_id" not in row:
                raise SystemExit(f"--sample_ids_path JSONL rows must contain `sample_id`: {path}:{idx}")
            sample_ids.append(str(row.get("sample_id") or "").strip())
        return _dedupe_preserve_order(sample_ids)

    if raw_text.startswith("["):
        data = json.loads(raw_text)
        if not isinstance(data, list):
            raise SystemExit(f"--sample_ids_path JSON must be a list of strings: {path}")
        return _dedupe_preserve_order([str(item).strip() for item in data])

    return _dedupe_preserve_order([line.strip() for line in raw_text.splitlines()])


def _relpath_str(path: Path, root: Path) -> str:
    """Return a path relative to `root` when possible."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _first_nonempty_line(text: str) -> str:
    """Extract the first non-empty generated line."""
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line
    return (text or "").strip()


def _normalized_token_f1(pred_text: str, gold_text: str) -> float:
    """Simple token-overlap F1 over normalized text."""
    pred_tokens = real._normalize_eval_text(pred_text).split()
    gold_tokens = real._normalize_eval_text(gold_text).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    shared = sum(int(v) for v in overlap.values())
    if shared <= 0:
        return 0.0
    precision = float(shared) / float(len(pred_tokens))
    recall = float(shared) / float(len(gold_tokens))
    return (2.0 * precision * recall) / (precision + recall)


def _parse_hotpot_output(raw_output: str, *, output_contract: Optional[str] = None) -> Tuple[List[str], str]:
    """Extract supporting-fact bullets and the final answer from a structured response."""
    parsed = parse_hotpot_reasoning_fields(raw_output, output_contract=output_contract)
    support_facts = [str(item).strip() for item in (parsed.get("supporting_facts") or []) if str(item).strip()]
    answer_text = str(parsed.get("answer") or "").strip()
    if not answer_text and support_facts:
        answer_text = support_facts[-1]
    return support_facts, answer_text


def _normalized_contains(text: str, subtext: str) -> bool:
    norm_text = real._normalize_eval_text(text)
    norm_subtext = real._normalize_eval_text(subtext)
    return bool(norm_subtext) and norm_subtext in norm_text


def _normalized_token_overlap_fraction(text: str, subtext: str) -> float:
    base_tokens = real._normalize_eval_text(subtext).split()
    if not base_tokens:
        return 0.0
    text_tokens = set(real._normalize_eval_text(text).split())
    shared = sum(1 for token in base_tokens if token in text_tokens)
    return float(shared) / float(len(base_tokens))


def _hotpot_response_cue_text(meta: Dict[str, Any]) -> str:
    contract = normalize_hotpot_reasoning_output_contract(meta.get("hotpotqa_reasoning_output_contract"))
    cue_text = meta.get("hotpotqa_reasoning_response_cue_text")
    if isinstance(cue_text, str) and cue_text:
        return cue_text
    return hotpot_reasoning_response_cue(
        contract,
        response_cue_variant=meta.get("hotpotqa_reasoning_response_cue_variant"),
    )


def _hotpot_effective_generated_text(meta: Dict[str, Any], raw_output: str) -> Tuple[str, int]:
    contract = normalize_hotpot_reasoning_output_contract(meta.get("hotpotqa_reasoning_output_contract"))
    return hotpot_reasoning_effective_generated_text(
        raw_output,
        output_contract=contract,
        response_cue_variant=meta.get("hotpotqa_reasoning_response_cue_variant"),
        response_cue_text=_hotpot_response_cue_text(meta),
    )


def _compute_hotpot_generated_structure(meta: Dict[str, Any], raw_output: str) -> Dict[str, Any]:
    """Parse one generated Hotpot response into a stable structure payload."""
    contract = normalize_hotpot_reasoning_output_contract(meta.get("hotpotqa_reasoning_output_contract"))
    effective_text, cue_prefix_chars = _hotpot_effective_generated_text(meta, raw_output)
    parsed = parse_hotpot_reasoning_fields(effective_text, output_contract=contract)
    fields = parsed.get("fields") or {}
    char_spans = extract_hotpot_reasoning_field_char_spans(effective_text, output_contract=contract)
    weak_structure = classify_hotpot_keyfacts_structure(effective_text, parsed=parsed)
    char_spans.update(dict(weak_structure.get("char_spans") or {}))
    char_spans = rebase_hotpot_char_spans_from_effective_text(char_spans, cue_prefix_chars)

    key_fact_1 = str(((fields.get("key_fact_1") or {}).get("text")) or "").strip()
    key_fact_2 = str(((fields.get("key_fact_2") or {}).get("text")) or "").strip()
    parsed_supporting_facts = [str(item).strip() for item in (parsed.get("supporting_facts") or []) if str(item).strip()]

    answer_field_name = "answer" if contract in {"keyfacts", "answer_only"} else "final_answer"
    answer_header_present = bool(((fields.get(answer_field_name) or {}).get("present")))
    answer_nonempty = bool(((fields.get(answer_field_name) or {}).get("nonempty")))
    parsed_answer = str(parsed.get("answer") or "").strip()
    if not parsed_answer and parsed_supporting_facts:
        parsed_answer = parsed_supporting_facts[-1]

    def _required_present(field_name: str) -> bool:
        field = fields.get(field_name) or {}
        return bool(field.get("present"))

    def _required_nonempty(field_name: str) -> bool:
        if field_name == "supporting_facts":
            return bool(parsed_supporting_facts)
        field = fields.get(field_name) or {}
        return bool(field.get("nonempty"))

    required_fields = hotpot_contract_field_names(contract)
    answer_in_key_fact_1_norm = _normalized_contains(key_fact_1, parsed_answer)
    answer_in_key_fact_2_norm = _normalized_contains(key_fact_2, parsed_answer)
    return {
        "contract": contract,
        "parse_coverage": 1.0,
        "has_key_fact_1": bool(((fields.get("key_fact_1") or {}).get("present"))),
        "has_key_fact_2": bool(((fields.get("key_fact_2") or {}).get("present"))),
        "has_answer_header": answer_header_present,
        "all_fields_present": all(_required_present(name) for name in required_fields),
        "key_fact_1_nonempty": bool(((fields.get("key_fact_1") or {}).get("nonempty"))),
        "key_fact_2_nonempty": bool(((fields.get("key_fact_2") or {}).get("nonempty"))),
        "answer_nonempty": answer_nonempty,
        "all_fields_nonempty": all(_required_nonempty(name) for name in required_fields),
        "parsed_key_fact_1": key_fact_1,
        "parsed_key_fact_2": key_fact_2,
        "parsed_answer": parsed_answer,
        "parsed_supporting_facts": parsed_supporting_facts,
        "structure_state": str(weak_structure.get("structure_state") or "no_structure"),
        "fact_scaffold_present": bool(weak_structure.get("fact_scaffold_present")),
        "answer_field_present": bool(weak_structure.get("answer_field_present")),
        "full_keyfacts_contract_present": bool(weak_structure.get("full_keyfacts_contract_present")),
        "labeled_fact_scaffold_present": bool(weak_structure.get("labeled_fact_scaffold_present")),
        "numbered_fact_scaffold_present": bool(weak_structure.get("numbered_fact_scaffold_present")),
        "has_numbered_fact_1": bool(weak_structure.get("has_numbered_fact_1")),
        "has_numbered_fact_2": bool(weak_structure.get("has_numbered_fact_2")),
        "numbered_fact_1_nonempty": bool(weak_structure.get("numbered_fact_1_nonempty")),
        "numbered_fact_2_nonempty": bool(weak_structure.get("numbered_fact_2_nonempty")),
        "numbered_answer_nonempty": bool(weak_structure.get("numbered_answer_nonempty")),
        "numbered_answer_header_present": bool(weak_structure.get("numbered_answer_header_present")),
        "numbered_key_facts_distinct": bool(weak_structure.get("numbered_key_facts_distinct")),
        "parsed_numbered_key_fact_1": str(weak_structure.get("parsed_numbered_key_fact_1") or "").strip(),
        "parsed_numbered_key_fact_2": str(weak_structure.get("parsed_numbered_key_fact_2") or "").strip(),
        "parsed_numbered_answer": str(weak_structure.get("parsed_numbered_answer") or "").strip(),
        "key_facts_distinct": bool(
            key_fact_1
            and key_fact_2
            and real._normalize_eval_text(key_fact_1) != real._normalize_eval_text(key_fact_2)
        ),
        "answer_in_key_fact_1_norm": answer_in_key_fact_1_norm,
        "answer_in_key_fact_2_norm": answer_in_key_fact_2_norm,
        "answer_in_any_key_fact_norm": bool(answer_in_key_fact_1_norm or answer_in_key_fact_2_norm),
        "answer_token_overlap_key_fact_1": _normalized_token_overlap_fraction(key_fact_1, parsed_answer),
        "answer_token_overlap_key_fact_2": _normalized_token_overlap_fraction(key_fact_2, parsed_answer),
        "char_spans": char_spans,
    }


def _hotpot_keyfacts_leakage(meta: Dict[str, Any]) -> Dict[str, Any]:
    answer_text = str(meta.get("gold_answer") or "")
    key_fact_1 = str(meta.get("gold_key_fact_1") or "")
    key_fact_2 = str(meta.get("gold_key_fact_2") or "")
    in_kf1 = _normalized_contains(key_fact_1, answer_text)
    in_kf2 = _normalized_contains(key_fact_2, answer_text)
    return {
        "answer_in_key_fact_1_norm": in_kf1,
        "answer_in_key_fact_2_norm": in_kf2,
        "answer_in_any_key_fact_norm": bool(in_kf1 or in_kf2),
        "answer_token_overlap_key_fact_1": _normalized_token_overlap_fraction(key_fact_1, answer_text),
        "answer_token_overlap_key_fact_2": _normalized_token_overlap_fraction(key_fact_2, answer_text),
    }


def _shift_char_span(char_span: Optional[List[int]], delta: int) -> Optional[Tuple[int, int]]:
    if not isinstance(char_span, list) or len(char_span) != 2:
        return None
    try:
        start = int(char_span[0]) + int(delta)
        end = int(char_span[1]) + int(delta)
    except (TypeError, ValueError):
        return None
    if end <= start:
        return None
    return start, end


def _char_span_tuple(char_span: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(char_span, (list, tuple)) or len(char_span) != 2:
        return None
    try:
        start = int(char_span[0])
        end = int(char_span[1])
    except (TypeError, ValueError):
        return None
    if end <= start:
        return None
    return start, end


@_lazy_no_grad
def _compute_hotpot_dependency_probe_from_replay_region(
    *,
    tokenizer,
    backbone,
    replay_text: str,
    replay_input_ids: torch.Tensor,
    full_input_ids: torch.Tensor,
    full_attention_mask: torch.Tensor,
    replay_token_offset: int,
    prefix_final: Optional[torch.Tensor],
    answer_char_span: Tuple[int, int],
    key_fact_1_value_span: Tuple[int, int],
    key_fact_2_value_span: Tuple[int, int],
    key_fact_1_header_span: Optional[Tuple[int, int]],
    key_fact_2_header_span: Optional[Tuple[int, int]],
    leakage: Dict[str, Any],
    probe_type: str,
) -> Dict[str, Any]:
    if replay_input_ids.ndim == 1:
        replay_input_ids = replay_input_ids.unsqueeze(0)

    answer_token_span_rel = token_span_from_char_span(tokenizer, replay_text, answer_char_span, replay_input_ids)
    if answer_token_span_rel is None:
        return {"missing_reason": "answer_span_to_tokens_failed"}
    answer_full_span = (
        int(replay_token_offset + answer_token_span_rel[0]),
        int(replay_token_offset + answer_token_span_rel[1]),
    )

    mask_char_spans = {
        "headers_only": [key_fact_1_header_span, key_fact_2_header_span],
        "key_fact_1": [key_fact_1_value_span],
        "key_fact_2": [key_fact_2_value_span],
        "both_key_facts": [key_fact_1_value_span, key_fact_2_value_span],
    }

    mask_payloads: Dict[str, Any] = {}
    for mask_name, candidate_char_spans in mask_char_spans.items():
        char_spans = [char_span for char_span in candidate_char_spans if char_span is not None]
        if not char_spans:
            return {"missing_reason": f"{mask_name}_no_mask_spans"}
        rel_token_spans: List[Tuple[int, int]] = []
        full_token_spans: List[Tuple[int, int]] = []
        for char_span in char_spans:
            token_span_rel = token_span_from_char_span(tokenizer, replay_text, char_span, replay_input_ids)
            if token_span_rel is None:
                return {"missing_reason": f"{mask_name}_span_to_tokens_failed"}
            rel_token_spans.append((int(token_span_rel[0]), int(token_span_rel[1])))
            full_token_spans.append(
                (int(replay_token_offset + token_span_rel[0]), int(replay_token_offset + token_span_rel[1]))
            )

        occlude_mask = make_token_mask(int(full_input_ids.size(1)), full_token_spans)
        probe = compute_masked_replay_probe(
            backbone=backbone,
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            prefix_final=prefix_final,
            score_token_span=answer_full_span,
            occlude_token_mask=occlude_mask,
            trace_prefix="answer",
            target_token_ids=replay_input_ids[:, answer_token_span_rel[0] : answer_token_span_rel[1]],
        )
        if "summary" not in probe:
            return {"missing_reason": str(probe.get("missing_reason") or f"{mask_name}_replay_failed")}
        mask_payloads[mask_name] = {
            "summary": probe["summary"],
            "answer_kl": list(probe.get("kl") or []),
            "answer_js": list(probe.get("js") or []),
            "answer_token_logprob_drop": list(probe.get("token_logprob_drop") or []),
            "answer_gap_drop": list(probe.get("gap_drop") or []),
            "answer_top1_flip": list(probe.get("top1_flip") or []),
            "masked_token_spans": [[int(start), int(end)] for start, end in full_token_spans],
            "masked_token_spans_replay_region": [[int(start), int(end)] for start, end in rel_token_spans],
        }

    return {
        "probe_type": probe_type,
        "contract": "keyfacts",
        "answer_token_span": [int(answer_full_span[0]), int(answer_full_span[1])],
        "answer_token_span_replay_region": [int(answer_token_span_rel[0]), int(answer_token_span_rel[1])],
        "replay_token_offset": int(replay_token_offset),
        "leakage": leakage,
        "masks": mask_payloads,
    }


@_lazy_no_grad
def _compute_hotpot_gold_dependency_probe(
    *,
    tokenizer,
    backbone,
    prompt_text: str,
    prompt_input_ids: torch.Tensor,
    prefix_final: Optional[torch.Tensor],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    contract = normalize_hotpot_reasoning_output_contract(meta.get("hotpotqa_reasoning_output_contract"))
    if contract != "keyfacts":
        return {"missing_reason": "unsupported_contract"}

    gold_support = [str(item) for item in (meta.get("gold_supporting_facts") or []) if str(item).strip()]
    gold_answer = str(meta.get("gold_answer") or "").strip()
    if not gold_support or not gold_answer:
        return {"missing_reason": "missing_gold_target"}

    response_cue_variant = normalize_hotpot_reasoning_response_cue_variant(
        meta.get("hotpotqa_reasoning_response_cue_variant")
    )
    response_cue_text = _hotpot_response_cue_text(meta)
    target_text = hotpot_reasoning_target_text(gold_support, gold_answer, output_contract=contract)
    target_continuation_text = hotpot_reasoning_target_continuation_text(
        gold_support,
        gold_answer,
        output_contract=contract,
        response_cue_variant=response_cue_variant,
    )
    parsed_target = parse_hotpot_reasoning_fields(target_text, output_contract=contract)
    fields = parsed_target.get("fields") or {}
    target_char_delta = len(prompt_text)
    if contract == "keyfacts" and response_cue_variant == "key_fact_1":
        target_char_delta -= len(response_cue_text)

    answer_char_span = _shift_char_span(((fields.get("answer") or {}).get("value_span")), target_char_delta)
    key_fact_1_value_span = _shift_char_span(((fields.get("key_fact_1") or {}).get("value_span")), target_char_delta)
    key_fact_2_value_span = _shift_char_span(((fields.get("key_fact_2") or {}).get("value_span")), target_char_delta)
    key_fact_1_header_span = _shift_char_span(((fields.get("key_fact_1") or {}).get("header_span")), target_char_delta)
    key_fact_2_header_span = _shift_char_span(((fields.get("key_fact_2") or {}).get("header_span")), target_char_delta)

    if answer_char_span is None:
        return {"missing_reason": "missing_answer_value_span"}
    if key_fact_1_value_span is None:
        return {"missing_reason": "missing_key_fact_1_value_span"}
    if key_fact_2_value_span is None:
        return {"missing_reason": "missing_key_fact_2_value_span"}
    full_text = prompt_text + target_continuation_text
    full_ids_list = _tokenizer_input_ids(tokenizer(full_text, add_special_tokens=False))
    bos_id = getattr(tokenizer, "bos_token_id", None)
    prompt_flat_ids = prompt_input_ids[0].detach().cpu().tolist()
    if bos_id is not None and prompt_flat_ids and prompt_flat_ids[0] == bos_id:
        full_ids_list = [bos_id] + full_ids_list
    full_ids = torch.tensor([full_ids_list], dtype=torch.long)
    full_attn = torch.ones_like(full_ids, dtype=torch.long)
    return _compute_hotpot_dependency_probe_from_replay_region(
        tokenizer=tokenizer,
        backbone=backbone,
        replay_text=full_text,
        replay_input_ids=full_ids,
        full_input_ids=full_ids,
        full_attention_mask=full_attn,
        replay_token_offset=0,
        prefix_final=prefix_final,
        answer_char_span=answer_char_span,
        key_fact_1_value_span=key_fact_1_value_span,
        key_fact_2_value_span=key_fact_2_value_span,
        key_fact_1_header_span=key_fact_1_header_span,
        key_fact_2_header_span=key_fact_2_header_span,
        leakage=_hotpot_keyfacts_leakage(meta),
        probe_type="teacher_forced_gold_target",
    )


@_lazy_no_grad
def _compute_hotpot_free_dependency_probe(
    *,
    tokenizer,
    backbone,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    generated_ids: torch.Tensor,
    prefix_final: Optional[torch.Tensor],
    meta: Dict[str, Any],
    raw_output: str,
) -> Dict[str, Any]:
    contract = normalize_hotpot_reasoning_output_contract(meta.get("hotpotqa_reasoning_output_contract"))
    if contract != "keyfacts":
        return {"missing_reason": "unsupported_contract"}

    if generated_ids.ndim == 1:
        generated_ids = generated_ids.unsqueeze(0)
    if generated_ids.numel() == 0:
        return {"missing_reason": "empty_generation"}

    generated_structure = _compute_hotpot_generated_structure(meta, raw_output)
    if not bool(generated_structure.get("all_fields_present")):
        structure_state = str(generated_structure.get("structure_state") or "no_structure")
        return {"missing_reason": f"generated_structure_{structure_state}"}
    if not bool(generated_structure.get("all_fields_nonempty")):
        return {"missing_reason": "generated_structure_nonempty_required"}

    char_spans = dict(generated_structure.get("char_spans") or {})
    answer_char_span = _char_span_tuple(char_spans.get("answer_value_span"))
    key_fact_1_value_span = _char_span_tuple(char_spans.get("key_fact_1_value_span"))
    key_fact_2_value_span = _char_span_tuple(char_spans.get("key_fact_2_value_span"))
    key_fact_1_header_span = _char_span_tuple(char_spans.get("key_fact_1_header_span"))
    key_fact_2_header_span = _char_span_tuple(char_spans.get("key_fact_2_header_span"))

    if answer_char_span is None:
        return {"missing_reason": "missing_answer_value_span"}
    if key_fact_1_value_span is None:
        return {"missing_reason": "missing_key_fact_1_value_span"}
    if key_fact_2_value_span is None:
        return {"missing_reason": "missing_key_fact_2_value_span"}
    prompt_ids = prompt_input_ids.detach().cpu()
    prompt_attn = prompt_attention_mask.detach().cpu()
    gen_ids = generated_ids.detach().cpu().long()
    gen_attn = torch.ones_like(gen_ids, dtype=prompt_attn.dtype)
    full_ids = torch.cat([prompt_ids, gen_ids], dim=1)
    full_attn = torch.cat([prompt_attn, gen_attn], dim=1)
    prompt_len = int(prompt_input_ids.size(1))

    leakage = {
        "answer_in_key_fact_1_norm": bool(generated_structure.get("answer_in_key_fact_1_norm")),
        "answer_in_key_fact_2_norm": bool(generated_structure.get("answer_in_key_fact_2_norm")),
        "answer_in_any_key_fact_norm": bool(generated_structure.get("answer_in_any_key_fact_norm")),
        "answer_token_overlap_key_fact_1": float(generated_structure.get("answer_token_overlap_key_fact_1") or 0.0),
        "answer_token_overlap_key_fact_2": float(generated_structure.get("answer_token_overlap_key_fact_2") or 0.0),
    }
    return _compute_hotpot_dependency_probe_from_replay_region(
        tokenizer=tokenizer,
        backbone=backbone,
        replay_text=raw_output,
        replay_input_ids=gen_ids,
        full_input_ids=full_ids,
        full_attention_mask=full_attn,
        replay_token_offset=prompt_len,
        prefix_final=prefix_final,
        answer_char_span=answer_char_span,
        key_fact_1_value_span=key_fact_1_value_span,
        key_fact_2_value_span=key_fact_2_value_span,
        key_fact_1_header_span=key_fact_1_header_span,
        key_fact_2_header_span=key_fact_2_header_span,
        leakage=leakage,
        probe_type="teacher_forced_generated_output",
    )


def _support_fact_overlap(predicted: List[str], gold: List[str]) -> Dict[str, float]:
    """Set-style exact overlap metrics for normalized supporting-fact strings."""
    pred_norm = list(dict.fromkeys(real._normalize_eval_text(item) for item in predicted if str(item).strip()))
    gold_norm = list(dict.fromkeys(real._normalize_eval_text(item) for item in gold if str(item).strip()))
    pred_set = {item for item in pred_norm if item}
    gold_set = {item for item in gold_norm if item}

    if not pred_set and not gold_set:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "em": 1.0,
            "predicted_normalized": [],
            "gold_normalized": [],
        }

    shared = len(pred_set & gold_set)
    precision = float(shared) / float(len(pred_set)) if pred_set else 0.0
    recall = float(shared) / float(len(gold_set)) if gold_set else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    em = 1.0 if pred_set == gold_set else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "em": em,
        "predicted_normalized": sorted(pred_set),
        "gold_normalized": sorted(gold_set),
    }


def _empty_task_stats(task_name: str) -> Dict[str, float]:
    """Return task-specific accumulator state for generated reasoning metrics."""
    if task_name in {"squad2_reasoning", "fever_reasoning"}:
        return real._empty_reasoning_stats()
    if task_name == "hotpotqa_reasoning":
        return {
            "total": 0.0,
            "concrete_pred_total": 0.0,
            "answer_em_sum": 0.0,
            "answer_f1_sum": 0.0,
            "support_precision_sum": 0.0,
            "support_recall_sum": 0.0,
            "support_f1_sum": 0.0,
            "support_em_sum": 0.0,
            "joint_em_sum": 0.0,
            "generated_structure_parse_coverage_sum": 0.0,
            "generated_structure_has_key_fact_1_sum": 0.0,
            "generated_structure_has_key_fact_2_sum": 0.0,
            "generated_structure_has_answer_header_sum": 0.0,
            "generated_structure_all_fields_present_sum": 0.0,
            "generated_structure_key_fact_1_nonempty_sum": 0.0,
            "generated_structure_key_fact_2_nonempty_sum": 0.0,
            "generated_structure_answer_nonempty_sum": 0.0,
            "generated_structure_all_fields_nonempty_sum": 0.0,
            "generated_structure_key_facts_distinct_sum": 0.0,
            "generated_structure_fact_scaffold_present_sum": 0.0,
            "generated_structure_answer_field_present_sum": 0.0,
            "generated_structure_full_keyfacts_contract_present_sum": 0.0,
            "generated_structure_labeled_fact_scaffold_present_sum": 0.0,
            "generated_structure_numbered_fact_scaffold_present_sum": 0.0,
            "generated_structure_state_no_structure_sum": 0.0,
            "generated_structure_state_facts_only_numbered_sum": 0.0,
            "generated_structure_state_facts_only_labeled_sum": 0.0,
            "generated_structure_state_answer_only_sum": 0.0,
            "generated_structure_state_facts_plus_answer_sum": 0.0,
            "generated_structure_state_full_keyfacts_contract_sum": 0.0,
        }
    raise ValueError(f"Unsupported task {task_name!r}")


def _finalize_task_metrics(task_name: str, stats: Dict[str, float], *, system: str) -> Dict[str, float]:
    """Finalize per-system generated reasoning metrics for one task."""
    if task_name in {"squad2_reasoning", "fever_reasoning"}:
        return real._finalize_reasoning_metrics(task_name, stats, system=system)
    if task_name == "hotpotqa_reasoning":
        total = stats["total"]
        metrics = {
            f"{task_name}_coverage_{system}": real._safe_rate(stats["concrete_pred_total"], total),
            f"{task_name}_answer_em_{system}": real._safe_rate(stats["answer_em_sum"], total),
            f"{task_name}_answer_f1_{system}": real._safe_rate(stats["answer_f1_sum"], total),
            f"{task_name}_support_precision_{system}": real._safe_rate(stats["support_precision_sum"], total),
            f"{task_name}_support_recall_{system}": real._safe_rate(stats["support_recall_sum"], total),
            f"{task_name}_support_f1_{system}": real._safe_rate(stats["support_f1_sum"], total),
            f"{task_name}_support_em_{system}": real._safe_rate(stats["support_em_sum"], total),
            f"{task_name}_joint_em_{system}": real._safe_rate(stats["joint_em_sum"], total),
        }
        if total > 0 and stats["generated_structure_parse_coverage_sum"] > 0.0:
            metrics.update(
                {
                    f"{task_name}_generated_structure_parse_coverage_{system}": real._safe_rate(
                        stats["generated_structure_parse_coverage_sum"], total
                    ),
                    f"{task_name}_generated_structure_has_key_fact_1_rate_{system}": real._safe_rate(
                        stats["generated_structure_has_key_fact_1_sum"], total
                    ),
                    f"{task_name}_generated_structure_has_key_fact_2_rate_{system}": real._safe_rate(
                        stats["generated_structure_has_key_fact_2_sum"], total
                    ),
                    f"{task_name}_generated_structure_has_answer_header_rate_{system}": real._safe_rate(
                        stats["generated_structure_has_answer_header_sum"], total
                    ),
                    f"{task_name}_generated_structure_all_fields_present_rate_{system}": real._safe_rate(
                        stats["generated_structure_all_fields_present_sum"], total
                    ),
                    f"{task_name}_generated_structure_key_fact_1_nonempty_rate_{system}": real._safe_rate(
                        stats["generated_structure_key_fact_1_nonempty_sum"], total
                    ),
                    f"{task_name}_generated_structure_key_fact_2_nonempty_rate_{system}": real._safe_rate(
                        stats["generated_structure_key_fact_2_nonempty_sum"], total
                    ),
                    f"{task_name}_generated_structure_answer_nonempty_rate_{system}": real._safe_rate(
                        stats["generated_structure_answer_nonempty_sum"], total
                    ),
                    f"{task_name}_generated_structure_all_fields_nonempty_rate_{system}": real._safe_rate(
                        stats["generated_structure_all_fields_nonempty_sum"], total
                    ),
                    f"{task_name}_generated_structure_key_facts_distinct_rate_{system}": real._safe_rate(
                        stats["generated_structure_key_facts_distinct_sum"], total
                    ),
                    f"{task_name}_generated_structure_fact_scaffold_emergence_rate_{system}": real._safe_rate(
                        stats["generated_structure_fact_scaffold_present_sum"], total
                    ),
                    f"{task_name}_generated_structure_answer_field_emergence_rate_{system}": real._safe_rate(
                        stats["generated_structure_answer_field_present_sum"], total
                    ),
                    f"{task_name}_generated_structure_full_contract_emergence_rate_{system}": real._safe_rate(
                        stats["generated_structure_full_keyfacts_contract_present_sum"], total
                    ),
                    f"{task_name}_generated_structure_labeled_fact_scaffold_rate_{system}": real._safe_rate(
                        stats["generated_structure_labeled_fact_scaffold_present_sum"], total
                    ),
                    f"{task_name}_generated_structure_numbered_fact_scaffold_rate_{system}": real._safe_rate(
                        stats["generated_structure_numbered_fact_scaffold_present_sum"], total
                    ),
                    f"{task_name}_generated_structure_state_no_structure_rate_{system}": real._safe_rate(
                        stats["generated_structure_state_no_structure_sum"], total
                    ),
                    f"{task_name}_generated_structure_state_facts_only_numbered_rate_{system}": real._safe_rate(
                        stats["generated_structure_state_facts_only_numbered_sum"], total
                    ),
                    f"{task_name}_generated_structure_state_facts_only_labeled_rate_{system}": real._safe_rate(
                        stats["generated_structure_state_facts_only_labeled_sum"], total
                    ),
                    f"{task_name}_generated_structure_state_answer_only_rate_{system}": real._safe_rate(
                        stats["generated_structure_state_answer_only_sum"], total
                    ),
                    f"{task_name}_generated_structure_state_facts_plus_answer_rate_{system}": real._safe_rate(
                        stats["generated_structure_state_facts_plus_answer_sum"], total
                    ),
                    f"{task_name}_generated_structure_state_full_keyfacts_contract_rate_{system}": real._safe_rate(
                        stats["generated_structure_state_full_keyfacts_contract_sum"], total
                    ),
                }
            )
        return metrics
    raise ValueError(f"Unsupported task {task_name!r}")


def _canonical_squad2_prediction(text: str) -> str:
    """Canonicalize a free-text SQuAD2 prediction for answer-vs-abstain scoring."""
    line = _first_nonempty_line(text)
    if line.endswith(real.ANSWER_DELIM.strip()):
        return ""
    return line.strip()


def _parse_fever_label(text: str) -> Optional[str]:
    """Map a generated FEVER response back to A/B/C labels when possible."""
    line = _first_nonempty_line(text)
    if not line:
        return None
    upper = line.upper()
    letter_match = re.match(r"^([ABC])(?:[\s\)\.\:\-]|$)", upper)
    if letter_match:
        return str(letter_match.group(1))

    # Some checkpoints emit numeric aliases for the FEVER choices. Accept only
    # standalone 1/2/3-style prefixes so strings like "100 (TV series) ..."
    # do not get misparsed as label A.
    numeric_match = re.match(r"^([123])(?:[\s\)\.\:\-]|$)", line)
    if numeric_match:
        return {"1": "A", "2": "B", "3": "C"}[str(numeric_match.group(1))]

    low = line.lower()
    if "not enough information" in low or "cannot be determined" in low or "cannot be inferred" in low:
        return "C"
    if "refut" in low:
        return "B"
    if "support" in low:
        return "A"
    return None


def _mean(values: List[float]) -> float:
    """Mean helper that returns NaN on empty inputs."""
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _tokenizer_input_ids(encoded: Any) -> List[int]:
    if isinstance(encoded, dict):
        return list(encoded.get("input_ids") or [])
    input_ids = getattr(encoded, "input_ids", None)
    if input_ids is None:
        return []
    return list(input_ids)


def _load_bakeoff_best(run_dir: Path) -> Optional[Dict[str, str]]:
    """Load the top bakeoff row for checkpoint discovery when available."""
    bakeoff = run_dir / "bakeoff_fast_fullval.csv"
    if not bakeoff.exists():
        return None
    with bakeoff.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def _resolve_head_ckpt(run_dir: Optional[Path], explicit: Optional[str]) -> Optional[Path]:
    """Resolve a REAL head checkpoint from CLI input or run artifacts."""
    if explicit:
        return Path(explicit)
    if run_dir is None:
        return None

    best = _load_bakeoff_best(run_dir)
    if best:
        ckpt = (best.get("ckpt") or "").strip()
        if ckpt:
            p = Path(ckpt)
            candidates = [p]
            if not p.is_absolute():
                candidates.append(run_dir / p)
                candidates.append(run_dir / "checkpoints" / p.name)
            for candidate in candidates:
                candidate = maybe_localize_outputs_path(candidate)
                if candidate.exists():
                    return candidate

    ckpts = sorted((run_dir / "checkpoints").glob("head_step*.pt"))
    if ckpts:
        return ckpts[-1]
    return None


def _resolve_static_ckpt(run_dir: Optional[Path], cfg_dict: Dict[str, Any], explicit: Optional[str]) -> Optional[Path]:
    """Resolve a StaticPrefix checkpoint from CLI input, run dir, or config."""
    if explicit:
        return Path(explicit)
    if run_dir is not None:
        default_path = run_dir / "checkpoints" / "static_prefix.pt"
        if default_path.exists():
            return default_path

    cfg_val = cfg_dict.get("static_prefix_ckpt")
    if cfg_val:
        p = maybe_localize_outputs_path(Path(str(cfg_val)))
        if p.exists():
            return p
        if run_dir is not None:
            p2 = run_dir / str(cfg_val)
            if p2.exists():
                return p2
    return None


def _resolve_output_dir(args, run_dir: Optional[Path], run_id: Optional[str]) -> Path:
    """Choose the reasoning artifact directory for this invocation."""
    if args.out_dir:
        return Path(args.out_dir)
    if args.out_json:
        return Path(args.out_json).resolve().parent
    if args.out_jsonl:
        return Path(args.out_jsonl).resolve().parent
    if run_dir is not None:
        return run_dir / "reasoning_eval"
    stem = run_id or time.strftime("reasoning_eval_%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / stem / "reasoning_eval"


def _build_task_bundle(
    task_name: str,
    tokenizer,
    cfg: real.TrainingConfig,
    max_samples_override: Optional[int],
    hotpotqa_output_contract_override: Optional[str] = None,
    hotpotqa_prompt_variant_override: Optional[str] = None,
    hotpotqa_response_cue_variant_override: Optional[str] = None,
) -> TaskBundle:
    """Instantiate one supported reasoning dataset using repo config defaults."""
    if task_name == "squad2_reasoning":
        max_samples = int(max_samples_override or cfg.squad2_reasoning_max_samples)
        dataset = real.SQuAD2ReasoningDataset(
            tokenizer,
            split=cfg.squad2_reasoning_split,
            max_length=cfg.squad2_reasoning_max_length,
            max_samples=max_samples,
            answer_loss_tokens=cfg.squad2_reasoning_loss_tokens,
            abstain_text=cfg.abstain_text,
            dataset_name=cfg.squad2_reasoning_dataset_name,
        )
        return TaskBundle(
            name=task_name,
            dataset=dataset,
            dataset_name=cfg.squad2_reasoning_dataset_name,
            dataset_config_name=None,
            split=cfg.squad2_reasoning_split,
            max_length=cfg.squad2_reasoning_max_length,
            answer_loss_tokens=cfg.squad2_reasoning_loss_tokens,
            benchmark_role="reasoning_measure",
            reasoning_eval_mode=True,
            allow_abstain=True,
            abstain_mode="text",
            abstain_text=cfg.abstain_text,
            abstain_option_letter=None,
            primary_metric_name=f"{task_name}_overall_accuracy_real",
        )

    if task_name == "fever_reasoning":
        max_samples = int(max_samples_override or cfg.fever_reasoning_max_samples)
        dataset = real.FEVERReasoningDataset(
            tokenizer,
            split=cfg.fever_reasoning_split,
            max_length=cfg.fever_reasoning_max_length,
            max_samples=max_samples,
            answer_loss_tokens=cfg.fever_reasoning_loss_tokens,
            dataset_name=cfg.fever_reasoning_dataset_name,
            dataset_config_name=cfg.fever_reasoning_dataset_config_name,
        )
        return TaskBundle(
            name=task_name,
            dataset=dataset,
            dataset_name=cfg.fever_reasoning_dataset_name,
            dataset_config_name=cfg.fever_reasoning_dataset_config_name,
            split=cfg.fever_reasoning_split,
            max_length=cfg.fever_reasoning_max_length,
            answer_loss_tokens=cfg.fever_reasoning_loss_tokens,
            benchmark_role="reasoning_measure",
            reasoning_eval_mode=True,
            allow_abstain=True,
            abstain_mode="native_label",
            abstain_text=real.FEVER_REASONING_NEI_TEXT,
            abstain_option_letter="C",
            primary_metric_name=f"{task_name}_overall_label_accuracy_real",
        )

    if task_name == "hotpotqa_reasoning":
        max_samples = int(max_samples_override or cfg.hotpotqa_reasoning_max_samples)
        output_contract = hotpotqa_output_contract_override or getattr(cfg, "hotpotqa_reasoning_output_contract", "default")
        prompt_variant = hotpotqa_prompt_variant_override or getattr(cfg, "hotpotqa_reasoning_prompt_variant", "default")
        response_cue_variant = hotpotqa_response_cue_variant_override or getattr(
            cfg,
            "hotpotqa_reasoning_response_cue_variant",
            "default",
        )
        dataset = real.HotpotQAReasoningDataset(
            tokenizer,
            split=cfg.hotpotqa_reasoning_split,
            max_length=cfg.hotpotqa_reasoning_max_length,
            max_samples=max_samples,
            answer_loss_tokens=cfg.hotpotqa_reasoning_loss_tokens,
            dataset_name=cfg.hotpotqa_reasoning_dataset_name,
            dataset_config_name=cfg.hotpotqa_reasoning_dataset_config_name,
            output_contract=output_contract,
            prompt_variant=prompt_variant,
            response_cue_variant=response_cue_variant,
        )
        return TaskBundle(
            name=task_name,
            dataset=dataset,
            dataset_name=cfg.hotpotqa_reasoning_dataset_name,
            dataset_config_name=cfg.hotpotqa_reasoning_dataset_config_name,
            split=cfg.hotpotqa_reasoning_split,
            max_length=cfg.hotpotqa_reasoning_max_length,
            answer_loss_tokens=cfg.hotpotqa_reasoning_loss_tokens,
            benchmark_role="basin_probe",
            reasoning_eval_mode=True,
            allow_abstain=False,
            abstain_mode="none",
            abstain_text=None,
            abstain_option_letter=None,
            primary_metric_name=f"{task_name}_answer_em_real",
            output_contract=output_contract,
            prompt_variant=prompt_variant,
            response_cue_variant=response_cue_variant,
        )

    raise ValueError(f"Unsupported task {task_name!r}")


def _build_single_batch(example: Dict[str, Any], pad_id: int) -> real.Batch:
    """Wrap one dataset example in the repo's padded batch structure."""
    return real.collate_batch([example], pad_id=pad_id)


def _prompt_tensors_from_batch(batch: real.Batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slice prompt-only tensors from a single-example batch."""
    prompt_len = int(batch.prompt_len[0].item())
    return batch.input_ids[:, :prompt_len], batch.attention_mask[:, :prompt_len]


def _target_text_from_batch(tokenizer, batch: real.Batch) -> str:
    """Decode the supervised answer segment for logging/export."""
    target_ids = batch.labels[0]
    target_ids = target_ids[target_ids != -100]
    return tokenizer.decode(target_ids.tolist(), skip_special_tokens=True).strip()


def _pressure_payload_from_scores(step_scores: Tuple[torch.Tensor, ...]) -> Optional[Dict[str, Any]]:
    """Convert generate() step scores into a pressure-trace payload."""
    if not step_scores:
        return None
    stacked = torch.stack([score[0].detach().float().cpu() for score in step_scores], dim=0)
    trace = compute_pressure_trace(stacked)
    return _pressure_payload_from_trace(trace)


def _pressure_payload_from_trace(trace: Optional[Dict[str, torch.Tensor]]) -> Optional[Dict[str, Any]]:
    """Convert a pressure-trace tensor dict into the JSON payload form."""
    if not isinstance(trace, dict):
        return None
    summary = summarize_pressure_trace(trace)
    payload: Dict[str, Any] = {"summary": summary}
    for key in ("entropy", "entropy_norm", "gap", "drift", "drift_tv", "pressure", "pressure_norm"):
        value = trace.get(key)
        if isinstance(value, torch.Tensor):
            payload[key] = [float(x) for x in value.tolist()]
    return payload


def _generated_ids_from_generate(gen_out) -> torch.Tensor:
    """Extract generated token ids using the actual generation-step count."""
    gen_steps = len(tuple(getattr(gen_out, "scores", ()) or ()))
    sequences = gen_out.sequences.detach().cpu()
    if gen_steps <= 0:
        return sequences[:, 0:0]
    return sequences[:, -gen_steps:]


def _decode_generated_text(tokenizer, generated_ids: torch.Tensor) -> str:
    """Decode only the generated continuation."""
    if generated_ids.ndim == 2:
        generated_ids = generated_ids[0]
    return tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)


@_lazy_no_grad
def _compute_hotpot_boundary_transition_probes(
    *,
    tokenizer,
    backbone,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    generated_ids: torch.Tensor,
    prefix_final: Optional[torch.Tensor],
    meta: Dict[str, Any],
    raw_output: str,
) -> Dict[str, Any]:
    """Replay local Hotpot keyfacts boundaries under value-span occlusion."""
    contract = normalize_hotpot_reasoning_output_contract(meta.get("hotpotqa_reasoning_output_contract"))
    if contract != "keyfacts":
        return {"missing_reason": "unsupported_contract"}

    if generated_ids.ndim == 1:
        generated_ids = generated_ids.unsqueeze(0)
    if generated_ids.numel() == 0:
        return {"missing_reason": "empty_generation"}

    generated_structure = _compute_hotpot_generated_structure(meta, raw_output)
    char_spans = dict(generated_structure.get("char_spans") or {})

    prompt_ids = prompt_input_ids.detach().cpu()
    prompt_attn = prompt_attention_mask.detach().cpu()
    gen_ids = generated_ids.detach().cpu().long()
    gen_attn = torch.ones_like(gen_ids, dtype=prompt_attn.dtype)
    full_ids = torch.cat([prompt_ids, gen_ids], dim=1)
    full_attn = torch.cat([prompt_attn, gen_attn], dim=1)
    prompt_len = int(prompt_input_ids.size(1))

    boundary_specs = {
        "key_fact_1_to_key_fact_2": {
            "source_label": "key_fact_1_value_span",
            "score_label": "key_fact_2_header_span",
            "source_char_span": _char_span_tuple(char_spans.get("key_fact_1_value_span")),
            "score_char_span": _char_span_tuple(char_spans.get("key_fact_2_header_span")),
        },
        "key_fact_2_to_answer": {
            "source_label": "key_fact_2_value_span",
            "score_label": "answer_header_span",
            "source_char_span": _char_span_tuple(char_spans.get("key_fact_2_value_span")),
            "score_char_span": _char_span_tuple(char_spans.get("answer_header_span")),
        },
    }

    out: Dict[str, Any] = {}
    for boundary_name, spec in boundary_specs.items():
        source_char_span = spec["source_char_span"]
        score_char_span = spec["score_char_span"]
        if source_char_span is None:
            out[boundary_name] = {"missing_reason": f"missing_{spec['source_label']}"}
            continue
        if score_char_span is None:
            out[boundary_name] = {"missing_reason": f"missing_{spec['score_label']}"}
            continue

        score_token_span_rel = token_span_from_char_span(tokenizer, raw_output, score_char_span, gen_ids)
        if score_token_span_rel is None:
            out[boundary_name] = {"missing_reason": f"{boundary_name}_score_span_to_tokens_failed"}
            continue
        source_token_span_rel = token_span_from_char_span(tokenizer, raw_output, source_char_span, gen_ids)
        if source_token_span_rel is None:
            out[boundary_name] = {"missing_reason": f"{boundary_name}_source_span_to_tokens_failed"}
            continue

        score_full_span = (
            int(prompt_len + score_token_span_rel[0]),
            int(prompt_len + score_token_span_rel[1]),
        )
        source_full_span = (
            int(prompt_len + source_token_span_rel[0]),
            int(prompt_len + source_token_span_rel[1]),
        )
        occlude_mask = make_token_mask(int(full_ids.size(1)), [source_full_span])
        probe = compute_masked_replay_probe(
            backbone=backbone,
            input_ids=full_ids,
            attention_mask=full_attn,
            prefix_final=prefix_final,
            score_token_span=score_full_span,
            occlude_token_mask=occlude_mask,
            trace_prefix="boundary",
            target_token_ids=gen_ids[:, score_token_span_rel[0] : score_token_span_rel[1]],
        )
        if "summary" not in probe:
            out[boundary_name] = {
                "missing_reason": str(probe.get("missing_reason") or f"{boundary_name}_replay_failed")
            }
            continue

        out[boundary_name] = {
            "boundary_name": boundary_name,
            "source_field": str(spec["source_label"]),
            "score_field": str(spec["score_label"]),
            "score_text": str(raw_output[score_char_span[0] : score_char_span[1]]),
            "source_text": str(raw_output[source_char_span[0] : source_char_span[1]]),
            "score_char_span": [int(score_char_span[0]), int(score_char_span[1])],
            "source_char_span": [int(source_char_span[0]), int(source_char_span[1])],
            "score_token_span": [int(score_full_span[0]), int(score_full_span[1])],
            "score_token_span_replay_region": [int(score_token_span_rel[0]), int(score_token_span_rel[1])],
            "source_token_span": [int(source_full_span[0]), int(source_full_span[1])],
            "source_token_span_replay_region": [int(source_token_span_rel[0]), int(source_token_span_rel[1])],
            "summary": probe["summary"],
            "boundary_kl": list(probe.get("kl") or []),
            "boundary_js": list(probe.get("js") or []),
            "boundary_token_logprob_drop": list(probe.get("token_logprob_drop") or []),
            "boundary_gap_drop": list(probe.get("gap_drop") or []),
            "boundary_top1_flip": list(probe.get("top1_flip") or []),
            "occluded_token_spans": [[int(source_full_span[0]), int(source_full_span[1])]],
            "occluded_token_spans_replay_region": [
                [int(source_token_span_rel[0]), int(source_token_span_rel[1])]
            ],
        }

    return {
        "probe_type": "teacher_forced_generated_boundary",
        "contract": "keyfacts",
        "generated_structure_state": str(generated_structure.get("structure_state") or "no_structure"),
        "boundaries": out,
    }


def _generate_with_pressure(
    tokenizer,
    backbone,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    *,
    prefix_final: Optional[torch.Tensor],
    max_new_tokens: int,
    min_new_tokens: int,
    entry_transition_probe: bool,
    generation_trace: bool,
    generation_trace_topk: int,
) -> Dict[str, Any]:
    """Greedy generation plus per-step pressure probe, with no behavior change."""
    emb_dev = backbone.get_input_embeddings().weight.device
    model_dtype = getattr(backbone, "dtype", torch.float16)
    pad = tokenizer.pad_token_id

    if prefix_final is None:
        gen_out = backbone.generate(
            input_ids=prompt_input_ids.to(emb_dev),
            attention_mask=prompt_attention_mask.to(emb_dev),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            pad_token_id=pad,
            output_scores=True,
            return_dict_in_generate=True,
        )
    else:
        emb = backbone.get_input_embeddings()(prompt_input_ids.to(emb_dev)).to(model_dtype)
        prefix_final = prefix_final.to(model_dtype)
        inputs_embeds, attn, _labels_unused = build_inputs_embeds_with_prefix(
            emb=emb,
            attention_mask=prompt_attention_mask.to(emb_dev),
            labels=None,
            prefix=prefix_final,
        )
        gen_out = backbone.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            pad_token_id=pad,
            output_scores=True,
            return_dict_in_generate=True,
        )

    generated_ids = _generated_ids_from_generate(gen_out)
    step_scores = tuple(gen_out.scores)
    pressure_trace = None
    if step_scores:
        stacked = torch.stack([score[0].detach().float().cpu() for score in step_scores], dim=0)
        pressure_trace = compute_pressure_trace(stacked)
    prompt_token_count = int(prompt_input_ids.size(1))
    return {
        "raw_output": _decode_generated_text(tokenizer, generated_ids),
        "generated_ids": generated_ids,
        "pressure_probe": _pressure_payload_from_trace(pressure_trace),
        "entry_probe": (
            compute_entry_transition_probe_from_scores(
                tokenizer,
                step_scores,
                generated_ids,
                prompt_token_count=prompt_token_count,
                top_k=int(generation_trace_topk),
                pressure_trace=pressure_trace,
            )
            if entry_transition_probe
            else None
        ),
        "generation_trace": (
            compute_generation_trace_from_scores(
                tokenizer,
                step_scores,
                generated_ids,
                prompt_token_count=prompt_token_count,
                top_k=int(generation_trace_topk),
                pressure_trace=pressure_trace,
            )
            if generation_trace
            else None
        ),
    }


def _evaluate_base(
    tokenizer,
    backbone,
    batch: real.Batch,
    *,
    task_name: str,
    prompt_text: str,
    meta: Optional[Dict[str, Any]],
    entry_transition_probe: bool,
    generation_trace: bool,
    generation_trace_topk: int,
    context_pull_probe: bool,
    dependency_probe_gold: bool,
    dependency_probe_free: bool,
    max_new_tokens: int,
    min_new_tokens: int,
) -> Dict[str, Any]:
    """Run base-model generation plus continuity loss for one example."""
    prompt_input_ids, prompt_attention_mask = _prompt_tensors_from_batch(batch)
    gen = _generate_with_pressure(
        tokenizer,
        backbone,
        prompt_input_ids,
        prompt_attention_mask,
        prefix_final=None,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        entry_transition_probe=entry_transition_probe,
        generation_trace=generation_trace,
        generation_trace_topk=generation_trace_topk,
    )
    context_probe = None
    if context_pull_probe:
        context_probe = compute_context_pull_probe(
            task_name=task_name,
            tokenizer=tokenizer,
            backbone=backbone,
            prompt_text=prompt_text,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            generated_ids=gen["generated_ids"],
            prefix_final=None,
            meta=meta,
        )
    dependency_probe = None
    if dependency_probe_gold and task_name == "hotpotqa_reasoning":
        dependency_probe = _compute_hotpot_gold_dependency_probe(
            tokenizer=tokenizer,
            backbone=backbone,
            prompt_text=prompt_text,
            prompt_input_ids=prompt_input_ids,
            prefix_final=None,
            meta=dict(meta or {}),
        )
    dependency_probe_free_payload = None
    if dependency_probe_free and task_name == "hotpotqa_reasoning":
        dependency_probe_free_payload = _compute_hotpot_free_dependency_probe(
            tokenizer=tokenizer,
            backbone=backbone,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            generated_ids=gen["generated_ids"],
            prefix_final=None,
            meta=dict(meta or {}),
            raw_output=gen["raw_output"],
        )
    entry_transition_payload = None
    if entry_transition_probe:
        entry_transition_payload = {"entry": gen.get("entry_probe")}
        if task_name == "hotpotqa_reasoning":
            entry_transition_payload.update(
                _compute_hotpot_boundary_transition_probes(
                    tokenizer=tokenizer,
                    backbone=backbone,
                    prompt_input_ids=prompt_input_ids,
                    prompt_attention_mask=prompt_attention_mask,
                    generated_ids=gen["generated_ids"],
                    prefix_final=None,
                    meta=dict(meta or {}),
                    raw_output=gen["raw_output"],
                )
            )
    return {
        "raw_output": gen["raw_output"],
        "generated_ids": gen["generated_ids"],
        "pressure_probe": gen.get("pressure_probe"),
        "entry_transition_probe": entry_transition_payload,
        "generation_trace": gen.get("generation_trace"),
        "context_pull_probe": context_probe,
        "dependency_probe_gold": dependency_probe,
        "dependency_probe_free": dependency_probe_free_payload,
        "loss": _safe_float(real.baseline_loss_mean(backbone, batch).item()),
    }


def _evaluate_static(
    tokenizer,
    backbone,
    batch: real.Batch,
    static_model: real.StaticPrefixWrapper,
    *,
    task_name: str,
    prompt_text: str,
    meta: Optional[Dict[str, Any]],
    entry_transition_probe: bool,
    generation_trace: bool,
    generation_trace_topk: int,
    context_pull_probe: bool,
    dependency_probe_gold: bool,
    dependency_probe_free: bool,
    max_new_tokens: int,
    min_new_tokens: int,
) -> Dict[str, Any]:
    """Run StaticPrefix generation plus continuity loss for one example."""
    prompt_input_ids, prompt_attention_mask = _prompt_tensors_from_batch(batch)
    prefix_static = static_model.static_prefix(1)
    gen = _generate_with_pressure(
        tokenizer,
        backbone,
        prompt_input_ids,
        prompt_attention_mask,
        prefix_final=prefix_static,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        entry_transition_probe=entry_transition_probe,
        generation_trace=generation_trace,
        generation_trace_topk=generation_trace_topk,
    )
    context_probe = None
    if context_pull_probe:
        context_probe = compute_context_pull_probe(
            task_name=task_name,
            tokenizer=tokenizer,
            backbone=backbone,
            prompt_text=prompt_text,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            generated_ids=gen["generated_ids"],
            prefix_final=prefix_static,
            meta=meta,
        )
    dependency_probe = None
    if dependency_probe_gold and task_name == "hotpotqa_reasoning":
        dependency_probe = _compute_hotpot_gold_dependency_probe(
            tokenizer=tokenizer,
            backbone=backbone,
            prompt_text=prompt_text,
            prompt_input_ids=prompt_input_ids,
            prefix_final=prefix_static,
            meta=dict(meta or {}),
        )
    dependency_probe_free_payload = None
    if dependency_probe_free and task_name == "hotpotqa_reasoning":
        dependency_probe_free_payload = _compute_hotpot_free_dependency_probe(
            tokenizer=tokenizer,
            backbone=backbone,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            generated_ids=gen["generated_ids"],
            prefix_final=prefix_static,
            meta=dict(meta or {}),
            raw_output=gen["raw_output"],
        )
    entry_transition_payload = None
    if entry_transition_probe:
        entry_transition_payload = {"entry": gen.get("entry_probe")}
        if task_name == "hotpotqa_reasoning":
            entry_transition_payload.update(
                _compute_hotpot_boundary_transition_probes(
                    tokenizer=tokenizer,
                    backbone=backbone,
                    prompt_input_ids=prompt_input_ids,
                    prompt_attention_mask=prompt_attention_mask,
                    generated_ids=gen["generated_ids"],
                    prefix_final=prefix_static,
                    meta=dict(meta or {}),
                    raw_output=gen["raw_output"],
                )
            )
    return {
        "raw_output": gen["raw_output"],
        "generated_ids": gen["generated_ids"],
        "pressure_probe": gen.get("pressure_probe"),
        "entry_transition_probe": entry_transition_payload,
        "generation_trace": gen.get("generation_trace"),
        "context_pull_probe": context_probe,
        "dependency_probe_gold": dependency_probe,
        "dependency_probe_free": dependency_probe_free_payload,
        "loss": _safe_float(static_model(batch).item()),
    }


def _evaluate_real(
    tokenizer,
    backbone,
    batch: real.Batch,
    head: real.REALHead,
    *,
    task_name: str,
    prompt_text: str,
    meta: Optional[Dict[str, Any]],
    entry_transition_probe: bool,
    generation_trace: bool,
    generation_trace_topk: int,
    context_pull_probe: bool,
    real_trace_probe: bool,
    real_trace_probe_dump_tensors: bool,
    real_trace_probe_max_tensor_rows: int,
    dependency_probe_gold: bool,
    dependency_probe_free: bool,
    refine_policy: str,
    refine_max_steps: Optional[int],
    refine_energy_delta_tol: float,
    refine_patience: int,
    max_new_tokens: int,
    min_new_tokens: int,
    trace_energy: bool,
) -> Dict[str, Any]:
    """Run REAL generation plus continuity loss and optional energy trace."""
    prompt_input_ids, prompt_attention_mask = _prompt_tensors_from_batch(batch)
    trace_payload = None
    if real_trace_probe:
        emb_dev = backbone.get_input_embeddings().weight.device
        input_ids = batch.input_ids.to(emb_dev)
        attention_mask = batch.attention_mask.to(emb_dev)
        prompt_len = batch.prompt_len.to(emb_dev)
        positions = torch.arange(input_ids.size(1), device=emb_dev).unsqueeze(0)
        prompt_mask = (positions < prompt_len.unsqueeze(1)) & attention_mask.bool()
        trace = infer.compute_real_prefix_trace_from_ids(
            backbone,
            head,
            input_ids,
            attention_mask,
            pool_mask=prompt_mask,
            refine_policy=refine_policy,
            refine_max_steps=refine_max_steps,
            energy_delta_tol=refine_energy_delta_tol,
            patience=refine_patience,
        )
        prefix_real = trace["prefix_final"]
        energy_trace_t = trace["energy_trace"]
        steps_used_t = trace["steps_used"]
        trace_payload = summarize_real_trace(
            energy_trace=energy_trace_t[0],
            z_traj=trace["z_traj"][0] if isinstance(trace.get("z_traj"), torch.Tensor) else None,
            prefix_traj=trace["prefix_traj"][0] if isinstance(trace.get("prefix_traj"), torch.Tensor) else None,
            selected_step=int(steps_used_t[0].item()),
            final_step=(
                int(trace["prefix_traj"].size(1) - 1)
                if isinstance(trace.get("prefix_traj"), torch.Tensor)
                else None
            ),
        )
        trace_payload["real_trace_tensor_dump_enabled"] = bool(real_trace_probe_dump_tensors)
        trace_payload["real_trace_tensor_dump_max_rows"] = int(max(0, real_trace_probe_max_tensor_rows))
        trace_payload["real_trace_tensor_dump_included"] = False
    else:
        prefix_real, energy_trace_t, steps_used_t = infer.compute_real_prefix_and_trace(
            backbone,
            head,
            batch,
            refine_policy=refine_policy,
            refine_max_steps=refine_max_steps,
            energy_delta_tol=refine_energy_delta_tol,
            patience=refine_patience,
        )
    gen = _generate_with_pressure(
        tokenizer,
        backbone,
        prompt_input_ids,
        prompt_attention_mask,
        prefix_final=prefix_real,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        entry_transition_probe=entry_transition_probe,
        generation_trace=generation_trace,
        generation_trace_topk=generation_trace_topk,
    )
    context_probe = None
    if context_pull_probe:
        context_probe = compute_context_pull_probe(
            task_name=task_name,
            tokenizer=tokenizer,
            backbone=backbone,
            prompt_text=prompt_text,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            generated_ids=gen["generated_ids"],
            prefix_final=prefix_real,
            meta=meta,
        )
    dependency_probe = None
    if dependency_probe_gold and task_name == "hotpotqa_reasoning":
        dependency_probe = _compute_hotpot_gold_dependency_probe(
            tokenizer=tokenizer,
            backbone=backbone,
            prompt_text=prompt_text,
            prompt_input_ids=prompt_input_ids,
            prefix_final=prefix_real,
            meta=dict(meta or {}),
        )
    dependency_probe_free_payload = None
    if dependency_probe_free and task_name == "hotpotqa_reasoning":
        dependency_probe_free_payload = _compute_hotpot_free_dependency_probe(
            tokenizer=tokenizer,
            backbone=backbone,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            generated_ids=gen["generated_ids"],
            prefix_final=prefix_real,
            meta=dict(meta or {}),
            raw_output=gen["raw_output"],
        )
    entry_transition_payload = None
    if entry_transition_probe:
        entry_transition_payload = {"entry": gen.get("entry_probe")}
        if task_name == "hotpotqa_reasoning":
            entry_transition_payload.update(
                _compute_hotpot_boundary_transition_probes(
                    tokenizer=tokenizer,
                    backbone=backbone,
                    prompt_input_ids=prompt_input_ids,
                    prompt_attention_mask=prompt_attention_mask,
                    generated_ids=gen["generated_ids"],
                    prefix_final=prefix_real,
                    meta=dict(meta or {}),
                    raw_output=gen["raw_output"],
                )
            )
    out = {
        "raw_output": gen["raw_output"],
        "generated_ids": gen["generated_ids"],
        "pressure_probe": gen.get("pressure_probe"),
        "entry_transition_probe": entry_transition_payload,
        "generation_trace": gen.get("generation_trace"),
        "context_pull_probe": context_probe,
        "real_trace_probe": trace_payload,
        "dependency_probe_gold": dependency_probe,
        "dependency_probe_free": dependency_probe_free_payload,
        "loss": _safe_float(infer.loss_with_prefix(backbone, batch, prefix_real)),
        "steps_used": int(steps_used_t[0].item()),
    }
    if trace_energy:
        out["pred_energy"] = [float(x) for x in energy_trace_t[0].detach().cpu().tolist()]
    return out


def _score_squad2_prediction(meta: Dict[str, Any], raw_output: str) -> Dict[str, Any]:
    """Score one SQuAD2 prediction against answer-vs-abstain semantics."""
    pred_text = _canonical_squad2_prediction(raw_output)
    pred_norm = real._normalize_eval_text(pred_text)
    abstain_text = str(meta.get("abstain_text") or real.DEFAULT_CONTEXT_ABSTAIN_TEXT)
    abstain_norm = real._normalize_eval_text(abstain_text)
    gold_answers = [str(x) for x in (meta.get("gold_answers") or []) if str(x).strip()]
    gold_norms = [real._normalize_eval_text(x) for x in gold_answers]
    pred_is_abstain = bool(pred_norm) and pred_norm == abstain_norm
    is_answerable = bool(meta.get("is_answerable"))
    is_correct = pred_is_abstain if not is_answerable else (bool(pred_norm) and pred_norm in gold_norms)
    return {
        "parsed_prediction": pred_text,
        "prediction_normalized": pred_norm,
        "is_abstain": pred_is_abstain,
        "committed": not pred_is_abstain,
        "is_correct": bool(is_correct),
        "gold_answers": gold_answers,
        "gold_answers_normalized": gold_norms,
    }


def _score_fever_prediction(meta: Dict[str, Any], raw_output: str) -> Dict[str, Any]:
    """Score one FEVER prediction against supported/refuted/NEI labels."""
    pred_label = _parse_fever_label(raw_output)
    abstain_label = str(meta.get("abstain_option_letter") or "C")
    gold_label = str(meta.get("gold_label") or "")
    reasoning_label = str(meta.get("reasoning_label") or "")
    pred_is_abstain = pred_label == abstain_label
    return {
        "parsed_prediction": pred_label,
        "is_abstain": pred_is_abstain,
        "committed": not pred_is_abstain,
        "is_correct": bool(pred_label == gold_label),
        "gold_label": gold_label,
        "reasoning_label": reasoning_label,
    }


def _score_hotpot_prediction(meta: Dict[str, Any], raw_output: str) -> Dict[str, Any]:
    """Score structured HotpotQA output using answer EM/F1 plus support overlap."""
    generated_structure = _compute_hotpot_generated_structure(meta, raw_output)
    predicted_support = [
        str(item).strip() for item in (generated_structure.get("parsed_supporting_facts") or []) if str(item).strip()
    ]
    predicted_answer = str(generated_structure.get("parsed_answer") or "").strip()
    gold_answer = str(meta.get("gold_answer") or "")
    gold_support = [str(item) for item in (meta.get("gold_supporting_facts") or []) if str(item).strip()]

    answer_em = float(bool(predicted_answer) and real._normalize_eval_text(predicted_answer) == real._normalize_eval_text(gold_answer))
    answer_f1 = _normalized_token_f1(predicted_answer, gold_answer)
    support_scores = _support_fact_overlap(predicted_support, gold_support)
    joint_em = float(answer_em * support_scores["em"])
    return {
        "parsed_prediction": predicted_answer,
        "parsed_supporting_facts": predicted_support,
        "is_abstain": False,
        "committed": True,
        "is_correct": bool(answer_em),
        "answer_em": answer_em,
        "answer_f1": answer_f1,
        "support_precision": support_scores["precision"],
        "support_recall": support_scores["recall"],
        "support_f1": support_scores["f1"],
        "support_em": support_scores["em"],
        "joint_em": joint_em,
        "gold_answer": gold_answer,
        "gold_supporting_facts": gold_support,
        "prediction_normalized": real._normalize_eval_text(predicted_answer),
        "gold_answer_normalized": real._normalize_eval_text(gold_answer),
        "predicted_supporting_facts_normalized": support_scores["predicted_normalized"],
        "gold_supporting_facts_normalized": support_scores["gold_normalized"],
        "generated_structure": generated_structure,
    }


def _score_prediction(task_name: str, meta: Dict[str, Any], raw_output: str) -> Dict[str, Any]:
    """Dispatch task-specific prediction scoring."""
    if task_name == "squad2_reasoning":
        return _score_squad2_prediction(meta, raw_output)
    if task_name == "fever_reasoning":
        return _score_fever_prediction(meta, raw_output)
    if task_name == "hotpotqa_reasoning":
        return _score_hotpot_prediction(meta, raw_output)
    raise ValueError(f"Unsupported task {task_name!r}")


def _update_reasoning_stats(task_name: str, stats: Dict[str, float], meta: Dict[str, Any], raw_output: str) -> None:
    """Project one generated prediction into the shared reasoning counters."""
    if task_name == "squad2_reasoning":
        pred_text = _canonical_squad2_prediction(raw_output)
        real._update_squad2_reasoning_stats(stats, [meta], [pred_text])
        return
    if task_name == "fever_reasoning":
        pred_label = _parse_fever_label(raw_output)
        real._update_fever_reasoning_stats(stats, [meta], [pred_label])
        return
    if task_name == "hotpotqa_reasoning":
        scored = _score_hotpot_prediction(meta, raw_output)
        stats["total"] += 1.0
        stats["concrete_pred_total"] += 1.0
        stats["answer_em_sum"] += float(scored["answer_em"])
        stats["answer_f1_sum"] += float(scored["answer_f1"])
        stats["support_precision_sum"] += float(scored["support_precision"])
        stats["support_recall_sum"] += float(scored["support_recall"])
        stats["support_f1_sum"] += float(scored["support_f1"])
        stats["support_em_sum"] += float(scored["support_em"])
        stats["joint_em_sum"] += float(scored["joint_em"])
        generated_structure = scored.get("generated_structure")
        if isinstance(generated_structure, dict) and normalize_hotpot_reasoning_output_contract(
            meta.get("hotpotqa_reasoning_output_contract")
        ) == "keyfacts":
            stats["generated_structure_parse_coverage_sum"] += 1.0
            stats["generated_structure_has_key_fact_1_sum"] += float(bool(generated_structure.get("has_key_fact_1")))
            stats["generated_structure_has_key_fact_2_sum"] += float(bool(generated_structure.get("has_key_fact_2")))
            stats["generated_structure_has_answer_header_sum"] += float(bool(generated_structure.get("has_answer_header")))
            stats["generated_structure_all_fields_present_sum"] += float(bool(generated_structure.get("all_fields_present")))
            stats["generated_structure_key_fact_1_nonempty_sum"] += float(bool(generated_structure.get("key_fact_1_nonempty")))
            stats["generated_structure_key_fact_2_nonempty_sum"] += float(bool(generated_structure.get("key_fact_2_nonempty")))
            stats["generated_structure_answer_nonempty_sum"] += float(bool(generated_structure.get("answer_nonempty")))
            stats["generated_structure_all_fields_nonempty_sum"] += float(bool(generated_structure.get("all_fields_nonempty")))
            stats["generated_structure_key_facts_distinct_sum"] += float(bool(generated_structure.get("key_facts_distinct")))
            stats["generated_structure_fact_scaffold_present_sum"] += float(bool(generated_structure.get("fact_scaffold_present")))
            stats["generated_structure_answer_field_present_sum"] += float(bool(generated_structure.get("answer_field_present")))
            stats["generated_structure_full_keyfacts_contract_present_sum"] += float(
                bool(generated_structure.get("full_keyfacts_contract_present"))
            )
            stats["generated_structure_labeled_fact_scaffold_present_sum"] += float(
                bool(generated_structure.get("labeled_fact_scaffold_present"))
            )
            stats["generated_structure_numbered_fact_scaffold_present_sum"] += float(
                bool(generated_structure.get("numbered_fact_scaffold_present"))
            )
            state = str(generated_structure.get("structure_state") or "no_structure")
            stats[f"generated_structure_state_{state}_sum"] += 1.0
        return
    raise ValueError(f"Unsupported task {task_name!r}")


def _task_summary_from_metrics(task_name: str, metrics: Dict[str, Any], sample_count: int, systems: List[str]) -> Dict[str, Any]:
    """Select the task headline metric for the active systems and package summary fields."""
    preferred_system = "real" if "real" in systems else ("static" if "static" in systems else "base")
    metric_stem = {
        "squad2_reasoning": "overall_accuracy",
        "fever_reasoning": "overall_label_accuracy",
        "hotpotqa_reasoning": "answer_em",
    }[task_name]
    primary_metric_name = f"{task_name}_{metric_stem}_{preferred_system}"
    primary_metric_value = _safe_float(metrics.get(primary_metric_name))
    return {
        "sample_count": int(sample_count),
        "systems": systems,
        "primary_metric_family": "reasoning_decision",
        "primary_metric_name": primary_metric_name,
        "primary_metric_value": primary_metric_value,
        "metrics": metrics,
    }


def _resolve_selected_examples(
    bundle: TaskBundle,
    *,
    requested_sample_ids: Optional[List[str]],
) -> Tuple[List[Tuple[int, Dict[str, Any]]], Dict[str, Any]]:
    """Resolve the examples to score, preserving dataset order."""
    if not requested_sample_ids:
        return (
            [(idx, bundle.dataset[idx]) for idx in range(len(bundle.dataset))],
            {
                "enabled": False,
                "requested_count": 0,
                "matched_count": 0,
                "missing_count": 0,
                "selected_row_count": int(len(bundle.dataset)),
            },
        )

    requested = _dedupe_preserve_order([str(item).strip() for item in requested_sample_ids])
    requested_set = set(requested)
    selected: List[Tuple[int, Dict[str, Any]]] = []
    matched_ids: List[str] = []
    matched_seen = set()
    for idx in range(len(bundle.dataset)):
        example = bundle.dataset[idx]
        meta = dict(example.get("meta") or {})
        sample_id = str(meta.get("sample_id") or "").strip()
        if sample_id and sample_id in requested_set:
            selected.append((idx, example))
            if sample_id not in matched_seen:
                matched_seen.add(sample_id)
                matched_ids.append(sample_id)

    missing_ids = [sample_id for sample_id in requested if sample_id not in matched_seen]
    if not selected:
        raise SystemExit(
            f"--sample_ids_path matched zero rows for task `{bundle.name}` "
            f"(requested={len(requested)} missing={len(missing_ids)})"
        )

    return selected, {
        "enabled": True,
        "requested_count": int(len(requested)),
        "matched_count": int(len(matched_ids)),
        "missing_count": int(len(missing_ids)),
        "selected_row_count": int(len(selected)),
        "missing_sample_ids": missing_ids,
    }


def _build_protocol(
    bundle: TaskBundle,
    *,
    mode: str,
    batch_size: int,
    sample_count: int,
    refine_policy: str,
    refine_max_steps: Optional[int],
    refine_energy_delta_tol: float,
    refine_patience: int,
    seed: int,
    primary_metric_name: str,
    sample_ids_path: Optional[str],
    sample_id_filter_info: Optional[Dict[str, Any]],
    entry_transition_probe: bool,
    context_pull_probe: bool,
    real_trace_probe: bool,
    real_trace_probe_dump_tensors: bool,
) -> EvalProtocol:
    """Build the role-aware protocol sidecar for one reasoning task."""
    if real_trace_probe:
        manifold_contact_evidence_level = "trajectory_observed"
    elif context_pull_probe or entry_transition_probe:
        manifold_contact_evidence_level = "manifold_contact_observed"
    else:
        manifold_contact_evidence_level = "snapshot_only"

    if mode in {"base", "static"}:
        evaluation_regime = mode
        dynamic_signal_active = False
        regime_role = "baseline"
    else:
        evaluation_regime = "real"
        dynamic_signal_active = True
        regime_role = "protagonist"

    return EvalProtocol(
        dataset_name=bundle.dataset_name,
        dataset_config_name=bundle.dataset_config_name,
        split=bundle.split,
        eval_batches=int(sample_count),
        batch_size=int(batch_size),
        max_length=int(bundle.max_length),
        refine_policy=str(refine_policy),
        refine_max_steps=int(refine_max_steps) if refine_max_steps is not None else None,
        refine_energy_delta_tol=float(refine_energy_delta_tol) if refine_policy not in {"forward", "fixed", "fixed_steps"} else None,
        refine_patience=int(refine_patience) if refine_policy not in {"forward", "fixed", "fixed_steps"} else None,
        prefix_norm_clamp=0.0,
        prefix_norm_clamp_applied=False,
        dtype_mode="auto",
        dtype_used=None,
        quantization="bnb_4bit",
        answer_loss_tokens=int(bundle.answer_loss_tokens),
        benchmark_role=bundle.benchmark_role,
        reasoning_eval_mode=bundle.reasoning_eval_mode,
        allow_abstain=bundle.allow_abstain,
        abstain_mode=bundle.abstain_mode,
        abstain_text=bundle.abstain_text,
        abstain_option_letter=bundle.abstain_option_letter,
        primary_metric_family="reasoning_decision",
        primary_metric_name=primary_metric_name,
        evaluation_regime=evaluation_regime,
        dynamic_signal_active=dynamic_signal_active,
        frozen_manifold_modified=False,
        regime_role=regime_role,
        manifold_contact_evidence_level=manifold_contact_evidence_level,
        seed=int(seed),
        deterministic=True,
        sample_ids_path=sample_ids_path,
        sample_id_filter_requested_count=(
            int(sample_id_filter_info.get("requested_count"))
            if isinstance(sample_id_filter_info, dict) and sample_id_filter_info.get("enabled")
            else None
        ),
        sample_id_filter_matched_count=(
            int(sample_id_filter_info.get("matched_count"))
            if isinstance(sample_id_filter_info, dict) and sample_id_filter_info.get("enabled")
            else None
        ),
        sample_id_filter_missing_count=(
            int(sample_id_filter_info.get("missing_count"))
            if isinstance(sample_id_filter_info, dict) and sample_id_filter_info.get("enabled")
            else None
        ),
        sample_id_filter_selected_row_count=(
            int(sample_id_filter_info.get("selected_row_count"))
            if isinstance(sample_id_filter_info, dict) and sample_id_filter_info.get("enabled")
            else None
        ),
        real_trace_probe=bool(real_trace_probe),
        real_trace_probe_schema_version=1 if real_trace_probe else None,
        real_trace_probe_dump_tensors=bool(real_trace_probe_dump_tensors),
        real_trace_probe_behavior_change=False,
    )


def evaluate_task(
    bundle: TaskBundle,
    *,
    tokenizer,
    backbone,
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
    entry_transition_probe: bool,
    generation_trace: bool,
    generation_trace_topk: int,
    context_pull_probe: bool,
    real_trace_probe: bool,
    real_trace_probe_dump_tensors: bool,
    real_trace_probe_max_tensor_rows: int,
    dependency_probe_gold: bool,
    dependency_probe_free: bool,
    requested_sample_ids: Optional[List[str]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluate one reasoning task and return task summary plus per-sample rows."""
    want_base = mode in {"base", "all"}
    want_static = mode in {"static", "all"} and static_model is not None
    want_real = mode in {"real", "all"} and head is not None
    systems = []
    if want_base:
        systems.append("base")
    if want_static:
        systems.append("static")
    if want_real:
        systems.append("real")

    pad_id = tokenizer.pad_token_id
    rows: List[Dict[str, Any]] = []
    stats_by_system = {system: _empty_task_stats(bundle.name) for system in systems}
    loss_by_system: Dict[str, List[float]] = {system: [] for system in systems}
    pressure_summary_by_system: Dict[str, Dict[str, List[float]]] = {system: {} for system in systems}
    entry_probe_summary_by_system: Dict[str, Dict[str, List[float]]] = {system: {} for system in systems}
    entry_probe_counts_by_system: Dict[str, Dict[str, int]] = {
        system: {"rows": 0, "missing_rows": 0} for system in systems
    }
    entry_probe_missing_by_system: Dict[str, Counter[str]] = {system: Counter() for system in systems}
    boundary_probe_summary_by_system: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        system: {name: {} for name in HOTPOT_ENTRY_BOUNDARY_NAMES} for system in systems
    }
    boundary_probe_counts_by_system: Dict[str, Dict[str, Dict[str, int]]] = {
        system: {name: {"rows": 0, "missing_rows": 0} for name in HOTPOT_ENTRY_BOUNDARY_NAMES}
        for system in systems
    }
    boundary_probe_missing_by_system: Dict[str, Dict[str, Counter[str]]] = {
        system: {name: Counter() for name in HOTPOT_ENTRY_BOUNDARY_NAMES} for system in systems
    }
    context_pull_summary_by_system: Dict[str, Dict[str, List[float]]] = {system: {} for system in systems}
    context_pull_counts_by_system: Dict[str, Dict[str, int]] = {
        system: {"rows": 0, "missing_rows": 0} for system in systems
    }
    context_pull_missing_by_system: Dict[str, Counter[str]] = {system: Counter() for system in systems}
    dependency_gold_summary_by_system: Dict[str, Dict[str, List[float]]] = {system: {} for system in systems}
    dependency_gold_counts_by_system: Dict[str, Dict[str, int]] = {
        system: {"rows": 0, "missing_rows": 0} for system in systems
    }
    dependency_gold_missing_by_system: Dict[str, Counter[str]] = {system: Counter() for system in systems}
    dependency_free_summary_by_system: Dict[str, Dict[str, List[float]]] = {system: {} for system in systems}
    dependency_free_counts_by_system: Dict[str, Dict[str, int]] = {
        system: {"rows": 0, "missing_rows": 0} for system in systems
    }
    dependency_free_missing_by_system: Dict[str, Counter[str]] = {system: Counter() for system in systems}
    generation_trace_counts_by_system: Dict[str, Dict[str, int]] = {
        system: {"rows": 0, "missing_rows": 0} for system in systems
    }
    generation_trace_missing_by_system: Dict[str, Counter[str]] = {system: Counter() for system in systems}
    steps_real: List[float] = []
    selected_examples, sample_id_filter_info = _resolve_selected_examples(
        bundle,
        requested_sample_ids=requested_sample_ids,
    )

    def _record_pressure_summary(system: str, probe: Optional[Dict[str, Any]]) -> None:
        if not probe:
            return
        summary = probe.get("summary")
        if not isinstance(summary, dict):
            return
        stats = pressure_summary_by_system[system]
        for key, raw_value in summary.items():
            if key == "steps":
                continue
            value = _safe_float(raw_value)
            if value is not None and not math.isnan(value):
                stats.setdefault(key, []).append(float(value))

    def _record_entry_transition_summary(system: str, probe: Optional[Dict[str, Any]]) -> None:
        counts = entry_probe_counts_by_system[system]
        counts["rows"] += 1
        if not isinstance(probe, dict):
            counts["missing_rows"] += 1
            entry_probe_missing_by_system[system]["unknown"] += 1
            return

        entry_payload = probe.get("entry")
        if not isinstance(entry_payload, dict):
            counts["missing_rows"] += 1
            entry_probe_missing_by_system[system]["missing_entry_payload"] += 1
            return

        missing_reason = str(entry_payload.get("missing_reason") or "").strip()
        if missing_reason:
            counts["missing_rows"] += 1
            entry_probe_missing_by_system[system][missing_reason] += 1
        else:
            stats = entry_probe_summary_by_system[system]
            for key in (
                "entry_entropy",
                "entry_top1_prob",
                "entry_top2_prob",
                "entry_top_gap",
                "realized_first_token_rank_at_entry",
                "realized_first_token_prob_at_entry",
            ):
                value = _safe_float(entry_payload.get(key))
                if value is not None and not math.isnan(value):
                    stats.setdefault(key, []).append(float(value))
            stats.setdefault("realized_first_token_top1_match", []).append(
                1.0 if bool(entry_payload.get("realized_first_token_top1_match")) else 0.0
            )

        if bundle.name != "hotpotqa_reasoning" or bundle.output_contract != "keyfacts":
            return

        boundaries = probe.get("boundaries")
        for boundary_name in HOTPOT_ENTRY_BOUNDARY_NAMES:
            boundary_counts = boundary_probe_counts_by_system[system][boundary_name]
            boundary_counts["rows"] += 1
            if not isinstance(boundaries, dict):
                boundary_counts["missing_rows"] += 1
                boundary_probe_missing_by_system[system][boundary_name]["missing_boundaries_payload"] += 1
                continue
            boundary_payload = boundaries.get(boundary_name)
            if not isinstance(boundary_payload, dict):
                boundary_counts["missing_rows"] += 1
                boundary_probe_missing_by_system[system][boundary_name]["missing_boundary_payload"] += 1
                continue

            boundary_missing_reason = str(boundary_payload.get("missing_reason") or "").strip()
            if boundary_missing_reason:
                boundary_counts["missing_rows"] += 1
                boundary_probe_missing_by_system[system][boundary_name][boundary_missing_reason] += 1
                continue

            summary = boundary_payload.get("summary")
            if not isinstance(summary, dict):
                boundary_counts["missing_rows"] += 1
                boundary_probe_missing_by_system[system][boundary_name]["missing_summary"] += 1
                continue

            stats = boundary_probe_summary_by_system[system][boundary_name]
            for key, raw_value in summary.items():
                value = _safe_float(raw_value)
                if value is not None and not math.isnan(value):
                    stats.setdefault(key, []).append(float(value))

    def _record_context_pull_summary(system: str, probe: Optional[Dict[str, Any]]) -> None:
        counts = context_pull_counts_by_system[system]
        counts["rows"] += 1
        if not isinstance(probe, dict):
            counts["missing_rows"] += 1
            context_pull_missing_by_system[system]["unknown"] += 1
            return

        summary = probe.get("summary")
        if not isinstance(summary, dict):
            counts["missing_rows"] += 1
            reason = str(probe.get("missing_reason") or "unknown").strip() or "unknown"
            context_pull_missing_by_system[system][reason] += 1
            return

        stats = context_pull_summary_by_system[system]
        for key, raw_value in summary.items():
            value = _safe_float(raw_value)
            if value is not None and not math.isnan(value):
                stats.setdefault(key, []).append(float(value))

    def _record_generation_trace_summary(system: str, trace_payload: Optional[Dict[str, Any]]) -> None:
        counts = generation_trace_counts_by_system[system]
        counts["rows"] += 1
        if not isinstance(trace_payload, dict):
            counts["missing_rows"] += 1
            generation_trace_missing_by_system[system]["unknown"] += 1
            return

        missing_reason = str(trace_payload.get("missing_reason") or "").strip()
        if missing_reason:
            counts["missing_rows"] += 1
            generation_trace_missing_by_system[system][missing_reason] += 1
            return

        if not isinstance(trace_payload.get("steps"), list):
            counts["missing_rows"] += 1
            generation_trace_missing_by_system[system]["missing_steps"] += 1
            return

    def _record_dependency_gold_summary(system: str, probe: Optional[Dict[str, Any]]) -> None:
        counts = dependency_gold_counts_by_system[system]
        counts["rows"] += 1
        if not isinstance(probe, dict):
            counts["missing_rows"] += 1
            dependency_gold_missing_by_system[system]["unknown"] += 1
            return

        masks = probe.get("masks")
        if not isinstance(masks, dict):
            counts["missing_rows"] += 1
            reason = str(probe.get("missing_reason") or "unknown").strip() or "unknown"
            dependency_gold_missing_by_system[system][reason] += 1
            return

        required_masks = ("headers_only", "key_fact_1", "key_fact_2", "both_key_facts")
        stats = dependency_gold_summary_by_system[system]
        for mask_name in required_masks:
            mask_payload = masks.get(mask_name)
            if not isinstance(mask_payload, dict):
                counts["missing_rows"] += 1
                dependency_gold_missing_by_system[system][f"missing_mask_{mask_name}"] += 1
                return
            summary = mask_payload.get("summary")
            if not isinstance(summary, dict):
                counts["missing_rows"] += 1
                reason = str(mask_payload.get("missing_reason") or f"missing_summary_{mask_name}")
                dependency_gold_missing_by_system[system][reason] += 1
                return
            for key, raw_value in summary.items():
                value = _safe_float(raw_value)
                if value is not None and not math.isnan(value):
                    stats.setdefault(f"{mask_name}:{key}", []).append(float(value))

    def _record_dependency_free_summary(system: str, probe: Optional[Dict[str, Any]]) -> None:
        counts = dependency_free_counts_by_system[system]
        counts["rows"] += 1
        if not isinstance(probe, dict):
            counts["missing_rows"] += 1
            dependency_free_missing_by_system[system]["unknown"] += 1
            return

        masks = probe.get("masks")
        if not isinstance(masks, dict):
            counts["missing_rows"] += 1
            reason = str(probe.get("missing_reason") or "unknown").strip() or "unknown"
            dependency_free_missing_by_system[system][reason] += 1
            return

        required_masks = ("headers_only", "key_fact_1", "key_fact_2", "both_key_facts")
        stats = dependency_free_summary_by_system[system]
        for mask_name in required_masks:
            mask_payload = masks.get(mask_name)
            if not isinstance(mask_payload, dict):
                counts["missing_rows"] += 1
                dependency_free_missing_by_system[system][f"missing_mask_{mask_name}"] += 1
                return
            summary = mask_payload.get("summary")
            if not isinstance(summary, dict):
                counts["missing_rows"] += 1
                reason = str(mask_payload.get("missing_reason") or f"missing_summary_{mask_name}")
                dependency_free_missing_by_system[system][reason] += 1
                return
            for key, raw_value in summary.items():
                value = _safe_float(raw_value)
                if value is not None and not math.isnan(value):
                    stats.setdefault(f"{mask_name}:{key}", []).append(float(value))

    for idx, example in selected_examples:
        batch = _build_single_batch(example, pad_id=pad_id)
        meta = dict((example.get("meta") or {}).items())
        prompt_input_ids, _prompt_attention_mask = _prompt_tensors_from_batch(batch)
        prompt_text = tokenizer.decode(prompt_input_ids[0].tolist(), skip_special_tokens=True)
        target_text = _target_text_from_batch(tokenizer, batch)

        row: Dict[str, Any] = {
            "task": bundle.name,
            "sample_index": idx,
            "sample_id": meta.get("sample_id"),
            "benchmark_role": bundle.benchmark_role,
            "reasoning_eval_mode": bundle.reasoning_eval_mode,
            "allow_abstain": bundle.allow_abstain,
            "abstain_mode": bundle.abstain_mode,
            "abstain_text": bundle.abstain_text,
            "abstain_option_letter": bundle.abstain_option_letter,
            "prompt": prompt_text,
            "target": target_text,
            "meta": meta,
            "predictions": {},
        }

        if want_base:
            base_out = _evaluate_base(
                tokenizer,
                backbone,
                batch,
                task_name=bundle.name,
                prompt_text=prompt_text,
                meta=meta,
                entry_transition_probe=entry_transition_probe,
                generation_trace=generation_trace,
                generation_trace_topk=generation_trace_topk,
                context_pull_probe=context_pull_probe,
                dependency_probe_gold=dependency_probe_gold,
                dependency_probe_free=dependency_probe_free,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
            )
            loss = base_out.get("loss")
            if loss is not None:
                loss_by_system["base"].append(float(loss))
            _update_reasoning_stats(bundle.name, stats_by_system["base"], meta, base_out["raw_output"])
            _record_pressure_summary("base", base_out.get("pressure_probe"))
            if entry_transition_probe:
                _record_entry_transition_summary("base", base_out.get("entry_transition_probe"))
            if generation_trace:
                _record_generation_trace_summary("base", base_out.get("generation_trace"))
            if context_pull_probe:
                _record_context_pull_summary("base", base_out.get("context_pull_probe"))
            if dependency_probe_gold and bundle.name == "hotpotqa_reasoning" and bundle.output_contract == "keyfacts":
                _record_dependency_gold_summary("base", base_out.get("dependency_probe_gold"))
            if dependency_probe_free and bundle.name == "hotpotqa_reasoning" and bundle.output_contract == "keyfacts":
                _record_dependency_free_summary("base", base_out.get("dependency_probe_free"))
            base_pred = {
                "raw_output": base_out["raw_output"],
                "loss": loss,
                "pressure_probe": base_out.get("pressure_probe"),
                **_score_prediction(bundle.name, meta, base_out["raw_output"]),
            }
            if entry_transition_probe:
                base_pred["entry_transition_probe"] = base_out.get("entry_transition_probe")
            if generation_trace:
                base_pred["generation_trace"] = base_out.get("generation_trace")
            if context_pull_probe:
                base_pred["context_pull_probe"] = base_out.get("context_pull_probe")
            if dependency_probe_gold and bundle.name == "hotpotqa_reasoning" and bundle.output_contract == "keyfacts":
                base_pred["dependency_probe_gold"] = base_out.get("dependency_probe_gold")
            if dependency_probe_free and bundle.name == "hotpotqa_reasoning" and bundle.output_contract == "keyfacts":
                base_pred["dependency_probe_free"] = base_out.get("dependency_probe_free")
            row["predictions"]["base"] = base_pred

        if want_static:
            static_out = _evaluate_static(
                tokenizer,
                backbone,
                batch,
                static_model,
                task_name=bundle.name,
                prompt_text=prompt_text,
                meta=meta,
                entry_transition_probe=entry_transition_probe,
                generation_trace=generation_trace,
                generation_trace_topk=generation_trace_topk,
                context_pull_probe=context_pull_probe,
                dependency_probe_gold=dependency_probe_gold,
                dependency_probe_free=dependency_probe_free,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
            )
            loss = static_out.get("loss")
            if loss is not None:
                loss_by_system["static"].append(float(loss))
            _update_reasoning_stats(bundle.name, stats_by_system["static"], meta, static_out["raw_output"])
            _record_pressure_summary("static", static_out.get("pressure_probe"))
            if entry_transition_probe:
                _record_entry_transition_summary("static", static_out.get("entry_transition_probe"))
            if generation_trace:
                _record_generation_trace_summary("static", static_out.get("generation_trace"))
            if context_pull_probe:
                _record_context_pull_summary("static", static_out.get("context_pull_probe"))
            if dependency_probe_gold and bundle.name == "hotpotqa_reasoning" and bundle.output_contract == "keyfacts":
                _record_dependency_gold_summary("static", static_out.get("dependency_probe_gold"))
            if dependency_probe_free and bundle.name == "hotpotqa_reasoning" and bundle.output_contract == "keyfacts":
                _record_dependency_free_summary("static", static_out.get("dependency_probe_free"))
            static_pred = {
                "raw_output": static_out["raw_output"],
                "loss": loss,
                "pressure_probe": static_out.get("pressure_probe"),
                **_score_prediction(bundle.name, meta, static_out["raw_output"]),
            }
            if entry_transition_probe:
                static_pred["entry_transition_probe"] = static_out.get("entry_transition_probe")
            if generation_trace:
                static_pred["generation_trace"] = static_out.get("generation_trace")
            if context_pull_probe:
                static_pred["context_pull_probe"] = static_out.get("context_pull_probe")
            if dependency_probe_gold and bundle.name == "hotpotqa_reasoning" and bundle.output_contract == "keyfacts":
                static_pred["dependency_probe_gold"] = static_out.get("dependency_probe_gold")
            if dependency_probe_free and bundle.name == "hotpotqa_reasoning" and bundle.output_contract == "keyfacts":
                static_pred["dependency_probe_free"] = static_out.get("dependency_probe_free")
            row["predictions"]["static"] = static_pred

        if want_real:
            real_out = _evaluate_real(
                tokenizer,
                backbone,
                batch,
                head,
                task_name=bundle.name,
                prompt_text=prompt_text,
                meta=meta,
                entry_transition_probe=entry_transition_probe,
                generation_trace=generation_trace,
                generation_trace_topk=generation_trace_topk,
                context_pull_probe=context_pull_probe,
                real_trace_probe=real_trace_probe,
                real_trace_probe_dump_tensors=real_trace_probe_dump_tensors,
                real_trace_probe_max_tensor_rows=real_trace_probe_max_tensor_rows,
                dependency_probe_gold=dependency_probe_gold,
                dependency_probe_free=dependency_probe_free,
                refine_policy=refine_policy,
                refine_max_steps=refine_max_steps,
                refine_energy_delta_tol=refine_energy_delta_tol,
                refine_patience=refine_patience,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                trace_energy=trace_energy,
            )
            loss = real_out.get("loss")
            if loss is not None:
                loss_by_system["real"].append(float(loss))
            if real_out.get("steps_used") is not None:
                steps_real.append(float(real_out["steps_used"]))
            _update_reasoning_stats(bundle.name, stats_by_system["real"], meta, real_out["raw_output"])
            _record_pressure_summary("real", real_out.get("pressure_probe"))
            if entry_transition_probe:
                _record_entry_transition_summary("real", real_out.get("entry_transition_probe"))
            if generation_trace:
                _record_generation_trace_summary("real", real_out.get("generation_trace"))
            if context_pull_probe:
                _record_context_pull_summary("real", real_out.get("context_pull_probe"))
            if dependency_probe_gold and bundle.name == "hotpotqa_reasoning" and bundle.output_contract == "keyfacts":
                _record_dependency_gold_summary("real", real_out.get("dependency_probe_gold"))
            if dependency_probe_free and bundle.name == "hotpotqa_reasoning" and bundle.output_contract == "keyfacts":
                _record_dependency_free_summary("real", real_out.get("dependency_probe_free"))
            real_pred = {
                "raw_output": real_out["raw_output"],
                "loss": loss,
                "steps_used": real_out.get("steps_used"),
                "pred_energy": real_out.get("pred_energy"),
                "pressure_probe": real_out.get("pressure_probe"),
                **_score_prediction(bundle.name, meta, real_out["raw_output"]),
            }
            if entry_transition_probe:
                real_pred["entry_transition_probe"] = real_out.get("entry_transition_probe")
            if generation_trace:
                real_pred["generation_trace"] = real_out.get("generation_trace")
            if context_pull_probe:
                real_pred["context_pull_probe"] = real_out.get("context_pull_probe")
            if real_trace_probe:
                real_pred["real_trace_probe"] = real_out.get("real_trace_probe")
            if dependency_probe_gold and bundle.name == "hotpotqa_reasoning" and bundle.output_contract == "keyfacts":
                real_pred["dependency_probe_gold"] = real_out.get("dependency_probe_gold")
            if dependency_probe_free and bundle.name == "hotpotqa_reasoning" and bundle.output_contract == "keyfacts":
                real_pred["dependency_probe_free"] = real_out.get("dependency_probe_free")
            row["predictions"]["real"] = real_pred

        rows.append(row)

    metrics: Dict[str, Any] = {}
    for system, stats in stats_by_system.items():
        metrics.update(_finalize_task_metrics(bundle.name, stats, system=system))

    for system, losses in loss_by_system.items():
        if losses:
            metrics[f"{bundle.name}_{system}"] = _mean(losses)

    for system, probe_stats in pressure_summary_by_system.items():
        for summary_key, values in probe_stats.items():
            if not values:
                continue
            metric_stem = "pressure_trace_steps" if summary_key == "trace_steps" else summary_key
            mean_value = _mean(values)
            metrics[f"{bundle.name}_{metric_stem}_{system}"] = mean_value
            if summary_key == "trace_steps":
                # Backward-compatible alias for older downstream consumers.
                metrics[f"{bundle.name}_pressure_steps_{system}"] = mean_value

    if entry_transition_probe:
        for system, probe_stats in entry_probe_summary_by_system.items():
            counts = entry_probe_counts_by_system[system]
            rows_count = int(counts["rows"])
            missing_rows = int(counts["missing_rows"])
            metrics[f"{bundle.name}_entry_transition_probe_rows_{system}"] = rows_count
            metrics[f"{bundle.name}_entry_transition_probe_missing_rows_{system}"] = missing_rows
            metrics[f"{bundle.name}_entry_transition_probe_coverage_rate_{system}"] = (
                float(rows_count - missing_rows) / float(rows_count) if rows_count > 0 else float("nan")
            )
            for reason, count in sorted(entry_probe_missing_by_system[system].items()):
                reason_key = re.sub(r"[^a-z0-9]+", "_", reason.lower()).strip("_") or "unknown"
                metrics[f"{bundle.name}_entry_transition_probe_missing_{reason_key}_{system}"] = int(count)
            for summary_key, values in probe_stats.items():
                if not values:
                    continue
                metrics[f"{bundle.name}_entry_transition_probe_{summary_key}_{system}"] = _mean(values)

        if bundle.name == "hotpotqa_reasoning" and bundle.output_contract == "keyfacts":
            for system, boundary_stats_by_name in boundary_probe_summary_by_system.items():
                for boundary_name in HOTPOT_ENTRY_BOUNDARY_NAMES:
                    counts = boundary_probe_counts_by_system[system][boundary_name]
                    rows_count = int(counts["rows"])
                    missing_rows = int(counts["missing_rows"])
                    metric_prefix = f"{bundle.name}_entry_transition_probe_{boundary_name}"
                    metrics[f"{metric_prefix}_rows_{system}"] = rows_count
                    metrics[f"{metric_prefix}_missing_rows_{system}"] = missing_rows
                    metrics[f"{metric_prefix}_coverage_rate_{system}"] = (
                        float(rows_count - missing_rows) / float(rows_count) if rows_count > 0 else float("nan")
                    )
                    for reason, count in sorted(boundary_probe_missing_by_system[system][boundary_name].items()):
                        reason_key = re.sub(r"[^a-z0-9]+", "_", reason.lower()).strip("_") or "unknown"
                        metrics[f"{metric_prefix}_missing_{reason_key}_{system}"] = int(count)
                    for summary_key, values in boundary_stats_by_name[boundary_name].items():
                        if not values:
                            continue
                        metrics[f"{metric_prefix}_{summary_key}_{system}"] = _mean(values)

    if generation_trace:
        for system, counts in generation_trace_counts_by_system.items():
            rows_count = int(counts["rows"])
            missing_rows = int(counts["missing_rows"])
            metrics[f"{bundle.name}_generation_trace_rows_{system}"] = rows_count
            metrics[f"{bundle.name}_generation_trace_missing_rows_{system}"] = missing_rows
            metrics[f"{bundle.name}_generation_trace_coverage_rate_{system}"] = (
                float(rows_count - missing_rows) / float(rows_count) if rows_count > 0 else float("nan")
            )
            for reason, count in sorted(generation_trace_missing_by_system[system].items()):
                reason_key = re.sub(r"[^a-z0-9]+", "_", reason.lower()).strip("_") or "unknown"
                metrics[f"{bundle.name}_generation_trace_missing_{reason_key}_{system}"] = int(count)

    if context_pull_probe:
        for system, probe_stats in context_pull_summary_by_system.items():
            counts = context_pull_counts_by_system[system]
            rows_count = int(counts["rows"])
            missing_rows = int(counts["missing_rows"])
            metrics[f"{bundle.name}_context_pull_probe_rows_{system}"] = rows_count
            metrics[f"{bundle.name}_context_pull_probe_missing_rows_{system}"] = missing_rows
            metrics[f"{bundle.name}_context_pull_probe_coverage_rate_{system}"] = (
                float(rows_count - missing_rows) / float(rows_count) if rows_count > 0 else float("nan")
            )
            for reason, count in sorted(context_pull_missing_by_system[system].items()):
                reason_key = re.sub(r"[^a-z0-9]+", "_", reason.lower()).strip("_") or "unknown"
                metrics[f"{bundle.name}_context_pull_probe_missing_{reason_key}_{system}"] = int(count)
            for summary_key, values in probe_stats.items():
                if not values:
                    continue
                metrics[f"{bundle.name}_{summary_key}_{system}"] = _mean(values)

    if dependency_probe_gold and bundle.name == "hotpotqa_reasoning" and bundle.output_contract == "keyfacts":
        for system, probe_stats in dependency_gold_summary_by_system.items():
            counts = dependency_gold_counts_by_system[system]
            rows_count = int(counts["rows"])
            missing_rows = int(counts["missing_rows"])
            metrics[f"{bundle.name}_dependency_probe_gold_rows_{system}"] = rows_count
            metrics[f"{bundle.name}_dependency_probe_gold_missing_rows_{system}"] = missing_rows
            metrics[f"{bundle.name}_dependency_probe_gold_coverage_rate_{system}"] = (
                float(rows_count - missing_rows) / float(rows_count) if rows_count > 0 else float("nan")
            )
            for reason, count in sorted(dependency_gold_missing_by_system[system].items()):
                reason_key = re.sub(r"[^a-z0-9]+", "_", reason.lower()).strip("_") or "unknown"
                metrics[f"{bundle.name}_dependency_probe_gold_missing_{reason_key}_{system}"] = int(count)
            for summary_key, values in probe_stats.items():
                if not values:
                    continue
                mask_name, metric_name = summary_key.split(":", 1)
                metrics[f"{bundle.name}_dependency_probe_gold_{mask_name}_{metric_name}_{system}"] = _mean(values)

    if dependency_probe_free and bundle.name == "hotpotqa_reasoning" and bundle.output_contract == "keyfacts":
        for system, probe_stats in dependency_free_summary_by_system.items():
            counts = dependency_free_counts_by_system[system]
            rows_count = int(counts["rows"])
            missing_rows = int(counts["missing_rows"])
            metrics[f"{bundle.name}_dependency_probe_free_rows_{system}"] = rows_count
            metrics[f"{bundle.name}_dependency_probe_free_missing_rows_{system}"] = missing_rows
            metrics[f"{bundle.name}_dependency_probe_free_coverage_rate_{system}"] = (
                float(rows_count - missing_rows) / float(rows_count) if rows_count > 0 else float("nan")
            )
            for reason, count in sorted(dependency_free_missing_by_system[system].items()):
                reason_key = re.sub(r"[^a-z0-9]+", "_", reason.lower()).strip("_") or "unknown"
                metrics[f"{bundle.name}_dependency_probe_free_missing_{reason_key}_{system}"] = int(count)
            for summary_key, values in probe_stats.items():
                if not values:
                    continue
                mask_name, metric_name = summary_key.split(":", 1)
                metrics[f"{bundle.name}_dependency_probe_free_{mask_name}_{metric_name}_{system}"] = _mean(values)

    if "base" in loss_by_system and "real" in loss_by_system and loss_by_system["base"] and loss_by_system["real"]:
        metrics[f"{bundle.name}_gain_real"] = metrics[f"{bundle.name}_base"] - metrics[f"{bundle.name}_real"]
    if "base" in loss_by_system and "static" in loss_by_system and loss_by_system["base"] and loss_by_system["static"]:
        metrics[f"{bundle.name}_gain_static"] = metrics[f"{bundle.name}_base"] - metrics[f"{bundle.name}_static"]
    if "static" in loss_by_system and "real" in loss_by_system and loss_by_system["static"] and loss_by_system["real"]:
        metrics[f"{bundle.name}_gain_conditional"] = metrics[f"{bundle.name}_static"] - metrics[f"{bundle.name}_real"]
    if steps_real:
        metrics[f"{bundle.name}_steps_used_real"] = _mean(steps_real)

    task_summary = _task_summary_from_metrics(bundle.name, metrics, len(rows), systems)
    task_summary.update(
        {
            "benchmark_role": bundle.benchmark_role,
            "reasoning_eval_mode": bundle.reasoning_eval_mode,
            "allow_abstain": bundle.allow_abstain,
            "abstain_mode": bundle.abstain_mode,
            "abstain_text": bundle.abstain_text,
            "abstain_option_letter": bundle.abstain_option_letter,
            "dataset_name": bundle.dataset_name,
            "dataset_config_name": bundle.dataset_config_name,
            "split": bundle.split,
            "max_length": bundle.max_length,
            "answer_loss_tokens": bundle.answer_loss_tokens,
            "output_contract": bundle.output_contract,
            "sample_id_filter": sample_id_filter_info,
        }
    )
    return task_summary, rows, sample_id_filter_info


def main() -> None:
    """CLI entrypoint for the dedicated reasoning artifact runner."""
    parser = argparse.ArgumentParser(description="Dedicated uncertainty-aware reasoning evaluation for REAL")
    parser.add_argument("--run_dir", type=str, default=None, help="Path to outputs/<run_id> for config/checkpoint inference")
    parser.add_argument("--head_ckpt", type=str, default=None, help="Path to head_stepXXXXX.pt (required for real/all unless discoverable from --run_dir)")
    parser.add_argument("--static_ckpt", type=str, default=None, help="Path to static_prefix.pt (optional)")
    parser.add_argument("--model_name", type=str, default=None, help="HF model name override")
    parser.add_argument("--mode", type=str, default="all", choices=["base", "static", "real", "all"])
    parser.add_argument("--tasks", type=str, nargs="+", default=list(DEFAULT_TASKS), choices=TASK_CHOICES)
    parser.add_argument("--refine_policy", type=str, default=None, choices=["forward", "accept_reject_v1"])
    parser.add_argument("--refine_max_steps", type=int, default=None)
    parser.add_argument("--refine_energy_delta_tol", type=float, default=None)
    parser.add_argument("--refine_patience", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None, help="Optional per-task cap overriding the config defaults")
    parser.add_argument(
        "--hotpotqa_reasoning_output_contract",
        type=str,
        default=None,
        choices=list(HOTPOT_REASONING_OUTPUT_CONTRACTS),
    )
    parser.add_argument(
        "--hotpotqa_reasoning_prompt_variant",
        type=str,
        default=None,
        choices=list(HOTPOT_REASONING_PROMPT_VARIANTS),
    )
    parser.add_argument(
        "--hotpotqa_reasoning_response_cue_variant",
        type=str,
        default=None,
        choices=list(HOTPOT_REASONING_RESPONSE_CUE_VARIANTS),
    )
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--min_new_tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace_energy", action="store_true", help="Include REAL predicted energy traces in the per-sample rows")
    parser.add_argument(
        "--generation_trace",
        action="store_true",
        help="Record a compact per-step generation trace from existing generate() scores without changing generation",
    )
    parser.add_argument(
        "--generation_trace_topk",
        type=int,
        default=GENERATION_TRACE_TOPK_DEFAULT,
        help="Compact top-k width for entry and per-step generation trace payloads",
    )
    parser.add_argument(
        "--entry_transition_probe",
        action="store_true",
        help="Record entry-step alignment plus Hotpot keyfacts boundary replay probes without changing generation",
    )
    parser.add_argument(
        "--context_pull_probe",
        action="store_true",
        help="Replay generated tokens with the source span occluded to measure residual prompt pull",
    )
    parser.add_argument(
        "--real_trace_probe",
        action="store_true",
        help="Attach a compact REAL refinement-trace summary to REAL rows without changing generation",
    )
    parser.add_argument(
        "--real_trace_probe_dump_tensors",
        action="store_true",
        help="Reserved debug flag; default behavior keeps tensor dumps disabled and row-safe",
    )
    parser.add_argument(
        "--real_trace_probe_max_tensor_rows",
        type=int,
        default=0,
        help="Reserved debug cap for optional tensor dumps; default 0 keeps the trace payload compact",
    )
    parser.add_argument(
        "--dependency_probe",
        action="store_true",
        help="Replay Hotpot keyfacts gold targets with scaffold spans masked to measure answer-region dependency",
    )
    parser.add_argument(
        "--dependency_probe_free",
        action="store_true",
        help="Replay generated Hotpot keyfacts continuations with emitted scaffold spans masked to measure answer-region dependency",
    )
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory (default: <run_dir>/reasoning_eval/)")
    parser.add_argument("--out_json", type=str, default=None, help="Optional summary JSON path")
    parser.add_argument("--out_jsonl", type=str, default=None, help="Optional combined per-sample JSONL path")
    parser.add_argument(
        "--sample_ids_path",
        type=str,
        default=None,
        help="Optional file containing exact sample_ids to replay (plain text, JSON list, or JSONL with sample_id field)",
    )
    args = parser.parse_args()

    _ensure_runtime()
    run_dir = Path(args.run_dir) if args.run_dir else None
    cfg_dict: Dict[str, Any] = {}
    if run_dir is not None:
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config.json under {run_dir}")
        cfg_dict = _load_json(cfg_path)

    cfg = real.TrainingConfig(**cfg_dict) if cfg_dict else real.TrainingConfig()
    run_id = str(cfg_dict.get("run_id") or (run_dir.name if run_dir is not None else "ad_hoc_reasoning_eval"))

    head_ckpt = _resolve_head_ckpt(run_dir, args.head_ckpt)
    static_ckpt = _resolve_static_ckpt(run_dir, cfg_dict, args.static_ckpt)

    refine_policy = (args.refine_policy or cfg.inference_refine_policy or "accept_reject_v1").strip().lower()
    refine_max_steps = args.refine_max_steps if args.refine_max_steps is not None else cfg.inference_refine_max_steps
    refine_energy_delta_tol = (
        float(args.refine_energy_delta_tol)
        if args.refine_energy_delta_tol is not None
        else float(cfg.inference_refine_energy_delta_tol)
    )
    refine_patience = int(args.refine_patience) if args.refine_patience is not None else int(cfg.inference_refine_patience)

    if args.mode in {"real", "all"} and head_ckpt is None:
        raise SystemExit("REAL mode requires --head_ckpt or a discoverable checkpoint under --run_dir")

    head_cfg: Dict[str, Any] = {}
    if head_ckpt is not None and head_ckpt.exists():
        head_cfg = infer._read_head_ckpt_config(head_ckpt)
    model_name = args.model_name or head_cfg.get("model_name") or cfg_dict.get("model_name") or infer.DEFAULT_MODEL_NAME

    out_dir = _resolve_output_dir(args, run_dir, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.out_json) if args.out_json else out_dir / "summary.json"
    rows_path = Path(args.out_jsonl) if args.out_jsonl else out_dir / "rows.jsonl"
    sample_ids_path = Path(args.sample_ids_path).resolve() if args.sample_ids_path else None
    requested_sample_ids = _load_sample_ids(sample_ids_path) if sample_ids_path is not None else None

    real.set_seed(args.seed)
    tokenizer, backbone = infer.load_backbone_and_tokenizer(model_name)

    head = None
    head_step = None
    if head_ckpt is not None:
        head, _head_cfg_loaded, head_step = infer.load_head(backbone, head_ckpt)

    static_model = None
    if static_ckpt is not None and static_ckpt.exists():
        _static_prefix, static_model = infer.load_static_prefix(backbone, static_ckpt)
    elif args.mode == "static":
        raise SystemExit("Static mode requires --static_ckpt or a discoverable static prefix checkpoint")
    elif args.mode == "all" and static_ckpt is None:
        print("[reasoning_eval] static_ckpt not provided; static mode skipped")

    tasks = list(dict.fromkeys(args.tasks))
    summary: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": run_id,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "run_dir_rel": portable_path_str(run_dir) if run_dir is not None else None,
        "benchmark_role": None,
        "reasoning_eval_mode": True,
        "allow_abstain": False,
        "abstain_modes": [],
        "primary_metric_family": "reasoning_decision",
        "primary_metric_name": None,
        "primary_metric_value": None,
        "model_name": model_name,
        "head_ckpt": str(head_ckpt) if head_ckpt is not None else None,
        "head_ckpt_rel": portable_path_str(head_ckpt) if head_ckpt is not None else None,
        "head_ckpt_name": basename_or_relpath(head_ckpt) if head_ckpt is not None else None,
        "head_step": head_step,
        "static_ckpt": str(static_ckpt) if static_ckpt is not None else None,
        "static_ckpt_rel": portable_path_str(static_ckpt) if static_ckpt is not None else None,
        "static_ckpt_name": basename_or_relpath(static_ckpt) if static_ckpt is not None else None,
        "mode": args.mode,
        "tasks": {},
        "refine_policy": refine_policy,
        "refine_max_steps": refine_max_steps,
        "refine_energy_delta_tol": refine_energy_delta_tol,
        "refine_patience": refine_patience,
        "entry_transition_probe": bool(args.entry_transition_probe),
        "generation_trace": bool(args.generation_trace),
        "generation_trace_topk": int(args.generation_trace_topk),
        "context_pull_probe": bool(args.context_pull_probe),
        "real_trace_probe": bool(args.real_trace_probe),
        "real_trace_probe_schema_version": 1 if args.real_trace_probe else None,
        "real_trace_probe_dump_tensors": bool(args.real_trace_probe_dump_tensors),
        "real_trace_probe_max_tensor_rows": int(args.real_trace_probe_max_tensor_rows),
        "real_trace_probe_behavior_change": False,
        "dependency_probe": bool(args.dependency_probe),
        "dependency_probe_free": bool(args.dependency_probe_free),
        "sample_ids_path": str(sample_ids_path) if sample_ids_path is not None else None,
        "sample_ids_path_rel": portable_path_str(sample_ids_path) if sample_ids_path is not None else None,
        "sample_id_filter_requested_count": int(len(requested_sample_ids or [])),
        "hotpotqa_reasoning_prompt_variant": args.hotpotqa_reasoning_prompt_variant,
        "hotpotqa_reasoning_response_cue_variant": args.hotpotqa_reasoning_response_cue_variant,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "seed": args.seed,
    }

    all_rows: List[Dict[str, Any]] = []
    primary_values: List[float] = []
    t0 = time.time()
    for task_name in tasks:
        bundle = _build_task_bundle(
            task_name,
            tokenizer,
            cfg,
            args.max_samples,
            hotpotqa_output_contract_override=args.hotpotqa_reasoning_output_contract if task_name == "hotpotqa_reasoning" else None,
            hotpotqa_prompt_variant_override=args.hotpotqa_reasoning_prompt_variant if task_name == "hotpotqa_reasoning" else None,
            hotpotqa_response_cue_variant_override=(
                args.hotpotqa_reasoning_response_cue_variant if task_name == "hotpotqa_reasoning" else None
            ),
        )
        task_summary, task_rows, task_sample_filter_info = evaluate_task(
            bundle,
            tokenizer=tokenizer,
            backbone=backbone,
            head=head,
            static_model=static_model,
            mode=args.mode,
            refine_policy=refine_policy,
            refine_max_steps=refine_max_steps,
            refine_energy_delta_tol=refine_energy_delta_tol,
            refine_patience=refine_patience,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            trace_energy=args.trace_energy,
            entry_transition_probe=bool(args.entry_transition_probe),
            generation_trace=bool(args.generation_trace),
            generation_trace_topk=int(args.generation_trace_topk),
            context_pull_probe=bool(args.context_pull_probe),
            real_trace_probe=bool(args.real_trace_probe),
            real_trace_probe_dump_tensors=bool(args.real_trace_probe_dump_tensors),
            real_trace_probe_max_tensor_rows=int(args.real_trace_probe_max_tensor_rows),
            dependency_probe_gold=bool(args.dependency_probe),
            dependency_probe_free=bool(args.dependency_probe_free),
            requested_sample_ids=requested_sample_ids,
        )

        protocol = _build_protocol(
            bundle,
            mode=args.mode,
            batch_size=1,
            sample_count=len(task_rows),
            refine_policy=refine_policy,
            refine_max_steps=refine_max_steps,
            refine_energy_delta_tol=refine_energy_delta_tol,
            refine_patience=refine_patience,
            seed=args.seed,
            primary_metric_name=str(task_summary.get("primary_metric_name") or bundle.primary_metric_name),
            sample_ids_path=portable_path_str(sample_ids_path) if sample_ids_path is not None else None,
            sample_id_filter_info=task_sample_filter_info,
            entry_transition_probe=bool(args.entry_transition_probe),
            context_pull_probe=bool(args.context_pull_probe),
            real_trace_probe=bool(args.real_trace_probe),
            real_trace_probe_dump_tensors=bool(args.real_trace_probe_dump_tensors),
        )
        metrics_path = out_dir / f"{task_name}_metrics.json"
        protocol_path = out_dir / f"{task_name}_protocol.json"
        task_rows_path = out_dir / f"{task_name}_rows.jsonl"

        task_summary["artifacts"] = {
            "metrics_json": _relpath_str(metrics_path, out_dir),
            "protocol_json": _relpath_str(protocol_path, out_dir),
            "rows_jsonl": _relpath_str(task_rows_path, out_dir),
        }
        if isinstance(task_summary.get("sample_id_filter"), dict) and task_summary["sample_id_filter"].get("enabled"):
            task_summary["sample_id_filter"]["path"] = str(sample_ids_path) if sample_ids_path is not None else None
            task_summary["sample_id_filter"]["path_rel"] = (
                portable_path_str(sample_ids_path) if sample_ids_path is not None else None
            )
        if bundle.name == "hotpotqa_reasoning":
            task_summary["output_contract"] = bundle.output_contract
            task_summary["prompt_variant"] = bundle.prompt_variant
            task_summary["response_cue_variant"] = bundle.response_cue_variant

        _write_json(metrics_path, task_summary)
        protocol.write_json(protocol_path)
        _write_jsonl(task_rows_path, task_rows)

        primary_value = _safe_float(task_summary.get("primary_metric_value"))
        if primary_value is not None and not math.isnan(primary_value):
            primary_values.append(primary_value)

        summary["tasks"][task_name] = task_summary
        all_rows.extend(task_rows)

        print(
            f"[reasoning_eval] task={task_name} n={len(task_rows)} "
            f"primary={task_summary.get('primary_metric_name')}={_fmt_metric(primary_value)}"
        )

    if primary_values:
        summary["primary_metric_name"] = "reasoning_mean_primary_selected_mode"
        summary["primary_metric_value"] = _mean(primary_values)

    task_summaries = list(summary["tasks"].values())
    if task_summaries:
        benchmark_roles = sorted(
            {
                str(task.get("benchmark_role"))
                for task in task_summaries
                if str(task.get("benchmark_role") or "").strip()
            }
        )
        if len(benchmark_roles) == 1:
            summary["benchmark_role"] = benchmark_roles[0]
        elif benchmark_roles:
            summary["benchmark_role"] = "mixed"
            summary["benchmark_roles"] = benchmark_roles
        summary["allow_abstain"] = any(bool(task.get("allow_abstain")) for task in task_summaries)
        summary["abstain_modes"] = sorted(
            {
                str(task.get("abstain_mode"))
                for task in task_summaries
                if str(task.get("abstain_mode") or "").strip()
            }
        )

    summary["row_count"] = len(all_rows)
    summary["runtime_s"] = round(time.time() - t0, 3)
    _write_json(summary_path, summary)
    _write_jsonl(rows_path, all_rows)

    print(f"[reasoning_eval] wrote {summary_path}")
    print(f"[reasoning_eval] wrote {rows_path}")


if __name__ == "__main__":
    main()
