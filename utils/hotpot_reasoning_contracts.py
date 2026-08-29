"""Shared HotpotQA reasoning-contract helpers.

These helpers are intentionally eval-focused and lightweight. They centralize
contract normalization, prompt/target rendering, and line-based field parsing
so training and reasoning-eval stay in sync as new output contracts land.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


HOTPOT_REASONING_OUTPUT_CONTRACTS: Tuple[str, ...] = ("default", "stepwise", "keyfacts", "answer_only")
HOTPOT_REASONING_PROMPT_VARIANTS: Tuple[str, ...] = ("default", "exact")
HOTPOT_REASONING_RESPONSE_CUE_VARIANTS: Tuple[str, ...] = ("default", "key_fact_1")

_STEP_1_RE = re.compile(r"^\s*(Step\s*1\s*:)(.*)$", re.IGNORECASE)
_STEP_2_RE = re.compile(r"^\s*(Step\s*2\s*:)(.*)$", re.IGNORECASE)
_SUPPORTING_FACTS_RE = re.compile(r"^\s*(Supporting facts\s*:)\s*$", re.IGNORECASE)
_FINAL_ANSWER_RE = re.compile(r"^\s*(Final answer\s*:)(.*)$", re.IGNORECASE)
_KEY_FACT_1_RE = re.compile(r"^\s*(Key fact\s*1\s*:)(.*)$", re.IGNORECASE)
_KEY_FACT_2_RE = re.compile(r"^\s*(Key fact\s*2\s*:)(.*)$", re.IGNORECASE)
_ANSWER_RE = re.compile(r"^\s*(Answer\s*:)(.*)$", re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*[-*]\s*(.+?)\s*$")
_ANSWER_FIELD_MARKER_RE = re.compile(
    r"(?:key fact\s*[12]\s*:|supporting facts\s*:|final answer\s*:|answer\s*:)",
    re.IGNORECASE,
)
_NUMBERED_KEYFACTS_RE = re.compile(
    r"^\s*1\.\s*(?P<fact1>.+?)\s+2\.\s*(?P<fact2>.+?)"
    r"(?:\s+(?P<answer_header>(?:Answer|Final answer)\s*:)\s*(?P<answer>.+))?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_NORMALIZE_TEXT_RE = re.compile(r"[^a-z0-9]+")


def normalize_hotpot_reasoning_output_contract(output_contract: Optional[str]) -> str:
    """Canonicalize the eval-only Hotpot reasoning output contract name."""
    contract = str(output_contract or "default").strip().lower().replace("-", "_")
    if contract not in HOTPOT_REASONING_OUTPUT_CONTRACTS:
        raise ValueError(f"Unsupported Hotpot reasoning output contract: {output_contract!r}")
    return contract


def normalize_hotpot_reasoning_prompt_variant(prompt_variant: Optional[str]) -> str:
    """Canonicalize the eval-only Hotpot prompt-variant name."""
    variant = str(prompt_variant or "default").strip().lower().replace("-", "_")
    if variant not in HOTPOT_REASONING_PROMPT_VARIANTS:
        raise ValueError(f"Unsupported Hotpot reasoning prompt variant: {prompt_variant!r}")
    return variant


def normalize_hotpot_reasoning_response_cue_variant(response_cue_variant: Optional[str]) -> str:
    """Canonicalize the eval-only Hotpot response-cue variant name."""
    variant = str(response_cue_variant or "default").strip().lower().replace("-", "_")
    if variant not in HOTPOT_REASONING_RESPONSE_CUE_VARIANTS:
        raise ValueError(f"Unsupported Hotpot reasoning response cue variant: {response_cue_variant!r}")
    return variant


def hotpot_reasoning_response_cue(
    output_contract: str,
    *,
    response_cue_variant: Optional[str] = None,
) -> str:
    """Render the generation-start cue appended after the question."""
    contract = normalize_hotpot_reasoning_output_contract(output_contract)
    variant = normalize_hotpot_reasoning_response_cue_variant(response_cue_variant)
    if variant == "key_fact_1":
        if contract != "keyfacts":
            raise ValueError("`key_fact_1` response cue is only supported for the `keyfacts` contract")
        return "Key fact 1: "
    return "Answer: "


def select_hotpot_key_facts(gold_supporting_facts: Sequence[str]) -> Dict[str, Any]:
    """Choose the two scaffold facts used by the `keyfacts` contract."""
    facts = [str(fact).strip() for fact in gold_supporting_facts if str(fact).strip()]
    if not facts:
        return {
            "gold_key_fact_1": "",
            "gold_key_fact_2": "",
            "gold_key_facts_selected": [],
            "gold_key_fact_2_is_copy": False,
        }

    fact_1 = facts[0]
    if len(facts) > 1:
        fact_2 = facts[1]
        fact_2_is_copy = False
    else:
        fact_2 = fact_1
        fact_2_is_copy = True
    return {
        "gold_key_fact_1": fact_1,
        "gold_key_fact_2": fact_2,
        "gold_key_facts_selected": [fact_1, fact_2],
        "gold_key_fact_2_is_copy": fact_2_is_copy,
    }


def hotpot_reasoning_prompt_prefix(output_contract: str, *, prompt_variant: Optional[str] = None) -> str:
    """Render the Hotpot prompt instructions for one output contract."""
    contract = normalize_hotpot_reasoning_output_contract(output_contract)
    variant = normalize_hotpot_reasoning_prompt_variant(prompt_variant)
    if contract == "stepwise":
        return (
            "Answer the multi-hop question using only the provided evidence.\n"
            "Return exactly in this format:\n"
            "Step 1: <short reasoning step 1>\n"
            "Step 2: <short reasoning step 2>\n"
            "Supporting facts:\n"
            "- <short fact 1>\n"
            "- <short fact 2>\n"
            "Final answer: <answer>\n"
            "Evidence:\n"
        )
    if contract == "keyfacts":
        if variant == "exact":
            return (
                "Answer the multi-hop question using only the provided evidence.\n"
                "Return exactly in this format:\n"
                "Key fact 1: <short fact 1>\n"
                "Key fact 2: <short fact 2>\n"
                "Answer: <answer>\n"
                "Evidence:\n"
            )
        return (
            "Answer the multi-hop question using only the provided evidence.\n"
            "A concise structured response can help. One acceptable format is:\n"
            "Key fact 1: <short fact 1>\n"
            "Key fact 2: <short fact 2>\n"
            "Answer: <answer>\n"
            "Evidence:\n"
        )
    if contract == "answer_only":
        return (
            "Answer the multi-hop question using only the provided evidence.\n"
            "Return exactly in this format:\n"
            "Answer: <answer>\n"
            "Evidence:\n"
        )
    return (
        "Answer the multi-hop question using only the provided evidence.\n"
        "Return exactly in this format:\n"
        "Supporting facts:\n"
        "- <short fact 1>\n"
        "- <short fact 2>\n"
        "Final answer: <answer>\n"
        "Evidence:\n"
    )


def hotpot_reasoning_target_lines(
    gold_supporting_facts: Sequence[str],
    answer: str,
    *,
    output_contract: str,
) -> List[str]:
    """Render the supervised target for one Hotpot reasoning output contract."""
    contract = normalize_hotpot_reasoning_output_contract(output_contract)
    facts = [str(fact).strip() for fact in gold_supporting_facts if str(fact).strip()]
    answer_text = str(answer).strip()

    if contract == "stepwise":
        step_1 = facts[0] if facts else answer_text
        step_2 = facts[1] if len(facts) > 1 else step_1
        return [
            f"Step 1: {step_1}",
            f"Step 2: {step_2}",
            "Supporting facts:",
            *(f"- {fact}" for fact in facts),
            f"Final answer: {answer_text}",
        ]

    if contract == "keyfacts":
        selected = select_hotpot_key_facts(facts)
        return [
            f"Key fact 1: {selected['gold_key_fact_1']}",
            f"Key fact 2: {selected['gold_key_fact_2']}",
            f"Answer: {answer_text}",
        ]

    if contract == "answer_only":
        return [f"Answer: {answer_text}"]

    return [
        "Supporting facts:",
        *(f"- {fact}" for fact in facts),
        f"Final answer: {answer_text}",
    ]


def hotpot_reasoning_target_text(
    gold_supporting_facts: Sequence[str],
    answer: str,
    *,
    output_contract: str,
) -> str:
    """Render the target as a single newline-joined string."""
    return "\n".join(
        hotpot_reasoning_target_lines(
            gold_supporting_facts,
            answer,
            output_contract=output_contract,
        )
    )


def hotpot_reasoning_target_continuation_text(
    gold_supporting_facts: Sequence[str],
    answer: str,
    *,
    output_contract: str,
    response_cue_variant: Optional[str] = None,
) -> str:
    """Render only the continuation portion after any prompt-side response cue."""
    full_target = hotpot_reasoning_target_text(
        gold_supporting_facts,
        answer,
        output_contract=output_contract,
    )
    cue = hotpot_reasoning_response_cue(
        output_contract,
        response_cue_variant=response_cue_variant,
    )
    if full_target.startswith(cue):
        return full_target[len(cue) :]
    return full_target


def hotpot_reasoning_effective_generated_text(
    raw_text: str,
    *,
    output_contract: str,
    response_cue_variant: Optional[str] = None,
    response_cue_text: Optional[str] = None,
) -> Tuple[str, int]:
    """Return the effective visible generated text including any prompt-side cue."""
    text = str(raw_text or "")
    contract = normalize_hotpot_reasoning_output_contract(output_contract)
    variant = normalize_hotpot_reasoning_response_cue_variant(response_cue_variant)
    cue_text = (
        str(response_cue_text)
        if isinstance(response_cue_text, str) and response_cue_text
        else hotpot_reasoning_response_cue(contract, response_cue_variant=variant)
    )
    if contract == "keyfacts" and variant == "key_fact_1" and cue_text:
        return cue_text + text, len(cue_text)
    if contract == "answer_only" and variant == "default" and cue_text:
        return cue_text + text, len(cue_text)
    return text, 0


def rebase_hotpot_char_spans_from_effective_text(char_spans: Dict[str, Any], prefix_chars: int) -> Dict[str, Any]:
    """Shift effective-text char spans back into raw-generation coordinates."""
    if prefix_chars <= 0:
        return dict(char_spans or {})

    def _rebase_one(char_span: Any) -> Optional[List[int]]:
        if not isinstance(char_span, (list, tuple)) or len(char_span) != 2:
            return None
        try:
            start = int(char_span[0]) - int(prefix_chars)
            end = int(char_span[1]) - int(prefix_chars)
        except (TypeError, ValueError):
            return None
        if end <= 0:
            return None
        if start < 0:
            start = 0
        if end <= start:
            return None
        return [int(start), int(end)]

    out: Dict[str, Any] = {}
    for key, value in dict(char_spans or {}).items():
        if isinstance(value, list) and value and all(isinstance(item, (list, tuple)) and len(item) == 2 for item in value):
            out[key] = [rebased for item in value if (rebased := _rebase_one(item)) is not None]
            continue
        rebased = _rebase_one(value)
        if rebased is not None:
            out[key] = rebased
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            out[key] = None
        else:
            out[key] = value
    return out


def hotpot_contract_field_names(output_contract: str) -> Tuple[str, ...]:
    """Return the canonical field names for one contract."""
    contract = normalize_hotpot_reasoning_output_contract(output_contract)
    if contract == "stepwise":
        return ("step_1", "step_2", "supporting_facts", "final_answer")
    if contract == "keyfacts":
        return ("key_fact_1", "key_fact_2", "answer")
    if contract == "answer_only":
        return ("answer",)
    return ("supporting_facts", "final_answer")


def hotpot_contract_descriptor(output_contract: str) -> Dict[str, Any]:
    """Return contract-local field order and close-boundary metadata."""
    contract = normalize_hotpot_reasoning_output_contract(output_contract)
    if contract == "keyfacts":
        return {
            "output_contract": contract,
            "ordered_fields": ("key_fact_1", "key_fact_2", "answer"),
            "close_field": "answer",
            "preclose_fields": ("key_fact_1", "key_fact_2"),
            "header_texts": {
                "key_fact_1": "Key fact 1:",
                "key_fact_2": "Key fact 2:",
                "answer": "Answer:",
            },
            "late_boundary_label": "key_fact_2_to_answer",
            "close_field_mode": "line",
        }
    if contract == "stepwise":
        return {
            "output_contract": contract,
            "ordered_fields": ("step_1", "step_2", "supporting_facts", "final_answer"),
            "close_field": "final_answer",
            "preclose_fields": ("step_1", "step_2", "supporting_facts"),
            "header_texts": {
                "step_1": "Step 1:",
                "step_2": "Step 2:",
                "supporting_facts": "Supporting facts:",
                "final_answer": "Final answer:",
            },
            "late_boundary_label": "supporting_facts_to_final_answer",
            "close_field_mode": "line",
        }
    if contract == "answer_only":
        return {
            "output_contract": contract,
            "ordered_fields": ("answer",),
            "close_field": "answer",
            "preclose_fields": (),
            "header_texts": {
                "answer": "Answer:",
            },
            "late_boundary_label": "answer_open_to_answer",
            "close_field_mode": "line",
        }
    return {
        "output_contract": contract,
        "ordered_fields": ("supporting_facts", "final_answer"),
        "close_field": "final_answer",
        "preclose_fields": ("supporting_facts",),
        "header_texts": {
            "supporting_facts": "Supporting facts:",
            "final_answer": "Final answer:",
        },
        "late_boundary_label": "supporting_facts_to_final_answer",
        "close_field_mode": "line",
    }


def hotpot_contract_ordered_fields(output_contract: str) -> Tuple[str, ...]:
    """Return descriptor-backed ordered field names for one contract."""
    return tuple(hotpot_contract_descriptor(output_contract)["ordered_fields"])


def _trim_span(text: str, start: int, end: int) -> Optional[Tuple[int, int]]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end <= start:
        return None
    return start, end


def _normalize_free_text(text: str) -> str:
    return " ".join(_NORMALIZE_TEXT_RE.sub(" ", str(text or "").lower()).split())


def _distinct_nonempty_text(text_a: str, text_b: str) -> bool:
    norm_a = _normalize_free_text(text_a)
    norm_b = _normalize_free_text(text_b)
    return bool(norm_a) and bool(norm_b) and norm_a != norm_b


def _is_fact_like_text(text: str) -> bool:
    return len(_normalize_free_text(text).split()) >= 3


def _token_count(text: str) -> int:
    return len(_normalize_free_text(text).split())


def _line_field_entry(
    text: str,
    line_start: int,
    match: re.Match[str],
    *,
    header_group: int = 1,
    value_group: int = 2,
) -> Dict[str, Any]:
    header_span = [
        int(line_start + match.start(header_group)),
        int(line_start + match.end(header_group)),
    ]
    value_abs_start = int(line_start + match.start(value_group))
    value_abs_end = int(line_start + match.end(value_group))
    trimmed = _trim_span(text, value_abs_start, value_abs_end)
    value_span = [int(trimmed[0]), int(trimmed[1])] if trimmed is not None else None
    value_text = text[trimmed[0] : trimmed[1]] if trimmed is not None else ""
    return {
        "present": True,
        "header_span": header_span,
        "value_span": value_span,
        "text": value_text,
        "nonempty": bool(value_text),
    }


def _empty_field() -> Dict[str, Any]:
    return {
        "present": False,
        "header_span": None,
        "value_span": None,
        "text": "",
        "nonempty": False,
    }


def parse_hotpot_keyfacts_weak_structure(raw_text: str) -> Dict[str, Any]:
    """Parse conservative numbered key-fact structure from free-form text."""
    text = str(raw_text or "")
    match = _NUMBERED_KEYFACTS_RE.match(text)
    if match is None:
        return {
            "has_numbered_fact_1": False,
            "has_numbered_fact_2": False,
            "numbered_key_fact_1": "",
            "numbered_key_fact_2": "",
            "numbered_answer": "",
            "numbered_fact_1_nonempty": False,
            "numbered_fact_2_nonempty": False,
            "numbered_answer_nonempty": False,
            "numbered_key_facts_distinct": False,
            "numbered_fact_scaffold_present": False,
            "numbered_answer_header_present": False,
            "char_spans": {
                "numbered_key_fact_1_value_span": None,
                "numbered_key_fact_2_value_span": None,
                "numbered_answer_header_span": None,
                "numbered_answer_value_span": None,
            },
        }

    fact_1_span = _trim_span(text, match.start("fact1"), match.end("fact1"))
    fact_2_span = _trim_span(text, match.start("fact2"), match.end("fact2"))
    answer_group = match.group("answer")
    answer_span = (
        _trim_span(text, match.start("answer"), match.end("answer"))
        if answer_group is not None
        else None
    )
    answer_header_span = (
        [int(match.start("answer_header")), int(match.end("answer_header"))]
        if match.group("answer_header") is not None
        else None
    )

    fact_1_text = text[fact_1_span[0] : fact_1_span[1]] if fact_1_span is not None else ""
    fact_2_text = text[fact_2_span[0] : fact_2_span[1]] if fact_2_span is not None else ""
    answer_text = text[answer_span[0] : answer_span[1]] if answer_span is not None else ""

    fact_pair_nonempty = bool(fact_1_text) and bool(fact_2_text)
    fact_pair_distinct = _distinct_nonempty_text(fact_1_text, fact_2_text)
    fact_pair_fact_like = _is_fact_like_text(fact_1_text) and _is_fact_like_text(fact_2_text)
    numbered_fact_scaffold_present = bool(fact_pair_nonempty and fact_pair_distinct and fact_pair_fact_like)

    return {
        "has_numbered_fact_1": bool(fact_1_text),
        "has_numbered_fact_2": bool(fact_2_text),
        "numbered_key_fact_1": fact_1_text,
        "numbered_key_fact_2": fact_2_text,
        "numbered_answer": answer_text,
        "numbered_fact_1_nonempty": bool(fact_1_text),
        "numbered_fact_2_nonempty": bool(fact_2_text),
        "numbered_answer_nonempty": bool(answer_text),
        "numbered_key_facts_distinct": fact_pair_distinct,
        "numbered_fact_scaffold_present": numbered_fact_scaffold_present,
        "numbered_answer_header_present": answer_header_span is not None,
        "char_spans": {
            "numbered_key_fact_1_value_span": [int(fact_1_span[0]), int(fact_1_span[1])] if fact_1_span is not None else None,
            "numbered_key_fact_2_value_span": [int(fact_2_span[0]), int(fact_2_span[1])] if fact_2_span is not None else None,
            "numbered_answer_header_span": answer_header_span,
            "numbered_answer_value_span": [int(answer_span[0]), int(answer_span[1])] if answer_span is not None else None,
        },
    }


def classify_hotpot_keyfacts_structure(
    raw_text: str,
    *,
    parsed: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Classify conservative weak-vs-strict keyfacts structure emergence."""
    parsed_obj = parsed or parse_hotpot_reasoning_fields(raw_text, output_contract="keyfacts")
    fields = dict(parsed_obj.get("fields") or {})
    weak = parse_hotpot_keyfacts_weak_structure(raw_text)

    labeled_key_fact_1 = str(((fields.get("key_fact_1") or {}).get("text")) or "").strip()
    labeled_key_fact_2 = str(((fields.get("key_fact_2") or {}).get("text")) or "").strip()
    labeled_answer = str(((fields.get("answer") or {}).get("text")) or ((fields.get("final_answer") or {}).get("text")) or "").strip()

    labeled_fact_scaffold_present = bool(
        ((fields.get("key_fact_1") or {}).get("nonempty"))
        and ((fields.get("key_fact_2") or {}).get("nonempty"))
        and _distinct_nonempty_text(labeled_key_fact_1, labeled_key_fact_2)
    )
    numbered_fact_scaffold_present = bool(weak.get("numbered_fact_scaffold_present"))
    answer_field_present = bool(labeled_answer or weak.get("numbered_answer_nonempty"))
    full_keyfacts_contract_present = bool(labeled_fact_scaffold_present and ((fields.get("answer") or {}).get("nonempty")))
    fact_scaffold_present = bool(labeled_fact_scaffold_present or numbered_fact_scaffold_present)

    if full_keyfacts_contract_present:
        state = "full_keyfacts_contract"
    elif labeled_fact_scaffold_present and not answer_field_present:
        state = "facts_only_labeled"
    elif numbered_fact_scaffold_present and not answer_field_present:
        state = "facts_only_numbered"
    elif fact_scaffold_present and answer_field_present:
        state = "facts_plus_answer"
    elif answer_field_present:
        state = "answer_only"
    else:
        state = "no_structure"

    return {
        "structure_state": state,
        "fact_scaffold_present": fact_scaffold_present,
        "answer_field_present": answer_field_present,
        "full_keyfacts_contract_present": full_keyfacts_contract_present,
        "labeled_fact_scaffold_present": labeled_fact_scaffold_present,
        "numbered_fact_scaffold_present": numbered_fact_scaffold_present,
        "has_numbered_fact_1": bool(weak.get("has_numbered_fact_1")),
        "has_numbered_fact_2": bool(weak.get("has_numbered_fact_2")),
        "numbered_fact_1_nonempty": bool(weak.get("numbered_fact_1_nonempty")),
        "numbered_fact_2_nonempty": bool(weak.get("numbered_fact_2_nonempty")),
        "numbered_answer_nonempty": bool(weak.get("numbered_answer_nonempty")),
        "numbered_answer_header_present": bool(weak.get("numbered_answer_header_present")),
        "numbered_key_facts_distinct": bool(weak.get("numbered_key_facts_distinct")),
        "parsed_numbered_key_fact_1": str(weak.get("numbered_key_fact_1") or "").strip(),
        "parsed_numbered_key_fact_2": str(weak.get("numbered_key_fact_2") or "").strip(),
        "parsed_numbered_answer": str(weak.get("numbered_answer") or "").strip(),
        "char_spans": dict(weak.get("char_spans") or {}),
    }


def classify_hotpot_keyfacts_first_broken_step(
    raw_text: str,
    *,
    parsed: Optional[Dict[str, Any]] = None,
    generated_structure: Optional[Dict[str, Any]] = None,
    is_correct: Optional[bool] = None,
) -> Dict[str, Any]:
    """Classify the earliest visible failure point for the `keyfacts` contract."""
    parsed_obj = parsed or parse_hotpot_reasoning_fields(raw_text, output_contract="keyfacts")
    structure = dict(generated_structure or classify_hotpot_keyfacts_structure(raw_text, parsed=parsed_obj))
    fields = dict(parsed_obj.get("fields") or {})

    key_fact_1 = str(((fields.get("key_fact_1") or {}).get("text")) or "").strip()
    key_fact_2 = str(((fields.get("key_fact_2") or {}).get("text")) or "").strip()
    answer_text = str(((fields.get("answer") or {}).get("text")) or "").strip()

    has_key_fact_1 = bool(((fields.get("key_fact_1") or {}).get("present")) or key_fact_1)
    has_key_fact_2 = bool(((fields.get("key_fact_2") or {}).get("present")) or key_fact_2)
    has_answer_header = bool((fields.get("answer") or {}).get("present"))
    answer_nonempty = bool((fields.get("answer") or {}).get("nonempty")) and bool(answer_text)
    answer_word_count = _token_count(answer_text)
    answer_has_field_marker = bool(_ANSWER_FIELD_MARKER_RE.search(answer_text))
    answer_has_linebreak = "\n" in answer_text
    answer_looks_list_like = bool(re.match(r"^\s*(?:[-*]|\d+[.)])", answer_text))
    answer_has_sentence_punct = any(ch in answer_text for ch in ".;")
    answer_too_long = answer_word_count > 8
    answer_extraction_spill = bool(
        answer_text
        and (
            answer_has_field_marker
            or answer_has_linebreak
            or answer_looks_list_like
            or answer_too_long
            or (answer_has_sentence_punct and answer_word_count > 4)
        )
    )
    answer_usable = bool(answer_nonempty and not answer_extraction_spill)

    if not has_key_fact_1:
        bucket = "pretoken_structure_miss"
        reason = "missing_key_fact_1"
    elif not has_key_fact_2:
        bucket = "reached_key_fact_1_only"
        reason = "missing_key_fact_2"
    elif not has_answer_header or not answer_nonempty:
        bucket = "reached_key_fact_2_missing_answer"
        reason = "missing_or_empty_answer_field"
    elif answer_extraction_spill:
        bucket = "answer_extraction_spill"
        reason = "answer_field_contaminated"
    elif bool(is_correct):
        bucket = "full_contract_correct_answer"
        reason = "correct_answer"
    else:
        bucket = "full_contract_wrong_answer"
        reason = "wrong_answer"

    return {
        "bucket": bucket,
        "reason": reason,
        "has_key_fact_1": has_key_fact_1,
        "has_key_fact_2": has_key_fact_2,
        "has_answer_header": has_answer_header,
        "answer_nonempty": answer_nonempty,
        "answer_usable": answer_usable,
        "answer_extraction_spill": answer_extraction_spill,
        "answer_word_count": answer_word_count,
        "answer_has_field_marker": answer_has_field_marker,
        "answer_has_linebreak": answer_has_linebreak,
        "answer_looks_list_like": answer_looks_list_like,
        "answer_has_sentence_punct": answer_has_sentence_punct,
        "answer_too_long": answer_too_long,
        "parsed_key_fact_1": key_fact_1,
        "parsed_key_fact_2": key_fact_2,
        "parsed_answer": answer_text,
        "full_contract_correct": bucket == "full_contract_correct_answer",
        "full_contract_complete": bucket in {
            "answer_extraction_spill",
            "full_contract_wrong_answer",
            "full_contract_correct_answer",
        },
    }


def _field_present_nonempty(parsed: Dict[str, Any], field_name: str) -> Tuple[bool, bool, str]:
    fields = dict(parsed.get("fields") or {})
    field = dict(fields.get(field_name) or {})
    if field_name == "supporting_facts":
        facts = [str(item).strip() for item in (parsed.get("supporting_facts") or []) if str(item).strip()]
        return bool(field.get("present") or facts), bool(facts), "\n".join(facts)
    text = str(field.get("text") or "").strip()
    return bool(field.get("present") or text), bool(field.get("nonempty") or text), text


def _close_field_contaminated(text: str) -> Dict[str, Any]:
    close_text = str(text or "").strip()
    word_count = _token_count(close_text)
    has_field_marker = bool(_ANSWER_FIELD_MARKER_RE.search(close_text))
    has_linebreak = "\n" in close_text
    looks_list_like = bool(re.match(r"^\s*(?:[-*]|\d+[.)])", close_text))
    has_sentence_punct = any(ch in close_text for ch in ".;")
    too_long = word_count > 8
    contaminated = bool(
        close_text
        and (
            has_field_marker
            or has_linebreak
            or looks_list_like
            or too_long
            or (has_sentence_punct and word_count > 4)
        )
    )
    return {
        "close_field_contaminated": contaminated,
        "close_field_word_count": word_count,
        "close_field_has_field_marker": has_field_marker,
        "close_field_has_linebreak": has_linebreak,
        "close_field_looks_list_like": looks_list_like,
        "close_field_has_sentence_punct": has_sentence_punct,
        "close_field_too_long": too_long,
    }


def classify_hotpot_contract_first_broken_step(
    raw_text: str,
    *,
    output_contract: str,
    parsed: Optional[Dict[str, Any]] = None,
    is_correct: Optional[bool] = None,
) -> Dict[str, Any]:
    """Classify a contract-local first broken field without keyfacts-only buckets."""
    contract = normalize_hotpot_reasoning_output_contract(output_contract)
    if contract == "keyfacts":
        parsed_obj = parsed or parse_hotpot_reasoning_fields(raw_text, output_contract=contract)
        keyfacts = classify_hotpot_keyfacts_first_broken_step(raw_text, parsed=parsed_obj, is_correct=is_correct)
        all_preclose = bool(keyfacts.get("has_key_fact_1") and keyfacts.get("has_key_fact_2"))
        close_present = bool(keyfacts.get("has_answer_header"))
        close_nonempty = bool(keyfacts.get("answer_nonempty"))
        contaminated = bool(keyfacts.get("answer_extraction_spill"))
        if not bool(keyfacts.get("has_key_fact_1")):
            generic_stage = "pre_boundary"
        elif not bool(keyfacts.get("has_key_fact_2")):
            generic_stage = "last_preclose"
        elif not close_present or not close_nonempty:
            generic_stage = "last_preclose"
        elif contaminated:
            generic_stage = "post_close"
        else:
            generic_stage = "close_field"
        return {
            **keyfacts,
            "output_contract": contract,
            "generic_stage": generic_stage,
            "ordered_fields": list(hotpot_contract_ordered_fields(contract)),
            "preclose_fields": list(hotpot_contract_descriptor(contract)["preclose_fields"]),
            "close_field": "answer",
            "all_preclose_fields_present": all_preclose,
            "close_field_present": close_present,
            "close_field_nonempty": close_nonempty,
            "close_field_contaminated": contaminated,
            "close_field_word_count": keyfacts.get("answer_word_count"),
            "close_field_has_field_marker": keyfacts.get("answer_has_field_marker"),
            "close_field_has_linebreak": keyfacts.get("answer_has_linebreak"),
            "close_field_looks_list_like": keyfacts.get("answer_looks_list_like"),
            "close_field_has_sentence_punct": keyfacts.get("answer_has_sentence_punct"),
            "close_field_too_long": keyfacts.get("answer_too_long"),
        }

    parsed_obj = parsed or parse_hotpot_reasoning_fields(raw_text, output_contract=contract)
    descriptor = hotpot_contract_descriptor(contract)
    ordered_fields = tuple(descriptor["ordered_fields"])
    preclose_fields = tuple(descriptor["preclose_fields"])
    close_field = str(descriptor["close_field"])

    field_payloads: Dict[str, Dict[str, Any]] = {}
    for field_name in ordered_fields:
        present, nonempty, text = _field_present_nonempty(parsed_obj, field_name)
        field_payloads[field_name] = {
            "present": present,
            "nonempty": nonempty,
            "text": text,
        }

    missing_preclose = [
        field_name for field_name in preclose_fields if not bool(field_payloads.get(field_name, {}).get("nonempty"))
    ]
    close_payload = field_payloads.get(close_field, {})
    close_present = bool(close_payload.get("present"))
    close_nonempty = bool(close_payload.get("nonempty"))
    contamination = _close_field_contaminated(str(close_payload.get("text") or ""))
    all_preclose_nonempty = not missing_preclose

    if missing_preclose:
        first_missing = str(missing_preclose[0])
        bucket = f"missing_{first_missing}"
        reason = f"missing_or_empty_{first_missing}"
        generic_stage = "pre_boundary" if first_missing != preclose_fields[-1] else "last_preclose"
    elif not close_present or not close_nonempty:
        bucket = f"reached_{preclose_fields[-1]}_missing_{close_field}" if preclose_fields else f"missing_{close_field}"
        reason = f"missing_or_empty_{close_field}"
        generic_stage = "last_preclose"
    elif contamination["close_field_contaminated"]:
        bucket = f"{close_field}_spill_or_contamination"
        reason = "close_field_contaminated"
        generic_stage = "post_close"
    elif bool(is_correct):
        bucket = "full_contract_correct_answer"
        reason = "correct_answer"
        generic_stage = "close_field"
    else:
        bucket = "full_contract_wrong_answer"
        reason = "wrong_answer"
        generic_stage = "close_field"

    return {
        "bucket": bucket,
        "reason": reason,
        "output_contract": contract,
        "generic_stage": generic_stage,
        "ordered_fields": list(ordered_fields),
        "preclose_fields": list(preclose_fields),
        "close_field": close_field,
        "all_preclose_fields_present": bool(all_preclose_nonempty),
        "close_field_present": close_present,
        "close_field_nonempty": close_nonempty,
        "full_contract_complete": bool(all_preclose_nonempty and close_present and close_nonempty),
        "full_contract_correct": bool(all_preclose_nonempty and close_present and close_nonempty and is_correct),
        "field_states": field_payloads,
        **contamination,
    }


def classify_hotpot_contract_late_boundary_state(
    raw_text: str,
    *,
    output_contract: str,
    parsed: Optional[Dict[str, Any]] = None,
    is_correct: Optional[bool] = None,
) -> Dict[str, Any]:
    """Return generic late-boundary/readiness flags for one generated output."""
    contract = normalize_hotpot_reasoning_output_contract(output_contract)
    parsed_obj = parsed or parse_hotpot_reasoning_fields(raw_text, output_contract=contract)
    broken = classify_hotpot_contract_first_broken_step(
        raw_text,
        output_contract=contract,
        parsed=parsed_obj,
        is_correct=is_correct,
    )
    all_preclose = bool(broken.get("all_preclose_fields_present"))
    close_present = bool(broken.get("close_field_present"))
    close_nonempty = bool(broken.get("close_field_nonempty"))
    contaminated = bool(broken.get("close_field_contaminated"))
    healthy_full_contract = bool(all_preclose and close_present and close_nonempty and not contaminated and is_correct)
    late_boundary_target = bool(all_preclose and (not close_present or not close_nonempty or contaminated))
    recovered_contrast_candidate = bool(healthy_full_contract)
    stable_control_candidate = bool(healthy_full_contract)
    return {
        **broken,
        "late_boundary_label": str(hotpot_contract_descriptor(contract)["late_boundary_label"]),
        "late_boundary_target": late_boundary_target,
        "healthy_full_contract": healthy_full_contract,
        "stable_control_candidate": stable_control_candidate,
        "recovered_contrast_candidate": recovered_contrast_candidate,
    }


def _iter_lines_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    lines: List[Tuple[str, int, int]] = []
    cursor = 0
    for raw_line in text.splitlines(keepends=True):
        if raw_line.endswith("\n"):
            line = raw_line[:-1]
        else:
            line = raw_line
        start = cursor
        end = cursor + len(line)
        lines.append((line, start, end))
        cursor += len(raw_line)
    if not lines and text == "":
        return []
    return lines


def parse_hotpot_reasoning_fields(raw_text: str, output_contract: Optional[str] = None) -> Dict[str, Any]:
    """Parse structured Hotpot reasoning output into contract-aware fields."""
    text = str(raw_text or "")
    contract = normalize_hotpot_reasoning_output_contract(output_contract)
    fields: Dict[str, Dict[str, Any]] = {
        "step_1": _empty_field(),
        "step_2": _empty_field(),
        "supporting_facts": _empty_field(),
        "final_answer": _empty_field(),
        "key_fact_1": _empty_field(),
        "key_fact_2": _empty_field(),
        "answer": _empty_field(),
    }
    supporting_fact_spans: List[List[int]] = []
    supporting_facts: List[str] = []
    in_support_block = False

    for line, line_start, _line_end in _iter_lines_with_offsets(text):
        stripped = line.strip()
        if not stripped:
            continue

        match = _STEP_1_RE.match(line)
        if match:
            fields["step_1"] = _line_field_entry(text, line_start, match)
            in_support_block = False
            continue

        match = _STEP_2_RE.match(line)
        if match:
            fields["step_2"] = _line_field_entry(text, line_start, match)
            in_support_block = False
            continue

        match = _KEY_FACT_1_RE.match(line)
        if match:
            fields["key_fact_1"] = _line_field_entry(text, line_start, match)
            in_support_block = False
            continue

        match = _KEY_FACT_2_RE.match(line)
        if match:
            fields["key_fact_2"] = _line_field_entry(text, line_start, match)
            in_support_block = False
            continue

        match = _FINAL_ANSWER_RE.match(line)
        if match:
            fields["final_answer"] = _line_field_entry(text, line_start, match)
            in_support_block = False
            continue

        match = _ANSWER_RE.match(line)
        if match:
            fields["answer"] = _line_field_entry(text, line_start, match)
            in_support_block = False
            continue

        match = _SUPPORTING_FACTS_RE.match(line)
        if match:
            fields["supporting_facts"] = {
                "present": True,
                "header_span": [
                    int(line_start + match.start(1)),
                    int(line_start + match.end(1)),
                ],
                "value_span": None,
                "text": "",
                "nonempty": False,
            }
            in_support_block = True
            continue

        if in_support_block:
            bullet_match = _BULLET_RE.match(line)
            if bullet_match:
                value_start = int(line_start + bullet_match.start(1))
                value_end = int(line_start + bullet_match.end(1))
                supporting_facts.append(text[value_start:value_end].strip())
                supporting_fact_spans.append([value_start, value_end])
                continue
            in_support_block = False

    fallback_lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered.startswith("supporting facts"):
            continue
        if _STEP_1_RE.match(line) or _STEP_2_RE.match(line):
            continue
        if _KEY_FACT_1_RE.match(line) or _KEY_FACT_2_RE.match(line):
            continue
        if _FINAL_ANSWER_RE.match(line) or _ANSWER_RE.match(line):
            continue
        if stripped.startswith(("-", "*")):
            continue
        fallback_lines.append(stripped)
    fallback_answer = fallback_lines[-1] if fallback_lines else ""

    if contract == "keyfacts":
        parsed_support_facts = [
            fields["key_fact_1"]["text"],
            fields["key_fact_2"]["text"],
        ]
        parsed_support_facts = [fact for fact in parsed_support_facts if fact]
        answer_text = fields["answer"]["text"] or fields["final_answer"]["text"] or fallback_answer
    elif contract == "answer_only":
        parsed_support_facts = []
        answer_text = fields["answer"]["text"] or fallback_answer
    else:
        parsed_support_facts = [fact for fact in supporting_facts if fact]
        answer_text = fields["final_answer"]["text"] or fields["answer"]["text"] or fallback_answer

    return {
        "contract": contract,
        "fields": fields,
        "supporting_facts": parsed_support_facts,
        "supporting_fact_value_spans": supporting_fact_spans,
        "answer": answer_text.strip(),
    }


def extract_hotpot_reasoning_field_char_spans(
    raw_text: str,
    output_contract: Optional[str] = None,
) -> Dict[str, Any]:
    """Return simple char-span mappings for the parsed Hotpot reasoning fields."""
    parsed = parse_hotpot_reasoning_fields(raw_text, output_contract=output_contract)
    fields = parsed["fields"]
    return {
        "step_1_header_span": fields["step_1"]["header_span"],
        "step_1_value_span": fields["step_1"]["value_span"],
        "step_2_header_span": fields["step_2"]["header_span"],
        "step_2_value_span": fields["step_2"]["value_span"],
        "supporting_facts_header_span": fields["supporting_facts"]["header_span"],
        "supporting_fact_value_spans": parsed["supporting_fact_value_spans"],
        "final_answer_header_span": fields["final_answer"]["header_span"],
        "final_answer_value_span": fields["final_answer"]["value_span"],
        "key_fact_1_header_span": fields["key_fact_1"]["header_span"],
        "key_fact_1_value_span": fields["key_fact_1"]["value_span"],
        "key_fact_2_header_span": fields["key_fact_2"]["header_span"],
        "key_fact_2_value_span": fields["key_fact_2"]["value_span"],
        "answer_header_span": fields["answer"]["header_span"],
        "answer_value_span": fields["answer"]["value_span"],
    }
