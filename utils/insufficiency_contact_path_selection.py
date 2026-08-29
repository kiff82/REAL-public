from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence


_ONE_TOKEN_REJECTS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
}

_DANGLING_TAILS = {
    "and",
    "or",
    "of",
    "the",
    "a",
    "an",
    "to",
    "in",
    "for",
    "with",
    "by",
    "from",
}

_QUESTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "with",
}


def contact_prefix_labels(mode: str, prefix_count: int) -> List[str]:
    """Stable labels for contact prefixes selected by the objective mode."""
    mode = str(mode or "final")
    prefix_count = max(0, int(prefix_count))
    if prefix_count <= 0:
        return []
    if mode == "final":
        return ["final"][:prefix_count]
    if mode in {"entry_final", "early_final"}:
        labels = ["entry", "final"]
    elif mode == "entry_mid_final":
        labels = ["entry", "mid", "final"]
    elif mode == "all":
        labels = [f"step_{idx}" for idx in range(prefix_count)]
    else:
        labels = [f"step_{idx}" for idx in range(prefix_count)]
    if len(labels) == prefix_count:
        return labels
    if len(labels) > prefix_count:
        return labels[:prefix_count]
    return labels + [f"step_{idx}" for idx in range(len(labels), prefix_count)]


def clean_insufficiency_candidate_text(
    text: str,
    *,
    abstain_text: str = "",
    source: str = "",
    max_capitalized_tokens: int = 8,
) -> Optional[str]:
    """Clean and reject malformed hard-negative candidate spans."""
    raw = str(text or "")
    cleaned = re.sub(r"\s+", " ", raw).strip()
    cleaned = cleaned.strip(" \t\r\n\"'`.,;:!?()[]{}<>")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None
    if not any(ch.isalnum() for ch in cleaned):
        return None
    if abstain_text and cleaned.casefold() == str(abstain_text).strip().casefold():
        return None
    tokens = cleaned.split()
    if len(tokens) == 1 and tokens[0].casefold() in _ONE_TOKEN_REJECTS:
        return None
    tail = tokens[-1].strip(".,;:!?()[]{}\"'`").casefold()
    if tail in _DANGLING_TAILS:
        return None
    if re.search(r"(?:'s|’s)$", tokens[-1]) and str(source) != "answer_bank":
        return None
    if str(source) != "answer_bank":
        capitalized = [tok for tok in tokens if tok[:1].isupper()]
        if len(tokens) > max_capitalized_tokens and len(capitalized) >= max(3, len(tokens) - 1):
            return None
    return cleaned


def dedupe_candidate_records(
    records: Sequence[Dict[str, Any]],
    *,
    abstain_text: str = "",
) -> Dict[str, Any]:
    """Return cleaned candidate records plus rejection accounting."""
    cleaned_records: List[Dict[str, str]] = []
    rejected: List[str] = []
    seen = set()
    for record in records:
        source = str(record.get("source") or "")
        text = clean_insufficiency_candidate_text(
            str(record.get("text") or ""),
            abstain_text=abstain_text,
            source=source,
        )
        if text is None:
            raw = str(record.get("text") or "").strip()
            if raw and len(rejected) < 12:
                rejected.append(raw)
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned_records.append({"text": text, "source": source})
    return {
        "records": cleaned_records,
        "rejected_count": max(0, len(records) - len(cleaned_records)),
        "rejected_examples": rejected,
    }


def question_terms(question: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'’-]*", str(question or "")):
        key = token.casefold()
        if len(key) < 3 or key in _QUESTION_STOPWORDS or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def select_hard_negative_by_scores(
    *,
    positive_scores_by_step: Sequence[float],
    negative_scores_by_candidate: Sequence[Sequence[float]],
    selection_mode: str,
    top_k: int = 1,
    step_labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Select fixed hard negatives from already-computed per-step scores."""
    positives = [float(value) for value in positive_scores_by_step]
    labels = list(step_labels or [f"step_{idx}" for idx in range(len(positives))])
    if not positives or not negative_scores_by_candidate:
        return {
            "selected_indices": [],
            "selected_index": None,
            "selected_step": "",
            "selected_score": None,
            "per_step_argmax_indices": [],
            "per_step_argmax_scores": [],
            "changed_under_per_step_argmax": False,
        }

    mode = str(selection_mode or "per_step")
    top_k = max(1, int(top_k))
    per_candidate_hard: List[List[float]] = []
    for candidate_scores in negative_scores_by_candidate:
        scores = [float(value) for value in candidate_scores]
        if len(scores) != len(positives):
            raise ValueError("candidate score length must match positive score length")
        per_candidate_hard.append([score - positives[idx] for idx, score in enumerate(scores)])

    per_step_argmax_indices: List[int] = []
    per_step_argmax_scores: List[float] = []
    for step_idx in range(len(positives)):
        best_idx = max(range(len(per_candidate_hard)), key=lambda idx: per_candidate_hard[idx][step_idx])
        per_step_argmax_indices.append(int(best_idx))
        per_step_argmax_scores.append(float(per_candidate_hard[best_idx][step_idx]))

    selection_rows: List[tuple[float, int, int]] = []
    if mode == "entry_step":
        step_idx = 0
        for idx, hard_scores in enumerate(per_candidate_hard):
            selection_rows.append((float(hard_scores[step_idx]), idx, step_idx))
    elif mode == "final_step":
        step_idx = len(positives) - 1
        for idx, hard_scores in enumerate(per_candidate_hard):
            selection_rows.append((float(hard_scores[step_idx]), idx, step_idx))
    elif mode == "max_over_steps_selected":
        for idx, hard_scores in enumerate(per_candidate_hard):
            step_idx = max(range(len(hard_scores)), key=lambda j: hard_scores[j])
            selection_rows.append((float(hard_scores[step_idx]), idx, step_idx))
    elif mode == "per_step":
        step_idx = len(positives) - 1
        for idx, hard_scores in enumerate(per_candidate_hard):
            selection_rows.append((float(hard_scores[step_idx]), idx, step_idx))
    else:
        raise ValueError(
            "selection_mode must be one of: per_step, final_step, entry_step, max_over_steps_selected"
        )

    selection_rows.sort(key=lambda item: item[0], reverse=True)
    selected_rows = selection_rows[: min(top_k, len(selection_rows))]
    selected_indices = [int(item[1]) for item in selected_rows]
    selected_step_idx = int(selected_rows[0][2])
    fixed_changed = any(idx not in selected_indices for idx in per_step_argmax_indices)
    if mode == "per_step":
        fixed_changed = len(set(per_step_argmax_indices)) > 1
    return {
        "selected_indices": selected_indices,
        "selected_index": selected_indices[0] if selected_indices else None,
        "selected_step": labels[selected_step_idx] if selected_step_idx < len(labels) else f"step_{selected_step_idx}",
        "selected_score": float(selected_rows[0][0]) if selected_rows else None,
        "per_step_argmax_indices": per_step_argmax_indices,
        "per_step_argmax_scores": per_step_argmax_scores,
        "changed_under_per_step_argmax": bool(fixed_changed),
    }
