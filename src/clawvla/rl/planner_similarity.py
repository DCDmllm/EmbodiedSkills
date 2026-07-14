from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
from difflib import SequenceMatcher
from typing import Any, Iterable


_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class PlannerReference:
    task_name: str
    episode_index: int
    seed: int
    task_instruction: str
    subgoals: tuple[tuple[str, ...], ...]
    subgoal_types: tuple[str, ...]
    completion_criteria: tuple[str, ...]


@lru_cache(maxsize=8)
def load_planner_reference_index(
    dataset_root: str,
    repair_ledger: str,
    split_manifest: str,
    split_name: str = "train",
    max_reference_plans_per_task: int = 64,
) -> dict[str, tuple[PlannerReference, ...]]:
    root = Path(dataset_root).expanduser().resolve()
    repairs = _load_repairs(Path(repair_ledger))
    split_path = Path(split_manifest).expanduser().resolve()
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Planner split manifest is required for split={split_name!r}: {split_path}"
        )
    allowed = _load_split(split_path, split_name)
    by_task: dict[str, list[PlannerReference]] = {}
    for path in sorted((root / "segments").glob("*/*.json"), key=_segment_path_key):
        payload = json.loads(path.read_text(encoding="utf-8"))
        task_name = str(payload.get("task_name") or path.parent.name)
        episode_index = int(payload.get("episode_index", 0))
        if allowed is not None and episode_index not in allowed.get(task_name, set()):
            continue
        subgoals: list[tuple[str, ...]] = []
        subgoal_types: list[str] = []
        completion: list[str] = []
        for segment in payload.get("segments") or []:
            if int(segment.get("frame_end_exclusive") or 0) <= int(segment.get("frame_start") or 0):
                continue
            if int(segment.get("num_saved_frames") or 0) <= 0:
                continue
            segment_index = int(segment.get("segment_index", len(subgoals)))
            record_id = f"{task_name}/episode{episode_index}/segment{segment_index}"
            repair = repairs.get(record_id)
            variants = _instruction_variants(segment, repair)
            if not variants:
                continue
            subgoals.append(tuple(variants))
            subgoal_types.append(str((repair or {}).get("subgoal_type") or segment.get("subgoal_type") or "act"))
            completion.append(
                str((repair or {}).get("completion_criteria") or segment.get("completion_criteria") or "").strip()
            )
        if not subgoals:
            continue
        reference = PlannerReference(
            task_name=task_name,
            episode_index=episode_index,
            seed=int(payload.get("seed", episode_index)),
            task_instruction=str(
                payload.get("instruction") or payload.get("task_instruction_from_config") or task_name
            ),
            subgoals=tuple(subgoals),
            subgoal_types=tuple(subgoal_types),
            completion_criteria=tuple(completion),
        )
        task_refs = by_task.setdefault(task_name, [])
        if len(task_refs) < max(1, int(max_reference_plans_per_task)):
            task_refs.append(reference)
    return {task: tuple(references) for task, references in by_task.items()}


def score_predicted_plan(predicted: dict[str, Any] | None, references: Iterable[PlannerReference]) -> float:
    predicted_subgoals = _predicted_instructions(predicted)
    if not predicted_subgoals:
        return 0.0
    scores = [ordered_plan_similarity(predicted_subgoals, reference.subgoals) for reference in references]
    return max(scores, default=0.0)


def ordered_plan_similarity(
    predicted: list[str] | tuple[str, ...],
    reference: tuple[tuple[str, ...], ...] | list[tuple[str, ...]],
) -> float:
    """Monotonic semantic-proxy alignment with unmatched-step penalties.

    References include canonical instructions and curated paraphrases. Taking the
    best token/sequence match over those variants provides a deterministic local
    semantic score without adding another model to the rollout critical path.
    """
    if not predicted or not reference:
        return 0.0
    n, m = len(predicted), len(reference)
    pair_scores = [
        [max((_text_similarity(text, variant) for variant in variants), default=0.0) for variants in reference]
        for text in predicted
    ]
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = max(
                dp[i - 1][j],
                dp[i][j - 1],
                dp[i - 1][j - 1] + pair_scores[i - 1][j - 1],
            )
    return max(0.0, min(1.0, dp[n][m] / max(n, m)))


def is_build_task_plan_call(call: Any) -> bool:
    if str(getattr(call, "role", "") or "") != "scheduler":
        return False
    parsed = getattr(call, "parsed_json", None)
    if isinstance(parsed, dict) and "subgoals" in parsed:
        return True
    for message in getattr(call, "messages", []) or []:
        content = message.get("content") if isinstance(message, dict) else None
        texts = [content] if isinstance(content, str) else [
            item.get("text") for item in content or [] if isinstance(item, dict) and item.get("type") == "text"
        ]
        if any("Build a complete ordered manipulation subgoal plan" in str(text or "") for text in texts):
            return True
    return False


def _predicted_instructions(payload: dict[str, Any] | None) -> list[str]:
    if not isinstance(payload, dict):
        return []
    return [
        str(item.get("instruction") or "").strip()
        for item in payload.get("subgoals") or []
        if isinstance(item, dict) and str(item.get("instruction") or "").strip()
    ]


def _text_similarity(left: str, right: str) -> float:
    left_normalized = _normalize_text(left)
    right_normalized = _normalize_text(right)
    if not left_normalized or not right_normalized:
        return 0.0
    left_tokens = left_normalized.split()
    right_tokens = right_normalized.split()
    left_counts = _counts(left_tokens)
    right_counts = _counts(right_tokens)
    overlap = sum(min(left_counts.get(token, 0), right_counts.get(token, 0)) for token in left_counts)
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    token_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    sequence = SequenceMatcher(None, left_normalized, right_normalized).ratio()
    return 0.7 * token_f1 + 0.3 * sequence


def _normalize_text(value: str) -> str:
    substitutions = {
        "grab": "grasp",
        "grip": "grasp",
        "raise": "lift",
        "bring": "move",
        "put": "place",
        "onto": "on",
        "upon": "on",
    }
    tokens = [substitutions.get(token, token) for token in _TOKEN_RE.findall(str(value).lower())]
    return " ".join(tokens)


def _counts(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def _load_repairs(path: Path) -> dict[str, dict[str, Any]]:
    repairs: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return repairs
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") == "accepted" and isinstance(record.get("repair"), dict):
            repairs[str(record.get("record_id"))] = dict(record["repair"])
    return repairs


def _load_split(path: Path, split_name: str) -> dict[str, set[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = f"{split_name}_episode_indices"
    return {
        str(task): {int(value) for value in task_payload.get(key, [])}
        for task, task_payload in (payload.get("tasks") or {}).items()
        if isinstance(task_payload, dict)
    }


def _instruction_variants(segment: dict[str, Any], repair: dict[str, Any] | None) -> list[str]:
    values: list[Any] = []
    if repair:
        values.extend([repair.get("instruction"), *(repair.get("paraphrases") or [])])
    values.extend(
        [
            segment.get("polished_instruction"),
            segment.get("canonical_instruction"),
            *(segment.get("paraphrases") or []),
        ]
    )
    return list(dict.fromkeys(str(value).strip() for value in values if str(value or "").strip()))


def _segment_path_key(path: Path) -> tuple[str, int]:
    match = re.search(r"episode(\d+)", path.stem)
    return path.parent.name, int(match.group(1)) if match else 0
