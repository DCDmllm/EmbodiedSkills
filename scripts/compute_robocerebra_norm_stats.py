#!/usr/bin/env python3
"""Compute RoboCerebra norm_stats from LeRobot parquet data.

The formal training stats should be computed from raw per-frame LeRobot
``observation.state`` and ``action`` values. This script does not use RoboTwin
stats, does not read raw RoboCerebra, and does not decode video.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DATA_ROOT = Path("/mnt/raid1/mjh/datasets/robocerebra_lerobot_unified")
DEFAULT_INDEX_JSONL = Path("outputs/robocerebra_lerobot_full_index.jsonl")
DEFAULT_SAVE_PATH = Path("outputs/openpi_assets/robocerebra_unified_full/norm_stats.json")
DEFAULT_OLD_STATS = Path("outputs/openpi_assets/robocerebra_unified/norm_stats.json")
DEFAULT_SUMMARY = Path("outputs/robocerebra_full_norm_stats_summary.json")
DEFAULT_PROBE = Path("outputs/robocerebra_full_norm_stats_probe.md")


def parse_max_frames(value: str) -> int | None:
    if value.lower() == "all":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--max_frames must be a positive integer or 'all'")
    return parsed


def load_index_summary(index_jsonl: Path) -> dict[str, Any]:
    rows = 0
    total_frames = 0
    all_aligned = True
    with index_jsonl.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows += 1
            total_frames += int(row["num_frames"])
            all_aligned = all_aligned and bool(row.get("aligned", False))
    return {"episodes": rows, "frames": total_frames, "all_aligned": all_aligned}


def stack_column(df: pd.DataFrame, column: str, max_frames: int | None) -> np.ndarray:
    values = df[column].to_numpy()
    if max_frames is not None:
        values = values[:max_frames]
    return np.stack(values).astype(np.float64)


def stats_for(values: np.ndarray) -> dict[str, Any]:
    finite_mask = np.isfinite(values)
    row_valid = finite_mask.all(axis=1)
    finite_values = values[row_valid]
    if len(finite_values) == 0:
        raise ValueError("No finite rows available for statistics")
    return {
        "mean": finite_values.mean(axis=0).astype(float).tolist(),
        "std": finite_values.std(axis=0).astype(float).tolist(),
        "min": finite_values.min(axis=0).astype(float).tolist(),
        "max": finite_values.max(axis=0).astype(float).tolist(),
        "q01": np.quantile(finite_values, 0.01, axis=0).astype(float).tolist(),
        "q99": np.quantile(finite_values, 0.99, axis=0).astype(float).tolist(),
        "valid_frame_count": int(row_valid.sum()),
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
        "invalid_row_count": int((~row_valid).sum()),
    }


def openpi_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean": stats["mean"],
        "std": stats["std"],
        "min": stats["min"],
        "max": stats["max"],
        "q01": stats["q01"],
        "q99": stats["q99"],
    }


def load_old_stats(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["norm_stats"]


def diff_vectors(new: list[float], old: list[float] | None) -> list[float] | None:
    if old is None:
        return None
    return (np.asarray(new, dtype=np.float64) - np.asarray(old, dtype=np.float64)).astype(float).tolist()


def ratio_vectors(new: list[float], old: list[float] | None) -> list[float] | None:
    if old is None:
        return None
    old_arr = np.asarray(old, dtype=np.float64)
    return (np.asarray(new, dtype=np.float64) / np.where(np.abs(old_arr) < 1e-12, np.nan, old_arr)).astype(float).tolist()


def format_vec(values: list[float] | None, precision: int = 6) -> str:
    if values is None:
        return "n/a"
    return "[" + ", ".join(f"{v:.{precision}g}" for v in values) + "]"


def compute(args: argparse.Namespace) -> dict[str, Any]:
    data_paths = sorted((args.data_root / "data").glob("chunk-*/file-*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"No data parquet files under {args.data_root / 'data'}")
    frames = [pd.read_parquet(path) for path in data_paths]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    df = df.sort_values("index").reset_index(drop=True)
    max_frames = parse_max_frames(args.max_frames)
    if max_frames is not None:
        df = df.iloc[:max_frames]

    states = stack_column(df, "observation.state", None)
    actions = stack_column(df, "action", None)
    if states.shape[1] != 8:
        raise ValueError(f"Expected state dim 8, got {states.shape}")
    if actions.shape[1] != 7:
        raise ValueError(f"Expected action dim 7, got {actions.shape}")

    state_stats = stats_for(states)
    action_stats = stats_for(actions)
    old_stats = load_old_stats(args.old_stats)
    index_summary = load_index_summary(args.index_jsonl)

    norm_stats = {
        "norm_stats": {
            "state": openpi_stats(state_stats),
            "actions": openpi_stats(action_stats),
        }
    }
    comparison = {
        "old_stats_path": str(args.old_stats),
        "old_stats_found": old_stats is not None,
        "state_std_delta": diff_vectors(state_stats["std"], old_stats["state"]["std"] if old_stats else None),
        "action_std_delta": diff_vectors(action_stats["std"], old_stats["actions"]["std"] if old_stats else None),
        "state_std_ratio_new_over_old": ratio_vectors(state_stats["std"], old_stats["state"]["std"] if old_stats else None),
        "action_std_ratio_new_over_old": ratio_vectors(action_stats["std"], old_stats["actions"]["std"] if old_stats else None),
    }
    near_zero_state_std = [i for i, value in enumerate(state_stats["std"]) if abs(value) < args.near_zero_std_threshold]
    near_zero_action_std = [i for i, value in enumerate(action_stats["std"]) if abs(value) < args.near_zero_std_threshold]
    abnormal_action_dims = [
        i
        for i, value in enumerate(np.maximum(np.abs(action_stats["min"]), np.abs(action_stats["max"])))
        if value > args.abnormal_action_threshold
    ]
    summary = {
        "data_root": str(args.data_root),
        "index_jsonl": str(args.index_jsonl),
        "data_parquet_files": [str(path) for path in data_paths],
        "max_frames": args.max_frames,
        "valid_frame_count": int(min(state_stats["valid_frame_count"], action_stats["valid_frame_count"])),
        "state_raw_dim": int(states.shape[1]),
        "action_raw_dim": int(actions.shape[1]),
        "state_nan_count": state_stats["nan_count"],
        "state_inf_count": state_stats["inf_count"],
        "action_nan_count": action_stats["nan_count"],
        "action_inf_count": action_stats["inf_count"],
        "state_invalid_row_count": state_stats["invalid_row_count"],
        "action_invalid_row_count": action_stats["invalid_row_count"],
        "index_episode_count": index_summary["episodes"],
        "index_frame_count": index_summary["frames"],
        "index_all_aligned": index_summary["all_aligned"],
        "state_std_near_zero_dims": near_zero_state_std,
        "action_std_near_zero_dims": near_zero_action_std,
        "abnormal_action_dims": abnormal_action_dims,
        "abnormal_action_threshold": args.abnormal_action_threshold,
        "norm_stats_path": str(args.save_path),
        "summary_path": str(args.summary_path),
        "probe_path": str(args.probe_path),
        "comparison": comparison,
        "full_stats": {
            "state": state_stats,
            "actions": action_stats,
        },
    }
    return {"norm_stats": norm_stats, "summary": summary, "old_stats": old_stats}


def write_probe(path: Path, summary: dict[str, Any], old_stats: dict[str, Any] | None) -> None:
    state = summary["full_stats"]["state"]
    actions = summary["full_stats"]["actions"]
    comparison = summary["comparison"]
    old_state = old_stats["state"] if old_stats else None
    old_actions = old_stats["actions"] if old_stats else None
    lines = [
        "# RoboCerebra Full Norm Stats Probe",
        "",
        f"- data_root: `{summary['data_root']}`",
        f"- index_jsonl: `{summary['index_jsonl']}`",
        f"- max_frames: `{summary['max_frames']}`",
        f"- valid frame count: {summary['valid_frame_count']}",
        f"- index frame count: {summary['index_frame_count']}",
        f"- index episodes: {summary['index_episode_count']}",
        f"- index all aligned: {summary['index_all_aligned']}",
        f"- new full stats: `{summary['norm_stats_path']}`",
        f"- old stats: `{comparison['old_stats_path']}`",
        f"- old stats found: {comparison['old_stats_found']}",
        "",
        "## Integrity",
        "",
        f"- state dim: {summary['state_raw_dim']}",
        f"- action dim: {summary['action_raw_dim']}",
        f"- state NaN / Inf: {summary['state_nan_count']} / {summary['state_inf_count']}",
        f"- action NaN / Inf: {summary['action_nan_count']} / {summary['action_inf_count']}",
        f"- state invalid rows: {summary['state_invalid_row_count']}",
        f"- action invalid rows: {summary['action_invalid_row_count']}",
        f"- state std near-zero dims: {summary['state_std_near_zero_dims']}",
        f"- action std near-zero dims: {summary['action_std_near_zero_dims']}",
        f"- abnormal action dims over threshold {summary['abnormal_action_threshold']}: {summary['abnormal_action_dims']}",
        "",
        "## New Full Stats",
        "",
        f"- state mean: `{format_vec(state['mean'])}`",
        f"- state std: `{format_vec(state['std'])}`",
        f"- state min: `{format_vec(state['min'])}`",
        f"- state max: `{format_vec(state['max'])}`",
        f"- action mean: `{format_vec(actions['mean'])}`",
        f"- action std: `{format_vec(actions['std'])}`",
        f"- action min: `{format_vec(actions['min'])}`",
        f"- action max: `{format_vec(actions['max'])}`",
        f"- action p01: `{format_vec(actions['q01'])}`",
        f"- action p99: `{format_vec(actions['q99'])}`",
        "",
        "## Old vs New",
        "",
    ]
    if old_stats:
        lines.extend(
            [
                f"- old state std: `{format_vec(old_state['std'])}`",
                f"- new state std: `{format_vec(state['std'])}`",
                f"- state std delta: `{format_vec(comparison['state_std_delta'])}`",
                f"- state std ratio new/old: `{format_vec(comparison['state_std_ratio_new_over_old'])}`",
                f"- old action std: `{format_vec(old_actions['std'])}`",
                f"- new action std: `{format_vec(actions['std'])}`",
                f"- action std delta: `{format_vec(comparison['action_std_delta'])}`",
                f"- action std ratio new/old: `{format_vec(comparison['action_std_ratio_new_over_old'])}`",
            ]
        )
    else:
        lines.append("- old stats not found")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- These stats are computed from full LeRobot parquet frame data only.",
            "- No raw RoboCerebra data was downloaded or read.",
            "- No videos were decoded and no PNGs were generated.",
            "- The previous 5000-frame stats should not be used as formal training stats.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--index_jsonl", type=Path, default=DEFAULT_INDEX_JSONL)
    parser.add_argument("--max_frames", default="all")
    parser.add_argument("--save_path", type=Path, default=DEFAULT_SAVE_PATH)
    parser.add_argument("--old_stats", type=Path, default=DEFAULT_OLD_STATS)
    parser.add_argument("--summary_path", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--probe_path", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--near_zero_std_threshold", type=float, default=1e-6)
    parser.add_argument("--abnormal_action_threshold", type=float, default=20.0)
    # Backward-compatible aliases from the old script.
    parser.add_argument("--data-parquet", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.data_parquet is not None:
        args.data_root = args.data_parquet.parents[2]
    if args.output_dir is not None:
        args.save_path = args.output_dir / "norm_stats.json"

    result = compute(args)
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    args.save_path.write_text(json.dumps(result["norm_stats"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(result["summary"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_probe(args.probe_path, result["summary"], result["old_stats"])
    print(json.dumps({k: v for k, v in result["summary"].items() if k != "full_stats"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
