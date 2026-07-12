from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from clawvla.components.scheduler import (
    _task_plan_instruction,
    _task_plan_style_examples,
    _vla_subgoal_instruction_style,
)
from clawvla.json_utils import extract_last_json_dict


DEFAULT_CONFIG = "configs/krill_gpt55.local.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate VLA-friendly paraphrases for short-horizon robot subgoal instructions."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Local OpenAI-compatible config JSON.")
    parser.add_argument("--input-jsonl", default=None, help="Optional JSONL records to paraphrase.")
    parser.add_argument("--output", default=None, help="Optional output JSON or JSONL path.")
    parser.add_argument("--segment-json", default=None, help="One JSON object string describing a segment.")
    parser.add_argument("--demo", action="store_true", help="Run built-in RoboTwin-style examples.")
    parser.add_argument("--n", type=int, default=8, help="Number of paraphrases per record.")
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--max-tokens", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(Path(args.config))
    client = OpenAI(
        api_key=str(config["api_key"]),
        base_url=str(config["base_url"]),
        timeout=float(config.get("timeout", 120.0)),
    )

    records = _load_records(args)
    outputs = [
        _paraphrase_record(
            client=client,
            model=str(config["model"]),
            record=record,
            n=args.n,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        for record in records
    ]

    rendered = json.dumps(outputs if len(outputs) != 1 else outputs[0], indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.input_jsonl and len(outputs) > 1:
            lines = [json.dumps(item, ensure_ascii=False) for item in outputs]
            output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            output_path.write_text(rendered + "\n", encoding="utf-8")


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in ("model", "base_url", "api_key") if not payload.get(key)]
    if missing:
        raise ValueError(f"{path} is missing required key(s): {', '.join(missing)}")
    return payload


def _load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    sources = sum(bool(item) for item in (args.input_jsonl, args.segment_json, args.demo))
    if sources > 1:
        raise ValueError("Use only one of --input-jsonl, --segment-json, or --demo.")
    if args.input_jsonl:
        return [
            json.loads(line)
            for line in Path(args.input_jsonl).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if args.segment_json:
        payload = json.loads(args.segment_json)
        if not isinstance(payload, dict):
            raise ValueError("--segment-json must be one JSON object.")
        return [payload]
    return _demo_records()


def _demo_records() -> list[dict[str, Any]]:
    examples = {str(example["pattern"]): example for example in _task_plan_style_examples()}
    selections = [
        ("single_arm_pick_place", 0),
        ("single_arm_pick_place", 2),
        ("dual_arm_two_objects", 0),
        ("direct_press_without_grasp", 1),
    ]
    records: list[dict[str, Any]] = []
    for pattern, subgoal_index in selections:
        example = examples[pattern]
        subgoal = example["subgoals"][subgoal_index]
        records.append(
            {
                "task_instruction": example["task_instruction"],
                "subgoal_type": subgoal["type"],
                "subgoal_instruction": subgoal["instruction"],
                "style_pattern": pattern,
            }
        )
    return records


def _paraphrase_record(
    *,
    client: OpenAI,
    model: str,
    record: dict[str, Any],
    n: int,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    prompt_payload = {
        "style_source": _task_plan_instruction(),
        "instruction_style_examples": _task_plan_style_examples(),
        "record": record,
        "required_schema": {
            "paraphrases": [
                {
                    "instruction": "one concise VLA-ready command, usually 6-14 words and max about 18",
                    "style_note": "short reason this keeps the same semantics",
                }
            ],
            "rejected_examples": [
                {
                    "instruction": "bad instruction example",
                    "reason": "why it would be unsafe or semantically wrong",
                }
            ],
        },
        "count": n,
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You rewrite short-horizon robot subgoal instructions for a vision-language-action policy. "
                "Preserve object names, colors, arm assignments, tool/object roles, and source-target relations "
                "exactly. Do not swap left/right arms, do not invent objects, and do not weaken concrete targets "
                "into vague nouns. Every paraphrase must request a physical action, not a check, confirmation, "
                "observation, or verification. "
                f"{_vla_subgoal_instruction_style()} "
                "Keep the style close to the provided training-aligned examples while preserving the input "
                "subgoal's exact physical stage."
            ),
        },
        {
            "role": "user",
            "content": (
                "Generate paraphrases for this subgoal. Return exactly one JSON object and no markdown.\n\n"
                f"{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
            ),
        },
    ]
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        request_kwargs["temperature"] = temperature
    response = client.chat.completions.create(**request_kwargs)
    raw_text = response.choices[0].message.content or ""
    parsed = extract_last_json_dict(raw_text, error_prefix="paraphrase model output")
    paraphrases = parsed.get("paraphrases")
    if not isinstance(paraphrases, list):
        raise ValueError("paraphrase model output missing paraphrases list")
    return {
        "input": record,
        "model": model,
        "paraphrases": paraphrases,
        "rejected_examples": parsed.get("rejected_examples", []),
        "raw_text_preview": raw_text[:1200],
    }


if __name__ == "__main__":
    main()
