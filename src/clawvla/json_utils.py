from __future__ import annotations

import json


def extract_last_json_dict(raw_text: str, error_prefix: str = "Model output") -> dict[str, object]:
    raw_text = raw_text.strip()
    if raw_text.startswith("{") and raw_text.endswith("}"):
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(payload, dict):
                return payload

    last_payload: dict[str, object] | None = None
    for payload in top_level_json_objects(raw_text):
        last_payload = payload
    if last_payload is not None:
        return last_payload
    raise ValueError(f"{error_prefix} did not contain a JSON object: {raw_text}")


def top_level_json_objects(raw_text: str) -> list[dict[str, object]]:
    depth = 0
    in_string = False
    escaped = False
    start: int | None = None
    payloads: list[dict[str, object]] = []

    for index, char in enumerate(raw_text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0 and not _looks_like_json_start(raw_text, index):
                continue
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                candidate = raw_text[start : index + 1]
                try:
                    payload = json.loads(candidate)
                except json.JSONDecodeError:
                    start = None
                    continue
                if isinstance(payload, dict):
                    payloads.append(payload)
                start = None
    return payloads


def _looks_like_json_start(raw_text: str, index: int) -> bool:
    tail = raw_text[index + 1 :]
    for char in tail:
        if char.isspace():
            continue
        return char == '"'
    return False
