from __future__ import annotations

from typing import Any

from .json_utils import extract_last_json_dict
from .rendering import render_payload
from .skills import SkillContext


def call_component_json(
    context: SkillContext,
    instruction: str,
    payload: dict[str, Any],
    image_paths: list[str] | None = None,
    render_format: str = "json",
    max_new_tokens: int | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    if context.model_runtime is None or not context.model_runtime.enabled:
        raise RuntimeError(f"Component {context.component_name} has no enabled model runtime.")

    context_text = render_payload(payload, fmt=render_format, root_tag="clawvla_context")
    prompt = (
        f"{instruction.strip()}\n\n"
        "Return exactly one JSON object. Do not include markdown fences or extra text. "
        "If the context contains required_schema, it is the exact output contract: return an object with "
        "those top-level keys directly. Do not echo required_schema, do not echo the input context, and do "
        "not wrap the answer inside keys such as result, output, response, perception, or data unless those "
        "keys are explicitly present in required_schema.\n\n"
        f"{context_text}"
    )
    content: list[dict[str, str]] = []
    for image_path in image_paths or []:
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": prompt})
    context.blackboard.append_event(
        "model.call",
        {
            "component": context.component_name,
            "image_count": len(image_paths or []),
            "render_format": render_format,
        },
    )
    raw_text = context.model_runtime.generate_text(
        messages=[{"role": "user", "content": content}],
        image_paths=image_paths,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    context.blackboard.append_event(
        "model.output",
        {
            "component": context.component_name,
            "raw_text_preview": raw_text[:4000],
            "raw_text_length": len(raw_text),
            "truncated": len(raw_text) > 4000,
        },
    )
    return extract_last_json_dict(raw_text, error_prefix=f"{context.component_name} model output")
