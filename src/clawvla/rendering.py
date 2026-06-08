from __future__ import annotations

import json
from html import escape
from typing import Any


def render_payload(payload: Any, fmt: str = "json", root_tag: str = "context") -> str:
    if fmt == "json":
        return render_json(payload)
    if fmt == "xml":
        return render_xml(payload, root_tag=root_tag)
    raise ValueError(f"Unsupported render format: {fmt}")


def render_json(payload: Any) -> str:
    return json.dumps(_to_plain(payload), ensure_ascii=True, indent=2)


def render_xml(payload: Any, root_tag: str = "context") -> str:
    return _xml_node(root_tag, _to_plain(payload), indent=0)


def _to_plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _to_plain(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item") and callable(value.item):
        try:
            return _to_plain(value.item())
        except Exception:
            pass
    return value


def _xml_node(tag: str, value: Any, indent: int) -> str:
    pad = " " * indent
    tag = _safe_tag(tag)
    if isinstance(value, dict):
        if not value:
            return f"{pad}<{tag}/>"
        children = "\n".join(_xml_node(str(key), item, indent + 2) for key, item in value.items())
        return f"{pad}<{tag}>\n{children}\n{pad}</{tag}>"
    if isinstance(value, list):
        if not value:
            return f"{pad}<{tag}/>"
        children = "\n".join(_xml_node("item", item, indent + 2) for item in value)
        return f"{pad}<{tag}>\n{children}\n{pad}</{tag}>"
    if value is None:
        return f"{pad}<{tag} null=\"true\"/>"
    return f"{pad}<{tag}>{escape(str(value))}</{tag}>"


def _safe_tag(tag: str) -> str:
    cleaned = []
    for index, char in enumerate(tag):
        if char.isalnum() or char in {"_", "-"}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    text = "".join(cleaned).strip("_") or "field"
    if text[0].isdigit():
        text = f"field_{text}"
    return text
