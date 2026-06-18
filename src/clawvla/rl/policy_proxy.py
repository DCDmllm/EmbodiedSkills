from __future__ import annotations

from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from typing import Any, Protocol
from uuid import uuid4

from clawvla.json_utils import extract_last_json_dict

from .trajectory import PolicyCallTrace, TrajectoryWriter


@dataclass
class PolicyGeneration:
    text: str
    prompt_ids: list[int] = field(default_factory=list)
    response_ids: list[int] = field(default_factory=list)
    response_logprobs: list[float] = field(default_factory=list)
    multi_modal_data: dict[str, Any] = field(default_factory=dict)
    mm_processor_kwargs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class PolicyBackend(Protocol):
    def generate(self, request: dict[str, Any], trace: PolicyCallTrace) -> PolicyGeneration:
        ...


class ExplicitUnavailableBackend:
    def generate(self, request: dict[str, Any], trace: PolicyCallTrace) -> PolicyGeneration:
        raise RuntimeError("policy_backend_unavailable")


class StaticPolicyBackend:
    def __init__(self, response: str):
        self.response = response

    def generate(self, request: dict[str, Any], trace: PolicyCallTrace) -> PolicyGeneration:
        return PolicyGeneration(text=self.response)


class OpenAIForwardBackend:
    def __init__(self, *, base_url: str, api_key: str, model: str):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def generate(self, request: dict[str, Any], trace: PolicyCallTrace) -> PolicyGeneration:
        forwarded = dict(request)
        forwarded["model"] = self.model
        response = self.client.chat.completions.create(**forwarded)
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("external_policy_returned_empty_content")
        return PolicyGeneration(text=content if isinstance(content, str) else str(content))


class PolicyProxy:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        backend: PolicyBackend | None = None,
        trajectory_writer: TrajectoryWriter | None = None,
    ):
        self.host = host
        self.port = port
        self.backend = backend or ExplicitUnavailableBackend()
        self.trajectory_writer = trajectory_writer
        self.calls: list[PolicyCallTrace] = []
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def start(self) -> None:
        if self._server is not None:
            raise RuntimeError("policy_proxy_already_started")
        proxy = self

        class Handler(_PolicyProxyHandler):
            policy_proxy = proxy

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.host = str(self._server.server_address[0])
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, name="clawvla-policy-proxy", daemon=True)
        self._thread.start()
        self._write_event("clawvla_rl_policy_proxy_started", {"base_url": self.base_url})

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._server = None
        self._thread = None
        self._write_event(
            "clawvla_rl_policy_proxy_stopped",
            {"base_url": self.base_url, "call_count": len(self.calls)},
        )

    def handle_chat_completion(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        model = str(payload.get("model") or "")
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return 400, _error_payload("invalid_request", "messages must be a list")
        trace = PolicyCallTrace.new(
            role=_role_from_model(model),
            model=model,
            messages=_compact_messages(messages),
            image_refs=_extract_image_refs(messages),
        )
        self._append_call(trace)
        self._write_event(
            "clawvla_rl_policy_call_start",
            {"call_id": trace.call_id, "role": trace.role, "model": model, "image_count": len(trace.image_refs)},
        )
        try:
            generation = self.backend.generate(payload, trace)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            trace.finish(raw_text=None, status="policy_generation_failed", error=reason)
            self._write_event(
                "clawvla_rl_policy_generation_failed",
                {"call_id": trace.call_id, "role": trace.role, "error": reason},
            )
            return 503, _error_payload("policy_generation_failed", reason)

        trace.prompt_ids = list(generation.prompt_ids)
        trace.response_ids = list(generation.response_ids)
        trace.response_logprobs = list(generation.response_logprobs)
        trace.metadata.update(generation.metadata)
        trace.metadata["multi_modal_counts"] = _multi_modal_counts(generation.multi_modal_data)
        trace.metadata["has_mm_processor_kwargs"] = bool(generation.mm_processor_kwargs)
        trace._clawvla_multi_modal_data = generation.multi_modal_data
        trace._clawvla_mm_processor_kwargs = generation.mm_processor_kwargs
        trace.finish(raw_text=generation.text, status="generated")
        try:
            trace.parsed_json = extract_last_json_dict(generation.text, error_prefix="policy proxy output")
        except Exception as exc:
            trace.metadata["json_parse_error"] = f"{type(exc).__name__}: {exc}"
            self._write_event(
                "clawvla_rl_policy_invalid_json",
                {"call_id": trace.call_id, "role": trace.role, "error": trace.metadata["json_parse_error"]},
            )
        self._write_event(
            "clawvla_rl_policy_call_finish",
            {
                "call_id": trace.call_id,
                "role": trace.role,
                "status": trace.status,
                "text_length": len(generation.text),
                "token_count": len(trace.response_ids),
            },
        )
        return 200, _completion_payload(model=model, text=generation.text)

    def _append_call(self, trace: PolicyCallTrace) -> None:
        with self._lock:
            self.calls.append(trace)

    def _write_event(self, event: str, payload: dict[str, Any]) -> None:
        if self.trajectory_writer is not None:
            self.trajectory_writer.write_event(event, payload)


class _PolicyProxyHandler(BaseHTTPRequestHandler):
    policy_proxy: PolicyProxy

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._send_json(200, {"object": "list", "data": [{"id": "clawvla-policy-proxy", "object": "model"}]})
            return
        self._send_json(404, _error_payload("not_found", self.path))

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._send_json(404, _error_payload("not_found", self.path))
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except Exception as exc:
            self._send_json(400, _error_payload("invalid_json", f"{type(exc).__name__}: {exc}"))
            return
        status, response = self.policy_proxy.handle_chat_completion(payload)
        self._send_json(status, response)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _completion_payload(*, model: str, text: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _error_payload(code: str, message: str) -> dict[str, Any]:
    return {"error": {"type": code, "message": message}}


def _multi_modal_counts(multi_modal_data: dict[str, Any]) -> dict[str, int]:
    counts = {}
    for key, value in multi_modal_data.items():
        if value is None:
            counts[key] = 0
        elif isinstance(value, (list, tuple)):
            counts[key] = len(value)
        else:
            counts[key] = 1
    return counts


def _role_from_model(model: str) -> str | None:
    if ":" in model:
        return model.rsplit(":", 1)[-1]
    for role in ("vision", "scheduler", "verifier", "recovery"):
        if model.endswith(f"-{role}") or model.endswith(f"_{role}"):
            return role
    return None


def _compact_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            compact_content = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    text = str(item.get("text", ""))
                    compact_content.append({"type": "text", "text": text[:8000], "truncated": len(text) > 8000})
                elif item.get("type") == "image_url":
                    compact_content.append(
                        {"type": "image_url", "image_url": _compact_image_url(item.get("image_url"))}
                    )
                else:
                    compact_content.append({"type": str(item.get("type"))})
            compact.append({"role": str(message.get("role", "user")), "content": compact_content})
        else:
            text = str(content if content is not None else "")
            compact.append(
                {"role": str(message.get("role", "user")), "content": text[:8000], "truncated": len(text) > 8000}
            )
    return compact


def _extract_image_refs(messages: list[dict[str, Any]]) -> list[str]:
    refs = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image_url":
                continue
            url = _raw_image_url(item.get("image_url"))
            if url:
                refs.append(url)
    return refs


def _raw_image_url(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("url") or "")
    return ""


def _compact_image_url(value: object) -> dict[str, str]:
    url = ""
    if isinstance(value, dict):
        url = str(value.get("url") or "")
    if url.startswith("data:"):
        return {"url": f"data_url:{len(url)}"}
    return {"url": url[:1024]}
