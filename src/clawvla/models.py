from __future__ import annotations

import base64
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any

from .config import ModelBackend, ModelConfig


class ModelRuntime:
    """Thin runtime wrapper for local HF and OpenAI-compatible vision/text models."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.processor = None
        self.model = None
        self.client = None

    @property
    def enabled(self) -> bool:
        return self.config.backend != ModelBackend.NONE and bool(self.config.model)

    def validate_config(self, lightweight: bool = True) -> dict[str, Any]:
        report = {
            "backend": self.config.backend,
            "model": self.config.model,
            "enabled": self.enabled,
            "api_base_url": self.config.api_base_url,
            "api_base_url_env": self.config.api_base_url_env,
            "api_key_configured": bool(self.config.api_key),
            "api_key_env": self.config.api_key_env,
            "request_timeout": self.config.request_timeout,
            "reasoning_effort": self.config.reasoning_effort,
        }
        if lightweight or not self.enabled:
            return report
        if self.config.backend in {ModelBackend.OPENAI_COMPATIBLE, ModelBackend.AZURE_OPENAI}:
            self._get_openai_client()
            return report
        if self.config.backend == ModelBackend.LOCAL_HF:
            processor = self._load_processor()
            report["processor_class"] = type(processor).__name__
            return report
        return report

    def generate_text(
        self,
        messages: list[dict[str, Any]],
        image_paths: list[str] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        if not self.enabled:
            raise RuntimeError("ModelRuntime is disabled for this component.")
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        if self.config.backend == ModelBackend.LOCAL_HF:
            return self._generate_text_local(messages, image_paths, max_new_tokens, temperature)
        if self.config.backend in {ModelBackend.OPENAI_COMPATIBLE, ModelBackend.AZURE_OPENAI}:
            return self._generate_text_remote(messages, max_new_tokens, temperature)
        raise ValueError(f"Unsupported model backend: {self.config.backend}")

    def _load_processor(self):
        if self.processor is None:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(self.config.model, trust_remote_code=True)
        return self.processor

    def _load_model(self):
        if self.model is None:
            import torch
            from transformers import AutoModelForImageTextToText

            dtype = getattr(torch, self.config.torch_dtype)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.config.model,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map=self.config.device_map,
            )
        return self.model

    def _generate_text_local(
        self,
        messages: list[dict[str, Any]],
        image_paths: list[str] | None,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        from PIL import Image

        processor = self._load_processor()
        model = self._load_model()
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = [Image.open(Path(path)).convert("RGB") for path in image_paths or []]
        inputs = processor(text=[chat_text], images=images or None, return_tensors="pt")
        model_inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        generated = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return processor.batch_decode(generated[:, model_inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]

    def _get_openai_client(self):
        if self.client is not None:
            return self.client
        api_key = self._resolve_api_key()
        api_base_url = self._resolve_api_base_url()
        if not api_base_url:
            raise ValueError("api_base_url is required for remote model backends.")
        if self.config.backend == ModelBackend.AZURE_OPENAI:
            from openai import AzureOpenAI

            if not self.config.api_version:
                raise ValueError("api_version is required for azure_openai.")
            self.client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=api_base_url,
                api_version=self.config.api_version,
                timeout=self.config.request_timeout,
            )
        else:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=api_key,
                base_url=api_base_url,
                timeout=self.config.request_timeout,
            )
        return self.client

    def _resolve_api_base_url(self) -> str | None:
        if self.config.api_base_url:
            return self.config.api_base_url
        if self.config.api_base_url_env:
            return os.environ.get(self.config.api_base_url_env)
        return None

    def _resolve_api_key(self) -> str:
        if self.config.api_key:
            return self.config.api_key
        if not self.config.api_key_env:
            raise ValueError("api_key or api_key_env is required for remote model backends.")
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {self.config.api_key_env} is not set.")
        return api_key

    def _generate_text_remote(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        client = self._get_openai_client()
        request_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": self._to_openai_messages(messages),
        }
        if self.config.backend == ModelBackend.AZURE_OPENAI:
            request_kwargs["max_completion_tokens"] = max_new_tokens
        else:
            request_kwargs["max_tokens"] = max_new_tokens
        if self._should_send_temperature(temperature):
            request_kwargs["temperature"] = temperature
        if self.config.reasoning_effort:
            request_kwargs["reasoning_effort"] = self.config.reasoning_effort
        if self.config.enable_thinking is not None:
            request_kwargs["enable_thinking"] = self.config.enable_thinking
        if self.config.stream:
            request_kwargs["stream"] = True
        response = client.chat.completions.create(**request_kwargs)
        if self.config.stream:
            return self._coerce_stream(response)
        return self._coerce_content(response.choices[0].message.content)

    def _should_send_temperature(self, temperature: float | None) -> bool:
        if temperature is None:
            return False
        model_name = str(self.config.model or "").lower()
        if self.config.backend == ModelBackend.AZURE_OPENAI and (model_name.startswith("gpt-5") or model_name.startswith("o")):
            return False
        return True

    def _to_openai_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                converted_content: list[dict[str, Any]] = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    if item_type == "text":
                        converted_content.append({"type": "text", "text": str(item.get("text", ""))})
                    elif item_type == "image":
                        image_path = item.get("image")
                        if image_path:
                            converted_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": self._image_path_to_data_url(str(image_path))},
                                }
                            )
                converted.append({"role": str(message.get("role", "user")), "content": converted_content})
            else:
                converted.append(
                    {
                        "role": str(message.get("role", "user")),
                        "content": str(content if content is not None else ""),
                    }
                )
        return converted

    @staticmethod
    def _coerce_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                item_type = getattr(item, "type", None)
                if item_type == "text":
                    parts.append(str(getattr(item, "text", "")))
            return "".join(parts)
        return str(content)

    @staticmethod
    def _coerce_stream(response: Any) -> str:
        parts: list[str] = []
        for chunk in response:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue
            content = ModelRuntime._stream_delta_text(delta)
            if content:
                parts.append(content)
                print(content, end="", flush=True)
        if parts:
            print(file=sys.stderr, flush=True)
        return "".join(parts)

    @staticmethod
    def _stream_delta_text(delta: Any) -> str:
        chunks: list[str] = []
        for field_name in ("reasoning_content", "content"):
            value = getattr(delta, field_name, None)
            if value:
                chunks.append(str(value))
        if not chunks and isinstance(delta, dict):
            for field_name in ("reasoning_content", "content"):
                value = delta.get(field_name)
                if value:
                    chunks.append(str(value))
        return "".join(chunks)

    @staticmethod
    def _image_path_to_data_url(image_path: str) -> str:
        path = Path(image_path)
        mime_type, _ = mimetypes.guess_type(path.name)
        if mime_type is None:
            mime_type = "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
