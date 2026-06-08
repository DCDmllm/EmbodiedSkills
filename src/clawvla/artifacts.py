from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .notices import emit_status_notice


class ArtifactStore:
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, relative_path: str, payload: Any) -> str:
        path = self.root_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_jsonable(payload), ensure_ascii=True, indent=2), encoding="utf-8")
        return str(path)

    def write_image(self, relative_path: str, image: Any) -> str:
        path = self.root_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        pil_image = _to_pil_image(image)
        pil_image.save(path)
        return str(path)

    def write_depth(self, relative_path: str, depth: Any) -> str:
        path = self.root_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import numpy as np

            np.save(path, np.asarray(depth))
            return str(path if str(path).endswith(".npy") else Path(f"{path}.npy"))
        except Exception as exc:
            emit_status_notice(
                "artifact_depth_numpy_write_unavailable",
                success=True,
                source="artifacts.write_depth",
                reason=f"{type(exc).__name__}: {exc}",
                payload={"relative_path": relative_path, "fallback": "json_unserializable_depth"},
            )
            return self.write_json(f"{relative_path}.json", {"unserializable_depth": str(type(depth).__name__)})

    def write_pointcloud(self, relative_path: str, pointcloud: Any) -> str:
        path = self.root_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import numpy as np

            if isinstance(pointcloud, dict):
                array_payload = {str(key): np.asarray(value) for key, value in pointcloud.items()}
                final_path = path if str(path).endswith(".npz") else Path(f"{path}.npz")
                np.savez_compressed(final_path, **array_payload)
                return str(final_path)
            if isinstance(pointcloud, list):
                final_path = path if str(path).endswith(".npz") else Path(f"{path}.npz")
                np.savez_compressed(final_path, **{f"item_{index}": np.asarray(value) for index, value in enumerate(pointcloud)})
                return str(final_path)
            np.save(path, np.asarray(pointcloud))
            return str(path if str(path).endswith(".npy") else Path(f"{path}.npy"))
        except Exception as exc:
            emit_status_notice(
                "artifact_pointcloud_numpy_write_unavailable",
                success=True,
                source="artifacts.write_pointcloud",
                reason=f"{type(exc).__name__}: {exc}",
                payload={"relative_path": relative_path, "fallback": "json_unserializable_pointcloud"},
            )
            return self.write_json(f"{relative_path}.json", {"unserializable_pointcloud": str(type(pointcloud).__name__)})

    def write_mask(self, relative_path: str, mask: Any) -> str:
        path = self.root_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import numpy as np

            np.save(path, np.asarray(mask))
            return str(path if str(path).endswith(".npy") else Path(f"{path}.npy"))
        except Exception as exc:
            emit_status_notice(
                "artifact_mask_numpy_write_unavailable",
                success=True,
                source="artifacts.write_mask",
                reason=f"{type(exc).__name__}: {exc}",
                payload={"relative_path": relative_path, "fallback": "json_unserializable_mask"},
            )
            return self.write_json(f"{relative_path}.json", {"unserializable_mask": str(type(mask).__name__)})


def _to_pil_image(image: Any):
    from PIL import Image

    if isinstance(image, Image.Image):
        return image
    try:
        import numpy as np

        array = np.asarray(image)
        if array.dtype != np.uint8:
            if array.max(initial=0) <= 1.0:
                array = array * 255.0
            array = array.clip(0, 255).astype(np.uint8)
        return Image.fromarray(array)
    except Exception as exc:
        raise TypeError(f"Unsupported image payload: {type(image).__name__}") from exc


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
