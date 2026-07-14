"""Backend entry point for the pinned GR00T-N1.6 PolicyClient."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import importlib
from pathlib import Path
from typing import Any, Protocol, cast

from evoworld.adapters.content import verify_manifest
from evoworld.adapters.process import BridgeRequest
from evoworld.contracts import JsonObject, JsonValue
from integrations.common.jsonl_bridge import serve


class ModalityConfig(Protocol):
    modality_keys: Sequence[str]
    delta_indices: Sequence[int]


class PolicyClient(Protocol):
    def ping(self) -> bool: ...

    def get_modality_config(self) -> Mapping[str, ModalityConfig]: ...

    def get_action(
        self, observation: Mapping[str, object], options: Mapping[str, object] | None = None
    ) -> tuple[Mapping[str, object], Mapping[str, object]]: ...

    def call_endpoint(
        self,
        endpoint: str,
        data: dict[str, object] | None = None,
        requires_input: bool = True,
    ) -> object: ...


ClientFactory = Callable[[str, int], PolicyClient]


def _official_client(host: str, port: int) -> PolicyClient:
    module = importlib.import_module("gr00t.policy.server_client")
    factory = getattr(module, "PolicyClient")
    return cast(PolicyClient, factory(host=host, port=port, timeout_ms=15000, strict=False))


def _string(value: JsonValue | None, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a nonempty string")
    return value


def _port(value: JsonValue | None) -> int:
    if type(value) is not int or not 1 <= value <= 65535:
        raise ValueError("port must be between 1 and 65535")
    return value


def _object(value: JsonValue | None, field_name: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _configured_root(payload: Mapping[str, JsonValue], allowed_media_root: Path) -> Path:
    root = allowed_media_root.resolve(strict=True)
    supplied = Path(_string(payload.get("allowed_media_root"), "allowed_media_root"))
    if supplied.resolve(strict=True) != root:
        raise ValueError("allowed_media_root does not match the backend configuration")
    return root


def _contained_file(value: JsonValue, root: Path, field_name: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{field_name} must be an absolute media path")
    try:
        path = Path(value).resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"{field_name} media file is missing") from error
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{field_name} must be contained by allowed_media_root") from error
    if not path.is_file():
        raise ValueError(f"{field_name} must resolve to a media file")
    return path


def _keys(config: ModalityConfig, modality: str) -> tuple[str, ...]:
    keys = tuple(config.modality_keys)
    if not keys or any(not isinstance(key, str) or not key for key in keys):
        raise ValueError(f"{modality} modality_keys must contain nonempty strings")
    return keys


def _horizon(config: ModalityConfig, modality: str) -> int:
    indices = tuple(config.delta_indices)
    if not indices:
        raise ValueError(f"{modality} delta_indices must not be empty")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise ValueError(f"{modality} delta_indices must contain integers")
    return len(indices)


def _modality_configs(value: Mapping[str, ModalityConfig]) -> Mapping[str, ModalityConfig]:
    expected = {"video", "state", "action", "language"}
    if set(value) != expected:
        raise ValueError("modality config must contain exactly video, state, action, and language")
    for name in expected:
        _keys(value[name], name)
        _horizon(value[name], name)
    return value


def _load_rgb(path: Path, numpy: Any, image_module: Any) -> Any:
    with image_module.open(path) as image:
        return numpy.asarray(image.convert("RGB"), dtype=numpy.uint8).copy()


def map_nested_observation(
    value: JsonObject,
    modality_configs: Mapping[str, ModalityConfig],
    allowed_media_root: Path,
) -> Mapping[str, object]:
    """Decode JSON paths and construct the exact pinned nested model observation."""
    if set(value) != {"video", "state", "language"}:
        raise ValueError("observation must contain exactly video, state, and language")
    video_json = _object(value["video"], "video")
    state_json = _object(value["state"], "state")
    instruction = _string(value["language"], "language")
    video_keys = _keys(modality_configs["video"], "video")
    state_keys = _keys(modality_configs["state"], "state")
    language_keys = _keys(modality_configs["language"], "language")
    if set(video_json) != set(video_keys):
        raise ValueError("video keys do not match the service modality config")
    if set(state_json) != set(state_keys):
        raise ValueError("state keys do not match the service modality config")
    numpy = importlib.import_module("numpy")
    image_module = importlib.import_module("PIL.Image")
    videos: dict[str, object] = {}
    for key in video_keys:
        raw_paths = video_json[key]
        if not isinstance(raw_paths, list) or len(raw_paths) != _horizon(
            modality_configs["video"], "video"
        ):
            raise ValueError(f"video.{key} horizon does not match modality config")
        frames = [
            _load_rgb(
                _contained_file(path, allowed_media_root, f"video.{key}"), numpy, image_module
            )
            for path in raw_paths
        ]
        try:
            videos[key] = numpy.stack(frames, axis=0)[None, ...]
        except ValueError as error:
            raise ValueError(f"video.{key} frames must have equal dimensions") from error
    states: dict[str, object] = {}
    for key in state_keys:
        raw_state = state_json[key]
        if not isinstance(raw_state, list) or len(raw_state) != _horizon(
            modality_configs["state"], "state"
        ):
            raise ValueError(f"state.{key} horizon does not match modality config")
        array = numpy.asarray(raw_state, dtype=numpy.float32)
        if array.ndim != 2 or array.shape[1] == 0:
            raise ValueError(f"state.{key} must have shape (T, D)")
        if not bool(numpy.isfinite(array).all()):
            raise ValueError(f"state.{key} must contain finite values")
        states[key] = array[None, ...]
    if _horizon(modality_configs["language"], "language") != 1:
        raise ValueError("pinned PolicyClient requires a language horizon of one")
    languages = {key: [[instruction]] for key in language_keys}
    return {"video": videos, "state": states, "language": languages}


def serialize_groot_actions(
    action: Mapping[str, object], modality_configs: Mapping[str, ModalityConfig]
) -> JsonObject:
    """Validate and serialize exact float32 `(B,T,D)` action modalities."""
    numpy = importlib.import_module("numpy")
    action_keys = _keys(modality_configs["action"], "action")
    delta_indices = tuple(modality_configs["action"].delta_indices)
    if set(action) != set(action_keys):
        raise ValueError("action keys do not match the service modality config")
    arrays: list[Any] = []
    batch_size: int | None = None
    for key in action_keys:
        array = action[key]
        if not isinstance(array, numpy.ndarray) or array.dtype != numpy.float32:
            raise ValueError(f"action.{key} must be a float32 numpy array")
        if array.ndim != 3:
            raise ValueError(f"action.{key} must have shape (B, T, D)")
        if any(dimension <= 0 for dimension in array.shape):
            raise ValueError(f"action.{key} must have nonempty B, T, and D dimensions")
        if array.shape[1] != len(delta_indices):
            raise ValueError(f"action.{key} horizon does not match modality config")
        if batch_size is None:
            batch_size = int(array.shape[0])
        if array.shape[0] != batch_size:
            raise ValueError("action keys must have the same batch size")
        if not bool(numpy.isfinite(array).all()):
            raise ValueError(f"action.{key} must contain finite values")
        arrays.append(array)
    combined = numpy.concatenate(arrays, axis=2)
    return {
        "dtype": "float32",
        "actions": cast(list[JsonValue], combined.tolist()),
        "delta_indices": list(delta_indices),
        "action_keys": list(action_keys),
    }


def handle_request(
    request: BridgeRequest,
    *,
    allowed_media_root: Path,
    client_factory: ClientFactory = _official_client,
) -> JsonObject:
    """Call only the three official PolicyClient operations required by EvoWorld."""
    if request.operation != "act":
        raise ValueError("GR00T bridge supports only act")
    if set(request.payload) != {
        "host",
        "port",
        "checkpoint_identity",
        "checkpoint_digest",
        "allowed_media_root",
        "media_manifest",
        "observation",
    }:
        raise ValueError("GR00T payload fields do not match the service contract")
    root = _configured_root(request.payload, allowed_media_root)
    checkpoint_identity = _string(request.payload["checkpoint_identity"], "checkpoint_identity")
    checkpoint_digest = _string(request.payload["checkpoint_digest"], "checkpoint_digest")
    media_manifest = _object(request.payload["media_manifest"], "media_manifest")
    verify_manifest(media_manifest, "GR00T input media")
    client = client_factory(
        _string(request.payload["host"], "host"), _port(request.payload["port"])
    )
    if not client.ping():
        raise RuntimeError("GR00T policy service ping failed")
    identity = _object(
        cast(JsonValue, client.call_endpoint("evoworld_identity", requires_input=False)),
        "evoworld_identity",
    )
    if set(identity) != {"checkpoint_identity", "checkpoint_digest"} or (
        identity.get("checkpoint_identity") != checkpoint_identity
        or identity.get("checkpoint_digest") != checkpoint_digest
    ):
        raise ValueError("GR00T service checkpoint identity does not match")
    configs = _modality_configs(client.get_modality_config())
    observation = map_nested_observation(
        _object(request.payload["observation"], "observation"), configs, root
    )
    action, _info = client.get_action(observation)
    result = serialize_groot_actions(action, configs)
    verify_manifest(media_manifest, "GR00T input media")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allowed-media-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.allowed_media_root.resolve(strict=True)
    serve(lambda request: handle_request(request, allowed_media_root=root))


if __name__ == "__main__":
    main()
