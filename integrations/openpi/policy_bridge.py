"""Backend entry point that calls the pinned openpi service lazily."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from http.client import HTTPConnection
import importlib
import math
from pathlib import Path
from typing import Protocol, cast

from evoworld.adapters.content import verify_manifest
from evoworld.adapters.process import BridgeRequest
from evoworld.contracts import JsonObject, JsonValue

_DEFAULT_HEALTH_TIMEOUT_SECONDS = 2.0


class PolicyHealthError(RuntimeError):
    """The externally managed policy service did not pass its bounded health check."""


class PolicyClient(Protocol):
    """The small official client surface used after the service becomes healthy."""

    def get_server_metadata(self) -> Mapping[str, object]: ...

    def infer(self, observation: Mapping[str, object]) -> Mapping[str, object]: ...


ClientFactory = Callable[[str, int], PolicyClient]
HealthCheck = Callable[[str, int, float], None]
ImageLoader = Callable[[JsonObject, Path], Mapping[str, object]]


def check_healthz(host: str, port: int, timeout: float) -> None:
    """Require an HTTP /healthz responder before opening the policy websocket."""
    connection = HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request("GET", "/healthz")
        response = connection.getresponse()
        body = response.read()
    except OSError as error:
        raise PolicyHealthError("openpi /healthz is unavailable") from error
    finally:
        connection.close()
    if response.status != 200 or body != b"OK\n":
        raise PolicyHealthError("openpi /healthz must return 200 and OK")


def _official_client(host: str, port: int) -> PolicyClient:
    """Import the pinned official client only inside the backend environment."""
    module = importlib.import_module("openpi_client.websocket_client_policy")
    factory = getattr(module, "WebsocketClientPolicy")
    return cast(PolicyClient, factory(host, port))


def _port(value: JsonValue) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise ValueError("port must be an integer between 1 and 65535")
    return int(value)


def _string(value: JsonValue, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _observation(value: JsonValue) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError("observation must be a JSON object")
    return value


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


def _state(value: JsonValue, width: int, field_name: str) -> list[JsonValue]:
    if not isinstance(value, list) or len(value) != width:
        raise ValueError(f"{field_name} must contain exactly {width} values")
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{field_name} must contain numeric values")
        if not math.isfinite(float(item)):
            raise ValueError(f"{field_name} must contain finite values")
    return value


def _validate_observation(value: JsonObject, root: Path) -> JsonObject:
    fields = {
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
        "observation/joint_position",
        "observation/gripper_position",
        "prompt",
    }
    if set(value) != fields:
        raise ValueError("DROID observation fields do not match the service contract")
    prompt = _string(value["prompt"], "prompt")
    return {
        "observation/exterior_image_1_left": str(
            _contained_file(
                value["observation/exterior_image_1_left"], root, "exterior_image_1_left"
            )
        ),
        "observation/wrist_image_left": str(
            _contained_file(value["observation/wrist_image_left"], root, "wrist_image_left")
        ),
        "observation/joint_position": _state(
            value["observation/joint_position"], 7, "joint_position"
        ),
        "observation/gripper_position": _state(
            value["observation/gripper_position"], 1, "gripper_position"
        ),
        "prompt": prompt,
    }


def load_droid_observation(observation: JsonObject, root: Path) -> Mapping[str, object]:
    """Load contained media and resize it through the pinned openpi client helper."""
    del root
    numpy = importlib.import_module("numpy")
    image_module = importlib.import_module("PIL.Image")
    image_tools = importlib.import_module("openpi_client.image_tools")
    loaded: dict[str, object] = dict(observation)
    for field in (
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
    ):
        path = cast(str, observation[field])
        with image_module.open(path) as image:
            rgb_image = image.convert("RGB")
            array = numpy.asarray(rgb_image).copy()
        resized = image_tools.resize_with_pad(array, 224, 224)
        if resized.shape != (224, 224, 3) or resized.dtype != numpy.uint8:
            raise ValueError("resized DROID images must be uint8 arrays shaped (224, 224, 3)")
        loaded[field] = resized
    return loaded


def _json_actions(value: object) -> list[JsonValue]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    normalized = _json_value(value)
    if not isinstance(normalized, list):
        raise ValueError("openpi client actions must be an array")
    return normalized


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("openpi client actions must be finite JSON values")
        return value
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("openpi client metadata keys must be strings")
            result[key] = _json_value(item)
        return result
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("openpi client actions must be JSON arrays")
    return [_json_value(item) for item in value]


def _timing(value: object, field_name: str) -> None:
    """Validate the exact timing object emitted by the pinned policy/server."""
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} must be a string-keyed timing object")
    fields = set(value)
    valid_fields = (
        ({"infer_ms"},)
        if field_name == "policy_timing"
        else ({"infer_ms"}, {"infer_ms", "prev_total_ms"})
    )
    if fields not in valid_fields:
        raise ValueError(f"{field_name} fields do not match the pinned service")
    for item in value.values():
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{field_name} values must be finite numbers")
        if not math.isfinite(float(item)):
            raise ValueError(f"{field_name} values must be finite numbers")


def handle_request(
    request: BridgeRequest,
    *,
    allowed_media_root: Path,
    client_factory: ClientFactory = _official_client,
    health_check: HealthCheck = check_healthz,
    image_loader: ImageLoader = load_droid_observation,
) -> JsonObject:
    """Validate one process request and return only the official actions response."""
    if request.operation != "act":
        raise ValueError("openpi bridge supports only act")
    if set(request.payload) != {
        "model",
        "host",
        "port",
        "checkpoint_identity",
        "checkpoint_digest",
        "allowed_media_root",
        "media_manifest",
        "observation",
    }:
        raise ValueError("openpi payload fields do not match the service contract")
    _string(request.payload.get("model"), "model")
    checkpoint_identity = _string(request.payload.get("checkpoint_identity"), "checkpoint_identity")
    checkpoint_digest = _string(request.payload.get("checkpoint_digest"), "checkpoint_digest")
    root = allowed_media_root.resolve(strict=True)
    payload_root = _string(request.payload.get("allowed_media_root"), "allowed_media_root")
    if Path(payload_root).resolve(strict=True) != root:
        raise ValueError("allowed_media_root does not match the configured backend root")
    observation = _validate_observation(_observation(request.payload.get("observation")), root)
    media_manifest = _observation(request.payload.get("media_manifest"))
    verify_manifest(media_manifest, "OpenPi input media")
    host = _string(request.payload.get("host", "127.0.0.1"), "host")
    port = _port(request.payload.get("port", 8000))
    health_check(host, port, _DEFAULT_HEALTH_TIMEOUT_SECONDS)
    client = client_factory(host, port)
    metadata = client.get_server_metadata()
    if not isinstance(metadata, Mapping):
        raise ValueError("openpi server metadata must be an object")
    if (
        metadata.get("checkpoint_identity") != checkpoint_identity
        or metadata.get("checkpoint_digest") != checkpoint_digest
    ):
        raise ValueError("openpi service checkpoint identity does not match")
    loaded_observation = image_loader(observation, root)
    response = client.infer(loaded_observation)
    if (
        not isinstance(response, Mapping)
        or "actions" not in response
        or not set(response)
        <= {
            "actions",
            "policy_timing",
            "server_timing",
        }
    ):
        raise ValueError("official client response contains unsupported fields")
    for field_name in ("policy_timing", "server_timing"):
        if field_name in response:
            _timing(response[field_name], field_name)
    actions = _json_actions(response["actions"])
    verify_manifest(media_manifest, "OpenPi input media")
    return {"actions": actions}


def main() -> None:
    """Serve one JSONL process request while retaining protocol stdout discipline."""
    from integrations.common.jsonl_bridge import serve

    parser = argparse.ArgumentParser()
    parser.add_argument("--allowed-media-root", type=Path, required=True)
    args = parser.parse_args()
    allowed_media_root = args.allowed_media_root.resolve(strict=True)
    serve(lambda request: handle_request(request, allowed_media_root=allowed_media_root))


if __name__ == "__main__":
    main()
