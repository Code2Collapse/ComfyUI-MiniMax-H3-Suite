"""Streaming H3 checkpoint readers — safetensors (quant) and GGUF, one tensor at a time."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Protocol

import torch

from mmx_utils.h3_hybrid import normalize_diffusion_key

_GGUF_DEQUANT = None
_GGUF_GET_ORIG_SHAPE = None
_GGUF_IMPORT_ERROR: str | None = None


def _load_gguf_helpers():
    global _GGUF_DEQUANT, _GGUF_GET_ORIG_SHAPE, _GGUF_IMPORT_ERROR
    if _GGUF_DEQUANT is not None or _GGUF_IMPORT_ERROR is not None:
        return _GGUF_DEQUANT, _GGUF_GET_ORIG_SHAPE
    pack_root = Path(__file__).resolve().parents[1]
    for candidate in (
        pack_root.parent.parent / "ComfyUI_windows_portable" / "ComfyUI" / "custom_nodes" / "ComfyUI-GGUF",
        pack_root.parent / "third_party" / "ComfyUI-GGUF",
    ):
        p = str(candidate)
        if p not in sys.path and candidate.is_dir():
            sys.path.insert(0, p)
    try:
        from dequant import dequantize as _dq  # type: ignore[import-not-found]
        from loader import get_orig_shape as _gos  # type: ignore[import-not-found]

        _GGUF_DEQUANT = _dq
        _GGUF_GET_ORIG_SHAPE = _gos
    except Exception as exc:  # noqa: BLE001
        _GGUF_IMPORT_ERROR = str(exc)
        _GGUF_DEQUANT = None
        _GGUF_GET_ORIG_SHAPE = None
    return _GGUF_DEQUANT, _GGUF_GET_ORIG_SHAPE


def _int8_per_row_noise_energy(shape: tuple[int, ...], scale: torch.Tensor) -> float:
    """Expected squared Frobenius quant error: sum_rows in_features * scale_row^2 / 12."""
    out_f, in_f = int(shape[0]), int(shape[1])
    scale_f = scale.float().reshape(-1)
    if scale_f.numel() == 1:
        return float(in_f * out_f * scale_f.item() ** 2 / 12.0)
    return float((in_f * scale_f.pow(2)).sum().item() / 12.0)


def _float_storage_noise_energy(weight: torch.Tensor) -> float | None:
    w_norm_sq = float(weight.float().pow(2).sum().item())
    if weight.dtype == torch.bfloat16:
        return (2.0**-16) * w_norm_sq / 3.0
    if weight.dtype == torch.float16:
        return (2.0**-22) * w_norm_sq / 3.0
    return None


def _noise_energy_from_safetensors(handle, norm_key: str, raw_key: str, shape: tuple[int, ...]) -> float | None:
    if not norm_key.endswith(".weight") or len(shape) != 2:
        return None
    prefix = raw_key[: -len("weight")]
    quant_raw = prefix + "comfy_quant"
    if quant_raw not in handle.keys():
        weight = handle.get_tensor(raw_key)
        return _float_storage_noise_energy(weight)

    blob = handle.get_tensor(quant_raw)
    layer_conf = json.loads(bytes(blob.numpy().tobytes()).decode("utf-8"))
    quant_format = layer_conf.get("format")
    if quant_format == "int8_tensorwise":
        scale_key = prefix + "weight_scale"
        if scale_key not in handle.keys():
            return None
        scale = handle.get_tensor(scale_key)
        return _int8_per_row_noise_energy(shape, scale)
    return None


class CheckpointReader(Protocol):
    def tensor_keys(self) -> set[str]: ...
    def shape(self, key: str) -> tuple[int, ...]: ...
    def read_tensor(self, key: str) -> torch.Tensor: ...
    def noise_energy(self, key: str) -> float | None: ...
    def close(self) -> None: ...


def open_checkpoint(path: str | Path, *, device: str = "cpu") -> CheckpointReader:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        return SafetensorsCheckpointReader(path, device=device)
    if suffix == ".gguf":
        return GGUFCheckpointReader(path, device=device)
    raise ValueError(f"Cannot open checkpoint {path}: unsupported extension {suffix!r}")


class SafetensorsCheckpointReader:
    def __init__(self, path: Path, *, device: str = "cpu") -> None:
        from safetensors import safe_open

        self._path = path
        self._device = device
        self._handle = safe_open(str(path), framework="pt", device=device)
        raw_keys = set(self._handle.keys())
        self._raw_by_norm: dict[str, str] = {}
        self._norm_keys: set[str] = set()
        for raw in raw_keys:
            norm = normalize_diffusion_key(raw)
            self._raw_by_norm[norm] = raw
            self._norm_keys.add(norm)
        self._shapes: dict[str, tuple[int, ...]] = {}
        for norm, raw in self._raw_by_norm.items():
            self._shapes[norm] = tuple(self._handle.get_slice(raw).get_shape())

    def tensor_keys(self) -> set[str]:
        return set(self._norm_keys)

    def shape(self, key: str) -> tuple[int, ...]:
        norm = normalize_diffusion_key(key)
        if norm not in self._shapes:
            raise KeyError(f"Tensor {key!r} not in {self._path.name}")
        return self._shapes[norm]

    def read_tensor(self, key: str) -> torch.Tensor:
        norm = normalize_diffusion_key(key)
        raw = self._raw_by_norm.get(norm)
        if raw is None:
            raise KeyError(f"Tensor {key!r} not in {self._path.name}")
        if not norm.endswith(".weight"):
            t = self._handle.get_tensor(raw)
            return t.float()
        return _decode_weight_tensor(self._handle, self._path, norm, raw, self._device)

    def noise_energy(self, key: str) -> float | None:
        norm = normalize_diffusion_key(key)
        raw = self._raw_by_norm.get(norm)
        if raw is None or norm not in self._shapes:
            return None
        return _noise_energy_from_safetensors(self._handle, norm, raw, self._shapes[norm])

    def close(self) -> None:
        self._handle = None


class GGUFCheckpointReader:
    def __init__(self, path: Path, *, device: str = "cpu") -> None:
        import gguf

        self._path = path
        self._device = device
        self._reader = gguf.GGUFReader(str(path))
        self._dequantize, self._get_orig_shape = _load_gguf_helpers()

        prefixes = ("model.diffusion_model.", "diffusion_model.", "")
        self._tensor_map: dict[str, object] = {}
        names = [t.name for t in self._reader.tensors]
        chosen = ""
        for prefix in prefixes:
            if any(n.startswith(prefix) for n in names):
                chosen = prefix
                break
        for tensor in self._reader.tensors:
            name = tensor.name
            if chosen and not name.startswith(chosen):
                continue
            rest = name[len(chosen) :] if chosen else name
            norm = normalize_diffusion_key(rest)
            self._tensor_map[norm] = tensor

        self._shapes: dict[str, tuple[int, ...]] = {}
        for norm, tensor in self._tensor_map.items():
            shape = getattr(tensor, "tensor_shape", None) or tuple(tensor.shape)
            self._shapes[norm] = tuple(int(x) for x in shape)

    def tensor_keys(self) -> set[str]:
        return set(self._tensor_map.keys())

    def shape(self, key: str) -> tuple[int, ...]:
        norm = normalize_diffusion_key(key)
        if norm not in self._shapes:
            raise KeyError(f"Tensor {key!r} not in {self._path.name}")
        return self._shapes[norm]

    def read_tensor(self, key: str) -> torch.Tensor:
        import gguf

        norm = normalize_diffusion_key(key)
        tensor = self._tensor_map.get(norm)
        if tensor is None:
            raise KeyError(f"Tensor {key!r} not in {self._path.name}")

        tensor_name = tensor.name
        data = torch.from_numpy(tensor.data.copy())
        if self._get_orig_shape is not None:
            oshape = self._get_orig_shape(self._reader, tensor_name)
        else:
            oshape = None
        if oshape is None:
            oshape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))

        qtype = getattr(tensor, "tensor_type", None)

        if qtype in (None, gguf.GGMLQuantizationType.F32):
            return data.reshape(oshape).to(device=self._device, dtype=torch.float32)
        if qtype == gguf.GGMLQuantizationType.F16:
            return data.view(torch.float16).reshape(oshape).float().to(device=self._device)
        if hasattr(gguf.GGMLQuantizationType, "BF16") and qtype == gguf.GGMLQuantizationType.BF16:
            return data.view(torch.bfloat16).reshape(oshape).float().to(device=self._device)

        if self._dequantize is not None:
            try:
                out = self._dequantize(data, qtype, oshape, dtype=torch.float32)
                return out.to(device=self._device, dtype=torch.float32)
            except Exception:
                pass

        try:
            arr = gguf.quants.dequantize(tensor.data, qtype)
            out = torch.from_numpy(arr).reshape(oshape)
            return out.to(device=self._device, dtype=torch.float32)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"Cannot decode {self._path.name}: {norm} uses GGUF format "
                f"{getattr(qtype, 'name', qtype)!r}: {exc}"
            ) from exc

    def noise_energy(self, key: str) -> float | None:
        return None

    def close(self) -> None:
        self._reader = None


def _decode_weight_tensor(handle, path: Path, norm_key: str, raw_key: str, device: str) -> torch.Tensor:
    prefix = raw_key[: -len("weight")]
    quant_raw = prefix + "comfy_quant"
    norm_quant = normalize_diffusion_key(quant_raw)
    quant_key = None
    for candidate in (quant_raw, norm_quant):
        if candidate in handle.keys():
            quant_key = candidate
            break

    weight = handle.get_tensor(raw_key)

    if quant_key is None:
        return weight.float()

    blob = handle.get_tensor(quant_key)
    layer_conf = json.loads(bytes(blob.numpy().tobytes()).decode("utf-8"))
    quant_format = layer_conf.get("format")
    if quant_format is None:
        raise ValueError(
            f"Cannot decode {path.name}: {norm_key} has comfy_quant without format field"
        )

    try:
        from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor, get_layout_class
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Cannot decode {path.name}: {norm_key} is quantized ({quant_format}) but "
            f"comfy.quant_ops is unavailable: {exc}"
        ) from exc

    if quant_format not in QUANT_ALGOS:
        raise ValueError(
            f"Cannot decode {path.name}: {norm_key} uses unsupported format {quant_format!r}"
        )

    qconfig = QUANT_ALGOS[quant_format]
    layout_cls = get_layout_class(qconfig["comfy_tensor_layout"])
    orig_shape = tuple(weight.shape)
    scales = _read_quant_scales(handle, prefix, quant_format, layer_conf, norm_key, path)

    params = layout_cls.Params(**scales, orig_dtype=torch.float32, orig_shape=orig_shape)
    qtensor = QuantizedTensor(
        weight.to(device=device, dtype=qconfig["storage_t"]),
        qconfig["comfy_tensor_layout"],
        params,
    )
    return qtensor.dequantize().float()


def _read_quant_scales(handle, prefix: str, quant_format: str, layer_conf: dict, norm_key: str, path: Path):
    def pop_scale(name: str, dtype=None):
        key = prefix + name
        if key not in handle.keys():
            return None
        v = handle.get_tensor(key)
        if dtype is not None:
            v = v.view(dtype=dtype)
        return v

    params_conf = layer_conf.get("params", {})
    if not isinstance(params_conf, dict):
        params_conf = {}

    if quant_format in ("float8_e4m3fn", "float8_e5m2"):
        scale = pop_scale("weight_scale")
        if scale is None:
            raise ValueError(f"Cannot decode {path.name}: {norm_key} missing weight_scale")
        return {"scale": scale}

    if quant_format == "mxfp8":
        bs = pop_scale("weight_scale", torch.float8_e8m0fnu)
        if bs is None:
            raise ValueError(f"Cannot decode {path.name}: {norm_key} missing MXFP8 weight_scale")
        return {"scale": bs}

    if quant_format == "nvfp4":
        ts = pop_scale("weight_scale_2")
        bs = pop_scale("weight_scale", torch.float8_e4m3fn)
        if ts is None or bs is None:
            raise ValueError(f"Cannot decode {path.name}: {norm_key} missing NVFP4 scales")
        return {"scale": ts, "block_scale": bs}

    if quant_format == "int8_tensorwise":
        scale = pop_scale("weight_scale")
        if scale is None:
            raise ValueError(f"Cannot decode {path.name}: {norm_key} missing INT8 weight_scale")
        scales = {"scale": scale}
        if layer_conf.get("convrot", params_conf.get("convrot", False)):
            scales["convrot"] = True
            scales["convrot_groupsize"] = int(
                layer_conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256))
            )
        return scales

    if quant_format == "convrot_w4a4":
        scale = pop_scale("weight_scale")
        if scale is None:
            raise ValueError(f"Cannot decode {path.name}: {norm_key} missing ConvRot W4A4 weight_scale")
        return {
            "scale": scale,
            "convrot_groupsize": int(
                layer_conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256))
            ),
            "quant_group_size": 64,
            "linear_dtype": layer_conf.get("linear_dtype", params_conf.get("linear_dtype", "int4")),
        }

    if quant_format == "asym_w4a8_int8":
        scale = pop_scale("weight_s_rel")
        if scale is None:
            raise ValueError(f"Cannot decode {path.name}: {norm_key} missing W4A8 weight_s_rel")
        if scale.dtype == torch.uint8:
            scale = scale.view(torch.float8_e4m3fn)
        return {
            "scale": scale,
            "s_channel": pop_scale("weight_s_channel"),
            "codebook": pop_scale("weight_codebook"),
            "group_size": int(layer_conf.get("group_size", params_conf.get("group_size", 16))),
            "convrot_groupsize": int(
                layer_conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256))
            ),
        }

    raise ValueError(f"Cannot decode {path.name}: {norm_key} uses unsupported format {quant_format!r}")
