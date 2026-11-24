"""Pinned hf-mem v1 adapter and instance catalog helpers."""

import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypedDict

from hf_gcp.errors import GcpError, redact

# Pin the published release: the isolated worker depends on this exact API.
HF_MEM_SOURCE = "hf-mem==1.0.0"
HF_MEM_VERSION = "hf-mem@1.0.0"


class HfMemError(GcpError):
    """Error running hf-mem estimation."""


@dataclass(frozen=True)
class EstimateOptions:
    """Workload and weight selection passed unchanged to hf-mem."""

    revision: str | None = None
    file: str | None = None
    format: str | None = None
    max_model_len: int | None = None
    batch_size: int = 1
    kv_cache_dtype: str = "auto"
    max_num_batched_tokens: int | None = None

    def validate(self) -> None:
        for name in ("max_model_len", "batch_size", "max_num_batched_tokens"):
            value = getattr(self, name)
            if (name == "batch_size" or value is not None) and (type(value) is not int or value < 1):
                raise HfMemError(f"--{name.replace('_', '-')} must be a positive integer")
        if self.format not in (None, "safetensors", "gguf", "onnx"):
            raise HfMemError("--format must be safetensors, gguf, or onnx")
        if not self.kv_cache_dtype:
            raise HfMemError("--kv-cache-dtype cannot be empty")


@dataclass(frozen=True)
class GpuFit:
    """Upstream's fit assessment for a specific GPU layout."""

    fits: bool
    required_bytes: int
    gpu_count: int
    memory_per_gpu_bytes: int
    needed_gpu_count: int | None
    max_model_len: int


@dataclass(frozen=True)
class HfMemEstimate:
    """Memory fit and warnings for one GPU layout."""

    fit: GpuFit
    warnings: tuple[str, ...]


class InstanceInfo(TypedDict, total=False):
    """Instance type information."""

    instanceType: str
    numGpu: int
    gpuMemGbPerGpu: float
    startupTimeoutSeconds: int
    pricePerHourUsd: float | None
    acceleratorType: str


@dataclass
class InstanceData:
    instances: list[InstanceInfo]
    safety_margin_default: float
    last_updated: str
    pricing_region: str | None = None


def _run_command(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout_s: int = 120,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(cmd, env=merged_env, input=input_text, text=True, capture_output=True, timeout=timeout_s)


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise HfMemError(f"Invalid hf-mem {name}: expected a non-negative integer")
    return value


def _parse_estimate(payload: Any) -> HfMemEstimate:
    if not isinstance(payload, dict):
        raise HfMemError("Invalid memory output from hf-mem: expected a JSON object")
    if payload.get("version") != HF_MEM_VERSION:
        raise HfMemError("Unexpected hf-mem version", details=f"Expected {HF_MEM_VERSION}")
    warnings = payload.get("warnings", [])
    if not isinstance(warnings, list) or not all(isinstance(warning, str) for warning in warnings):
        raise HfMemError("Invalid hf-mem warnings: expected a list of strings")
    if "files" in payload:
        files = payload["files"]
        names = [item.get("file") for item in files if isinstance(item, dict)] if isinstance(files, list) else []
        raise HfMemError(
            "Multiple weight files found; select one with --file or a GGUF quantization tag",
            details="Available files: " + ", ".join(str(name) for name in names if name),
        )
    raw_fit = payload.get("fit")
    if not isinstance(raw_fit, dict) or type(raw_fit.get("fits")) is not bool:
        raise HfMemError("hf-mem did not return a valid GPU fit")
    count = raw_fit.get("gpus")
    count = 1 if count is None else _nonnegative_int(count, "GPU count")
    memory = _nonnegative_int(raw_fit.get("gpu_memory"), "GPU memory bytes")
    needed = raw_fit.get("needed")
    if needed is not None:
        needed = _nonnegative_int(needed, "needed GPU count")
    if count < 1 or memory < 1 or needed == 0:
        raise HfMemError("Invalid hf-mem GPU fit counts or memory")
    fit = GpuFit(
        fits=raw_fit["fits"],
        required_bytes=_nonnegative_int(raw_fit.get("required"), "required GPU bytes"),
        gpu_count=count,
        memory_per_gpu_bytes=memory,
        needed_gpu_count=needed,
        max_model_len=_nonnegative_int(raw_fit.get("max_model_len"), "maximum context length"),
    )
    return HfMemEstimate(fit, tuple(warnings))


def run_hf_mem_batch(
    model_id: str,
    layouts: list[tuple[int, float]],
    *,
    options: EstimateOptions,
    gpu_memory_utilization: float,
    timeout: int = 120,
    token: str | None = None,
) -> list[HfMemEstimate]:
    """Evaluate ordered GPU layouts in one process with shared model metadata."""
    options.validate()
    if not math.isfinite(gpu_memory_utilization) or not 0 < gpu_memory_utilization <= 1:
        raise HfMemError("GPU memory utilization must be finite and in (0, 1]")
    if not layouts:
        return []
    request = {
        "version": HF_MEM_VERSION,
        "model": model_id,
        "layouts": layouts,
        "options": asdict(options),
        "utilization": gpu_memory_utilization,
    }
    try:
        result = _run_command(
            [
                "uvx",
                "--python",
                ">=3.10",
                "--from",
                HF_MEM_SOURCE,
                "python",
                "-I",  # Avoid sibling utility modules shadowing Python standard libraries.
                str(Path(__file__).with_name("_hf_mem_worker.py")),
            ],
            input_text=json.dumps(request),
            timeout_s=timeout,
            **({"env": {"HF_TOKEN": token}} if token is not None else {}),
        )
        if result.returncode != 0:
            raise HfMemError(f"hf-mem failed for model {model_id}", details=redact(result.stderr, token) or None)
        payload = json.loads(result.stdout)
        if not isinstance(payload, list):
            raise HfMemError("Invalid hf-mem batch result")
        estimates = [_parse_estimate(item) for item in payload]
        if len(estimates) != len(layouts):
            raise HfMemError("Incomplete hf-mem batch result")
        for (count, memory), estimate in zip(layouts, estimates, strict=True):
            if estimate.fit.gpu_count != count or estimate.fit.memory_per_gpu_bytes != int(memory * 1024**3):
                raise HfMemError("hf-mem GPU fit does not match the requested GPU layout")
        return estimates
    except subprocess.TimeoutExpired:
        raise HfMemError(
            f"hf-mem timed out for model {model_id}", details=f"Command exceeded {timeout}s timeout"
        ) from None
    except json.JSONDecodeError as error:
        raise HfMemError("Invalid JSON output from hf-mem batch", details=str(error)) from error
    except FileNotFoundError:
        raise HfMemError("hf-mem not found", details="Install uv to provide uvx: pip install uv") from None
