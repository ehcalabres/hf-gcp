"""Vertex AI machine estimates and deployment plans."""

import json
import math
import re
from dataclasses import asdict, dataclass
from importlib.resources import files
from uuid import uuid4

from huggingface_hub import HfApi, get_token
from huggingface_hub.utils import validate_repo_id

from hf_gcp.environment import validate_environment
from hf_gcp.errors import GcpError
from hf_gcp.hf_mem import HF_MEM_VERSION, EstimateOptions, InstanceData, InstanceInfo, run_hf_mem_batch

ENDPOINT_NAME = re.compile(r"[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?")
MACHINE_TYPE = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)+")
SECRET_NAME = re.compile(r"TOKEN|PASSWORD|SECRET|CREDENTIAL|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY", re.IGNORECASE)


def load_catalog() -> InstanceData:
    data = json.loads(files("hf_gcp").joinpath("data", "vertex.json").read_text(encoding="utf-8"))
    return InstanceData(
        data["instances"], data["safetyMarginDefault"], data["last_updated"], data.get("pricingRegion")
    )


def projection(instance: InstanceInfo, count: int, hours: float) -> dict:
    if count < 1 or not math.isfinite(hours) or hours <= 0:
        raise GcpError("--instances must be positive and --hours must be finite and positive")
    price = instance.get("pricePerHourUsd")
    if price is None:
        return {"price_per_hour": None, "deployment_price_per_hour": None, "estimated_cost": None}
    try:
        hourly = price * count
        total = hourly * hours
    except OverflowError as error:
        raise GcpError("Projected compute cost is too large") from error
    if not math.isfinite(total):
        raise GcpError("Projected compute cost is too large")
    return {"price_per_hour": price, "deployment_price_per_hour": round(hourly, 6), "estimated_cost": round(total, 6)}


def estimate(
    model_id: str,
    options: EstimateOptions,
    *,
    instance_type: str | None = None,
    accelerator_type: str | None = None,
    accelerator_count: int | None = None,
    tensor_parallel_size: int | None = None,
    safety_margin: float | None = None,
    count: int = 1,
    hours: float = 1,
    token: str | None = None,
) -> dict:
    catalog = load_catalog()
    margin = catalog.safety_margin_default if safety_margin is None else safety_margin
    if not math.isfinite(margin) or not 0 <= margin < 1:
        raise GcpError("--safety-margin must be finite and in [0, 1)")
    if tensor_parallel_size is not None and tensor_parallel_size < 1:
        raise GcpError("--tensor-parallel-size must be positive")
    projection({}, count, hours)
    candidates = [
        item
        for item in catalog.instances
        if (instance_type is None or item["instanceType"] == instance_type)
        and (accelerator_type is None or item["acceleratorType"] == accelerator_type)
        and (accelerator_count is None or item["numGpu"] == accelerator_count)
        and item["numGpu"] % (tensor_parallel_size or 1) == 0
    ]
    if not candidates:
        raise GcpError("No catalog GPU layout matches the requested machine and accelerator settings")
    layouts = [(tensor_parallel_size or item["numGpu"], item["gpuMemGbPerGpu"]) for item in candidates]
    unique_layouts = list(dict.fromkeys(layouts))
    results = run_hf_mem_batch(
        model_id, unique_layouts, options=options, gpu_memory_utilization=1 - margin, token=token
    )
    by_layout = dict(zip(unique_layouts, results, strict=True))
    matches = []
    for item, layout in zip(candidates, layouts, strict=True):
        result = by_layout[layout]
        if result.fit.fits:
            matches.append(
                {
                    "instance_type": item["instanceType"],
                    "accelerator": item["acceleratorType"],
                    "gpus": item["numGpu"],
                    "gib_per_gpu": item["gpuMemGbPerGpu"],
                    "tensor_parallel_size": layout[0],
                    **projection(item, count, hours),
                    "fit": asdict(result.fit),
                }
            )
    matches.sort(key=lambda row: (row["price_per_hour"] is None, row["price_per_hour"] or 0, row["instance_type"]))
    return {
        "model_id": model_id,
        "estimator": HF_MEM_VERSION,
        "workload": asdict(options),
        "safety_margin": margin,
        "instances": count,
        "hours": hours,
        "currency": "USD",
        "pricing_region": catalog.pricing_region,
        "catalog_date": catalog.last_updated,
        "runtime_verified": False,
        "warnings": list(dict.fromkeys(warning for result in results for warning in result.warnings)),
        "matches": matches,
    }


PYTORCH_REGISTRY = "deeplearning-platform-release/gcr.io"
VLLM_IMAGE = "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20250312_0916_RC01"
GPU_FAMILIES = ("a2-", "a3-", "a4-", "a4x-", "g2-", "g4-")


@dataclass(frozen=True)
class DeploymentPlan:
    model_id: str
    name: str
    project: str | None
    location: str
    task: str
    engine: str
    machine_type: str
    accelerator_type: str | None
    accelerator_count: int
    min_replicas: int
    max_replicas: int
    service_account: str | None
    tensor_parallel_size: int | None
    image_uri: str
    container_args: list[str]
    predict_route: str
    health_route: str
    startup_timeout_seconds: int
    environment: dict[str, str]
    secret_environment: list[str]
    use_hf_token: bool
    wait: bool

    def public(self) -> dict:
        plan = asdict(self)
        for key in self.secret_environment:
            del plan["environment"][key]
        return plan


def hub_token(explicit: str | None) -> str | None:
    return explicit or get_token()


def deployment_plan(
    model_id: str,
    *,
    name: str | None,
    project: str | None,
    location: str,
    machine_type: str | None,
    accelerator_type: str | None,
    accelerator_count: int | None,
    min_replicas: int,
    max_replicas: int,
    service_account: str | None,
    tensor_parallel_size: int | None,
    max_model_len: int | None,
    engine: str,
    image_uri: str | None,
    environment: dict[str, str],
    token: str | None,
    use_hf_token: bool,
    wait: bool,
) -> DeploymentPlan:
    validate_repo_id(model_id)
    if name is None:
        prefix = re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-")[:50].rstrip("-")
        name = f"hf-{prefix or 'model'}-{uuid4().hex[:8]}"
    if not ENDPOINT_NAME.fullmatch(name):
        raise GcpError("Endpoint names must start with a lowercase letter and use up to 63 letters, digits or hyphens")
    if min_replicas < 1 or max_replicas < min_replicas:
        raise GcpError("Replica counts must be positive and --max-replicas must be at least --min-replicas")
    if machine_type and not MACHINE_TYPE.fullmatch(machine_type):
        raise GcpError("Use a Vertex AI machine type such as g2-standard-4")

    env = validate_environment(environment)
    for key in ("HF_MODEL_ID", "MODEL_ID", "HF_TOKEN"):
        if key in env:
            raise GcpError(f"Use MODEL_ID or --token to set {key}")
    task = env.get("HF_TASK")
    if not task:
        info = HfApi().model_info(model_id, token=token)
        task = info.pipeline_tag or "text-generation"
        use_hf_token = use_hf_token or bool(info.gated or info.private)

    catalog = load_catalog()
    selected = None
    if machine_type is None:
        parallel = tensor_parallel_size
        if engine == "pytorch" or (engine == "auto" and task != "text-generation"):
            parallel = 1
        matches = estimate(
            model_id,
            EstimateOptions(max_model_len=max_model_len),
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            tensor_parallel_size=parallel,
            token=token,
        )["matches"]
        if not matches or matches[0]["price_per_hour"] is None:
            raise GcpError("No priced machine fits this model. Run estimate or specify --machine-type")
        selected = matches[0]
        machine_type = selected["instance_type"]
    item = next((entry for entry in catalog.instances if entry["instanceType"] == machine_type), None)
    if item:
        if accelerator_type not in (None, item["acceleratorType"]) or accelerator_count not in (None, item["numGpu"]):
            raise GcpError(f"{machine_type} uses {item['numGpu']} × {item['acceleratorType']}")
        accelerator_type, accelerator_count = item["acceleratorType"], item["numGpu"]
    elif accelerator_type is None and (accelerator_count is not None or machine_type.startswith(GPU_FAMILIES)):
        raise GcpError("Supply --accelerator-type for a GPU machine outside the catalog")
    else:
        accelerator_count = (accelerator_count or 1) if accelerator_type else 0

    if engine == "auto":
        engine = "vllm" if accelerator_count and task == "text-generation" else "pytorch"
    if engine == "vllm" and not accelerator_count:
        raise GcpError("vLLM requires a GPU machine")
    if engine != "vllm" and (tensor_parallel_size is not None or max_model_len is not None):
        raise GcpError("--tensor-parallel-size and --max-model-len require vLLM")
    env["HF_TASK"] = task
    parallel = None
    args = []
    if engine == "vllm":
        parallel = tensor_parallel_size or accelerator_count
        if parallel > accelerator_count or accelerator_count % parallel:
            raise GcpError("--tensor-parallel-size must divide the machine's GPU count")
        args = [
            "python",
            "-m",
            "vllm.entrypoints.api_server",
            "--host=0.0.0.0",
            "--port=8080",
            f"--model={model_id}",
            f"--tensor-parallel-size={parallel}",
            f"--gpu-memory-utilization={1 - catalog.safety_margin_default:g}",
            "--disable-log-stats",
        ]
        context = max_model_len or (selected and selected["fit"]["max_model_len"])
        if context:
            args.append(f"--max-model-len={context}")
        image_uri = image_uri or VLLM_IMAGE
        predict_route, health_route, timeout = "/generate", "/ping", 7200
        env["MODEL_ID"] = model_id
    else:
        registry = location.split("-", 1)[0]
        if registry not in ("europe", "asia"):
            registry = "us"
        variant = "cu121" if accelerator_count else "cpu"
        image_uri = image_uri or (
            f"{registry}-docker.pkg.dev/{PYTORCH_REGISTRY}/"
            f"huggingface-pytorch-inference-{variant}.2-3.transformers.4-48.ubuntu2204.py311"
        )
        predict_route, health_route = "/predict", "/health"
        timeout = (item or {}).get("startupTimeoutSeconds", 900)
        env["HF_MODEL_ID"] = model_id
    return DeploymentPlan(
        model_id=model_id,
        name=name,
        project=project,
        location=location,
        task=task,
        engine=engine,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        service_account=service_account,
        tensor_parallel_size=parallel,
        image_uri=image_uri,
        container_args=args,
        predict_route=predict_route,
        health_route=health_route,
        startup_timeout_seconds=timeout,
        environment=env,
        secret_environment=sorted(key for key in env if SECRET_NAME.search(key)),
        use_hf_token=use_hf_token,
        wait=wait,
    )
