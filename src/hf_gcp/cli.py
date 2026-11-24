"""Command-line entry point for hf gcp and hf-gcp."""

import argparse
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

from google.api_core.exceptions import GoogleAPIError
from google.auth.exceptions import GoogleAuthError
from google.cloud.aiplatform_v1 import AcceleratorType
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from requests.exceptions import RequestException
from rich.text import Text
from rich_argparse import RichHelpFormatter

from hf_gcp import display
from hf_gcp._serve import (
    VertexScope,
    apply_plan,
    default_location,
    default_project,
    validate_location,
    validate_project,
)
from hf_gcp.deploy_code import deployment_code
from hf_gcp.deployments import Deployments, endpoint_summary
from hf_gcp.environment import load_environment
from hf_gcp.errors import GcpError, redact
from hf_gcp.hf_mem import EstimateOptions
from hf_gcp.planning import deployment_plan, estimate, hub_token, load_catalog, projection
from hf_gcp.quota import quotas

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def json_default(value: object) -> str:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def emit(args: argparse.Namespace, data: object) -> None:
    if isinstance(data, str) and args.output != "json":
        print(data)
    else:
        print(json.dumps(data, indent=2, ensure_ascii=False, default=json_default))


def emit_endpoint(args: argparse.Namespace, title: str, data: dict) -> None:
    if args.output == "json":
        emit(args, data)
    else:
        display.endpoint(title, data)


def emit_listing(
    args: argparse.Namespace,
    title: str,
    headers: dict[str, str],
    rows: list[dict],
    *,
    empty: str,
    numeric: tuple[str, ...] = (),
) -> None:
    if args.output == "json":
        emit(args, rows)
    else:
        display.listing(title, headers, rows, empty=empty, numeric=numeric)


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def accelerator(value: str) -> str:
    name = value.upper().replace("-", "_")
    if name == "NVIDIA_A100_40GB":
        name = "NVIDIA_TESLA_A100"
    if name not in AcceleratorType.__members__ or not name.startswith("NVIDIA_"):
        raise argparse.ArgumentTypeError("use a Vertex AI GPU accelerator such as NVIDIA_L4")
    return name


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("formatter_class", RichHelpFormatter)
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> None:
        print_error(message, output_format(sys.argv[1:]), code=2)
        raise SystemExit(2)


def shared_options(parser: argparse.ArgumentParser) -> None:
    # Suppress defaults so options can be passed at any command level.
    parser.add_argument(
        "-o",
        "--output",
        choices=("table", "json"),
        default=argparse.SUPPRESS,
        help="Output format (default: table)",
    )
    parser.add_argument(
        "-j",
        "--json",
        action="store_const",
        const="json",
        dest="output",
        default=argparse.SUPPRESS,
        help="Output machine-readable JSON",
    )
    parser.add_argument("-l", "--location", "--region", default=argparse.SUPPRESS, help="Vertex AI region")
    parser.add_argument("-p", "--project", default=argparse.SUPPRESS, help="Google Cloud project ID or number")


def hardware_filters(command: argparse._ActionsContainer, *, machine_help: str) -> None:
    command.add_argument("--machine-type", "--instance-type", "-m", help=machine_help)
    command.add_argument("--accelerator-type", type=accelerator, help="Vertex AI GPU type, such as NVIDIA_L4")
    command.add_argument("--accelerator-count", type=positive, metavar="GPUS", help="GPUs per replica")


def add_cost_options(command: argparse.ArgumentParser) -> None:
    cost = command.add_argument_group("Cost projection")
    cost.add_argument(
        "--instances",
        "--instance-count",
        type=positive,
        default=1,
        metavar="COUNT",
        help="Number of endpoint replicas",
    )
    cost.add_argument("--hours", type=float, default=1, metavar="HOURS", help="Continuous running time in hours")


def parser() -> argparse.ArgumentParser:
    root = ArgumentParser(
        prog="hf gcp",
        usage="%(prog)s [OPTIONS] COMMAND [ARGS]...",
        description="Plan and deploy Hub models on Google Cloud Vertex AI.",
    )
    shared_options(root)
    commands = root.add_subparsers(prog=root.prog, dest="command", required=True, title="Commands", metavar="")

    def command(name: str, description: str) -> argparse.ArgumentParser:
        child = commands.add_parser(name, help=description, description=description)
        shared_options(child)
        return child

    models = command("models", "Search the Hugging Face Hub or inspect a model.")
    models.add_argument("model_id", nargs="?", metavar="MODEL_ID", help="Hub model ID, or list to browse the Hub")
    models.add_argument("--search", help="Search Hub model names")
    models.add_argument("--task", help="Hub pipeline tag filter for model search")
    models.add_argument("--limit", type=positive, default=20)

    instances = command("instances", "List or inspect GPU machines in the bundled Vertex AI catalog.")
    instances.add_argument(
        "machine_type", nargs="?", metavar="MACHINE_TYPE", help="Inspect a Vertex AI machine such as g2-standard-4"
    )
    add_cost_options(instances)

    estimation = command("estimate", "Rank Vertex AI GPU machines by memory fit and projected cost.")
    estimation.usage = "%(prog)s [OPTIONS] MODEL"
    estimation.add_argument("model_id", metavar="MODEL", help="Hub model ID, URL, or local model path")
    comparison = estimation.add_argument_group("Comparison")
    hardware_filters(comparison, machine_help="Estimate one bundled machine type")
    comparison.add_argument("--tensor-parallel-size", type=positive, metavar="GPUS", help="GPUs used per replica")
    comparison.add_argument(
        "--safety-margin", type=float, metavar="FRACTION", help="Fraction of GPU memory to reserve"
    )
    comparison.add_argument("--limit", type=positive, metavar="ROWS", help="Maximum number of matching rows to show")
    workload = estimation.add_argument_group("Workload")
    workload.add_argument("--max-model-len", type=positive, metavar="TOKENS", help="Context length in tokens")
    workload.add_argument("--batch-size", type=positive, metavar="SEQUENCES", default=1, help="Concurrent sequences")
    workload.add_argument(
        "--kv-cache-dtype", default="auto", metavar="DTYPE", help="KV cache dtype, such as auto or fp8"
    )
    workload.add_argument("--max-num-batched-tokens", type=positive, metavar="TOKENS", help="Engine tokens per batch")
    weights = estimation.add_argument_group("Model selection")
    weights.add_argument("--revision", help="Hub branch, tag, or commit")
    weights.add_argument("--file", help="Specific model weights file")
    weights.add_argument("--format", choices=("safetensors", "gguf", "onnx"), help="Model weights format")
    weights.add_argument("--token", help="Hugging Face token for private or gated repositories")
    add_cost_options(estimation)

    deploy = command("deploy", "Deploy a Hub model to Vertex AI or preview the deployment.")
    deploy.usage = "%(prog)s [OPTIONS] MODEL_ID"
    deploy.add_argument("model_id", metavar="MODEL_ID", help="Hugging Face Hub model ID")
    deploy.add_argument("--name", "-n", help="Endpoint ID and display name (generated by default)")
    hardware = deploy.add_argument_group("Hardware")
    hardware_filters(hardware, machine_help="Vertex AI machine type (defaults to the cheapest catalog fit)")
    replicas = hardware.add_mutually_exclusive_group()
    replicas.add_argument("--instances", "--instance-count", "-c", type=positive, help="Fixed replica count")
    replicas.add_argument("--min-replicas", type=positive, help="Minimum replica count (default: 1)")
    hardware.add_argument("--max-replicas", type=positive, help="Maximum replica count (defaults to the minimum)")
    auth = deploy.add_argument_group("Authentication")
    auth.add_argument("--service-account", help="Service account for the serving container")
    auth.add_argument("--token", help="Hugging Face token for private or gated repositories")
    runtime = deploy.add_argument_group("Serving runtime")
    runtime.add_argument("--tensor-parallel-size", type=positive, metavar="GPUS", help="vLLM GPUs used per replica")
    runtime.add_argument("--max-model-len", type=positive, metavar="TOKENS", help="vLLM context length in tokens")
    runtime.add_argument("--engine", choices=("auto", "vllm", "pytorch"), default="auto", help="Serving engine")
    runtime.add_argument("--image-uri", help="Override the selected engine's container image")
    runtime.add_argument(
        "--env", action="append", metavar="KEY=VALUE", help="Set a runtime variable. Repeat as needed."
    )
    runtime.add_argument(
        "--env-file", action="append", type=Path, metavar="PATH", help="Load runtime variables from a file."
    )
    execution = deploy.add_argument_group("Execution")
    execution.add_argument("--no-wait", action="store_true", help="Return after submitting model deployment")
    preview = execution.add_mutually_exclusive_group()
    preview.add_argument("--dry-run", action="store_true", help="Print the resolved plan as JSON")
    preview.add_argument("--print-code", action="store_true", help="Export standalone Vertex AI SDK Python code")

    quota = command("quota", "Show applied Vertex AI online prediction quotas.")
    quota_filter = quota.add_mutually_exclusive_group()
    quota_filter.add_argument("--machine-type", "--instance-type", "-m", help="GPU quota for a catalog machine")
    quota_filter.add_argument("--accelerator-type", type=accelerator, help="Quota for a Vertex AI GPU type")

    endpoints = command("deployments", "Manage Vertex AI online prediction endpoints.")
    endpoints.usage = "%(prog)s [OPTIONS] COMMAND [ARGS]..."
    actions = endpoints.add_subparsers(prog=endpoints.prog, dest="action", required=True, title="Commands", metavar="")
    for action, description in {
        "list": "List Vertex AI endpoints in the selected location.",
        "status": "Check endpoint status.",
        "describe": "Inspect the endpoint and its deployed models.",
        "wait": "Wait for a model deployment or a Vertex AI operation.",
        "logs": "Read recent endpoint logs from Cloud Logging.",
        "invoke": "Send an inference request to an endpoint.",
        "scale": "Set replica counts for a deployed model.",
        "delete": "Undeploy models and delete a Vertex AI endpoint.",
    }.items():
        child = actions.add_parser(action, help=description, description=description)
        shared_options(child)
        if action == "wait":
            child.add_argument(
                "name", metavar="ENDPOINT_OR_OPERATION", help="Endpoint or full operation resource name"
            )
        elif action != "list":
            child.add_argument("name", metavar="ENDPOINT", help="Endpoint ID, display name, or full resource name")
    actions.choices["list"].add_argument("--all", action="store_true", help="Include endpoints created by other tools")
    actions.choices["wait"].add_argument("--timeout", type=positive, default=9000, help="Timeout in seconds")
    actions.choices["logs"].add_argument(
        "--tail", type=positive, default=100, metavar="LINES", help="Recent lines to show (default: 100)"
    )
    actions.choices["logs"].add_argument("--raw", action="store_true", help="Preserve ANSI escapes")
    scale = actions.choices["scale"]
    scale_count = scale.add_mutually_exclusive_group(required=True)
    scale_count.add_argument("--instances", type=positive, help="Fixed replica count")
    scale_count.add_argument("--min-replicas", type=positive)
    scale.add_argument("--max-replicas", type=positive)
    scale.add_argument("--deployed-model-id", help="Required when the endpoint hosts multiple models")
    delete = actions.choices["delete"]
    delete.add_argument("--yes", "--force", action="store_true", help="Skip confirmation")
    delete.add_argument("--wait", action="store_true", help="Wait until the endpoint is deleted")
    invoke = actions.choices["invoke"]
    request = invoke.add_mutually_exclusive_group(required=True)
    request.add_argument("--input", help="Text input")
    request.add_argument("--payload", help="JSON body, @FILE, or - to read stdin")
    invoke.add_argument("--raw", action="store_true", help="Send the payload directly to the container")
    invoke.add_argument("--max-tokens", type=positive, default=100, help="vLLM token limit for --input")
    return root


def run_models(args: argparse.Namespace) -> None:
    api = HfApi()
    if args.model_id and args.model_id != "list":
        if args.search or args.task:
            raise GcpError("--search and --task apply to model listing")
        models = [api.model_info(args.model_id)]
    else:
        models = api.list_models(search=args.search, pipeline_tag=args.task, limit=args.limit)
    rows = [
        {"model_id": model.id, "task": model.pipeline_tag, "library": model.library_name, "gated": model.gated}
        for model in models
    ]
    emit_listing(
        args,
        "Hugging Face Hub Models",
        {"model_id": "Model", "task": "Task", "library": "Library", "gated": "Gated"},
        rows,
        empty="No Hub models matched the search.",
    )


def catalog_output(args: argparse.Namespace) -> None:
    if args.command == "instances":
        catalog = load_catalog()
        projection({}, args.instances, args.hours)
        rows = [
            {
                "instance_type": item["instanceType"],
                "accelerator": item["acceleratorType"],
                "gpus": item["numGpu"],
                "gib_per_gpu": item["gpuMemGbPerGpu"],
                "startup_timeout_seconds": item.get("startupTimeoutSeconds"),
                **projection(item, args.instances, args.hours),
            }
            for item in catalog.instances
            if args.machine_type is None or args.machine_type == item["instanceType"]
        ]
        if args.machine_type and not rows:
            raise GcpError(f"{args.machine_type} is absent from the bundled Vertex AI catalog")
        result = {
            "catalog_date": catalog.last_updated,
            "pricing_region": catalog.pricing_region,
            "currency": "USD",
            "hours": args.hours,
            "instances": args.instances,
            "matches": rows,
        }
    else:
        options = EstimateOptions(
            revision=args.revision,
            file=args.file,
            format=args.format,
            max_model_len=args.max_model_len,
            batch_size=args.batch_size,
            kv_cache_dtype=args.kv_cache_dtype,
            max_num_batched_tokens=args.max_num_batched_tokens,
        )
        with display.activity(f"Estimating GPU memory for {args.model_id}…", enabled=args.output != "json"):
            result = estimate(
                args.model_id,
                options,
                instance_type=args.machine_type,
                accelerator_type=args.accelerator_type,
                accelerator_count=args.accelerator_count,
                tensor_parallel_size=args.tensor_parallel_size,
                safety_margin=args.safety_margin,
                count=args.instances,
                hours=args.hours,
                token=hub_token(args.token),
            )
        result["total_matches"] = len(result["matches"])
        result["matches"] = result["matches"][: args.limit]
    if args.output == "json":
        emit(args, result)
    else:
        display.catalog(result, location=args.location, project=args.project)


def replica_counts(args: argparse.Namespace) -> tuple[int, int]:
    if args.instances is not None and args.max_replicas is not None:
        raise GcpError("--instances sets a fixed count. Use --min-replicas with --max-replicas for autoscaling")
    minimum = args.instances or args.min_replicas or 1
    maximum = args.max_replicas or minimum
    if maximum < minimum:
        raise GcpError("--max-replicas must be at least --min-replicas")
    return minimum, maximum


def run_deploy(args: argparse.Namespace) -> None:
    environment = load_environment(args.env, args.env_file)
    environment_token = environment.pop("HF_TOKEN", None)
    args.token = args.token or environment_token
    token = hub_token(args.token)
    minimum, maximum = replica_counts(args)
    project, location = args.project or default_project(), args.location or default_location()
    if project:
        validate_project(project)
    validate_location(location)
    show_progress = args.output != "json" and not (args.dry_run or args.print_code)
    with display.activity(f"Resolving Vertex AI deployment for {args.model_id}…", enabled=show_progress):
        plan = deployment_plan(
            args.model_id,
            name=args.name,
            project=project,
            location=location,
            machine_type=args.machine_type,
            accelerator_type=args.accelerator_type,
            accelerator_count=args.accelerator_count,
            min_replicas=minimum,
            max_replicas=maximum,
            service_account=args.service_account,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            engine=args.engine,
            image_uri=args.image_uri,
            environment=environment,
            token=token,
            use_hf_token=bool(args.token),
            wait=not args.no_wait,
        )
    if args.dry_run:
        print(json.dumps(plan.public(), indent=2))
    elif args.print_code:
        print(deployment_code(plan), end="")
    else:
        with display.activity(f"Deploying {plan.model_id} to {plan.name} in {plan.location}…", enabled=show_progress):
            result = apply_plan({**plan.public(), "environment": plan.environment}, token=token)
        emit_endpoint(args, "Vertex AI deployment", result)


def run_deployments(args: argparse.Namespace) -> None:
    endpoints = Deployments(args.project, args.location, getattr(args, "name", None))
    show_progress = args.output != "json"
    if args.action == "list":
        rows = endpoints.list(all_endpoints=args.all)
        if args.output == "json":
            emit(args, rows)
        else:
            location = endpoints.scope.location
            display.listing(
                f"Vertex AI Deployments · {location}",
                {"name": "Endpoint", "id": "ID", "status": "Status", "engine": "Runtime", "created_at": "Created"},
                [{**row, "id": row["endpoint"].rsplit("/", 1)[-1]} for row in rows],
                empty=f"No Vertex AI deployments found in {location}.",
            )
    elif args.action == "status":
        emit_endpoint(args, "Endpoint status", endpoint_summary(endpoints.resolve(args.name)))
    elif args.action == "describe":
        emit_endpoint(args, "Vertex AI endpoint", endpoints.describe(args.name))
    elif args.action == "wait":
        with display.activity(f"Waiting for {args.name}…", enabled=show_progress):
            result = endpoints.wait(args.name, args.timeout)
        emit_endpoint(args, "Endpoint status", result)
    elif args.action == "scale":
        minimum, maximum = replica_counts(args)
        with display.activity(f"Scaling {args.name}…", enabled=show_progress):
            result = endpoints.scale(args.name, minimum, maximum, args.deployed_model_id)
        emit_endpoint(args, "Endpoint capacity", result)
    elif args.action == "delete":
        endpoint = endpoints.resolve(args.name)
        if not args.yes:
            if not sys.stdin.isatty():
                raise GcpError("Pass --yes to confirm endpoint deletion in a non-interactive session")
            print(f"Delete {endpoint.name}? [y/N] ", end="", file=sys.stderr, flush=True)
            if sys.stdin.readline().strip().lower() not in ("y", "yes"):
                raise GcpError("Deletion cancelled")
        with display.activity(f"Deleting {args.name}…", enabled=show_progress):
            result = endpoints.delete(endpoint, wait=args.wait)
        emit_endpoint(args, "Endpoint deletion", result)
    elif args.action == "logs":
        logs = endpoints.logs(args.name, args.tail)
        if not args.raw:
            logs = [ANSI.sub("", line) for line in logs]
        emit(args, logs if args.output == "json" else "\n".join(logs) or "No logs available.")
    elif args.action == "invoke":
        payload = None
        if args.payload is not None:
            data = args.payload
            if data.startswith("@"):
                data = Path(data[1:]).read_text(encoding="utf-8")
            elif data == "-":
                data = sys.stdin.read()
            payload = json.loads(data)
            if not isinstance(payload, dict):
                raise GcpError("Use a JSON object for --payload")
        with display.activity(f"Invoking {args.name}…", enabled=show_progress):
            result = endpoints.invoke(
                args.name, text=args.input, payload=payload, raw=args.raw, max_tokens=args.max_tokens
            )
        emit(args, result)


def run(args: argparse.Namespace) -> None:
    if args.command == "models":
        run_models(args)
    elif args.command in {"estimate", "instances"}:
        catalog_output(args)
    elif args.command == "deploy":
        run_deploy(args)
    elif args.command == "quota":
        gpu = args.accelerator_type
        if args.machine_type:
            machine = next(
                (item for item in load_catalog().instances if item["instanceType"] == args.machine_type), None
            )
            if machine is None:
                raise GcpError(f"{args.machine_type} is absent from the bundled catalog. Use --accelerator-type")
            gpu = machine["acceleratorType"]
        scope = VertexScope(args.project, args.location)
        emit_listing(
            args,
            f"Vertex AI Endpoint Quotas · {scope.location}",
            {"name": "Quota", "limit": "Limit", "region": "Region", "quota_id": "Quota ID"},
            quotas(scope, gpu),
            empty=f"No matching endpoint quotas found for {scope.location}.",
            numeric=("Limit",),
        )
    else:
        run_deployments(args)


def output_format(argv: list[str]) -> str:
    selected = "table"
    for index, token in enumerate(argv):
        if token == "--":
            break
        if token in {"-j", "--json"}:
            selected = "json"
        elif token in {"-o", "--output"} and index + 1 < len(argv):
            selected = argv[index + 1]
        elif token.startswith("--output="):
            selected = token.split("=", 1)[1]
        elif token.startswith("-o") and len(token) > 2:
            selected = token[2:].removeprefix("=")
    return selected


def print_error(message: str, output: str, *, code: int = 1) -> None:
    if output == "json":
        print(json.dumps({"error": message, "exit_code": code}), file=sys.stderr)
    else:
        display.terminal(stderr=True).print(Text.assemble(("Error: ", "bold red"), message))


def main() -> int:
    root = parser()
    if len(sys.argv) == 1:
        root.print_help()
        return 0
    args = root.parse_args(namespace=argparse.Namespace(output="table", project=None, location=None))
    try:
        run(args)
    except (
        GcpError,
        GoogleAPIError,
        GoogleAuthError,
        HfHubHTTPError,
        RequestException,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print_error(redact(str(error), getattr(args, "token", None), os.environ.get("HF_TOKEN")), args.output)
        return 1
    except KeyboardInterrupt:
        print_error("Interrupted", args.output, code=130)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
