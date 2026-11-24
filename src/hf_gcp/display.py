"""Rich terminal output for Vertex AI planning and endpoint commands."""

import shlex
from collections.abc import Iterable
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

from huggingface_hub.utils import HFValidationError, validate_repo_id
from rich import box
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

STATUS_STYLES = {
    "DEPLOYED": "bold green",
    "DELETED": "bold green",
    "SUCCEEDED": "bold green",
    "FAILED": "bold red",
    "DEPLOYING": "yellow",
    "DELETING": "yellow",
    "EMPTY": "dim",
}


def terminal(*, stderr: bool = False) -> Console:
    return Console(stderr=stderr, markup=False, highlight=False)


def activity(message: str, *, enabled: bool = True):
    console = terminal(stderr=True)
    return (
        console.status(Text(message, style="cyan"), spinner="dots")
        if enabled and console.is_terminal
        else nullcontext()
    )


def value(item: Any) -> Text:
    if item is None:
        return Text("—", style="dim")
    if isinstance(item, bool):
        return Text("Yes" if item else "No")
    if isinstance(item, datetime):
        return Text(item.isoformat(sep=" ", timespec="seconds"))
    return Text(str(item))


def status_text(status: str) -> Text:
    return Text(status, style=STATUS_STYLES.get(status, ""))


def usd(amount: float | None) -> str:
    return f"${amount:,.2f}" if amount is not None else "—"


def table(
    console: Console,
    title: str,
    headers: tuple[str, ...],
    rows: Iterable[tuple[RenderableType, ...]],
    *,
    numeric: tuple[str, ...] = (),
    empty: str = "No results.",
) -> None:
    result = Table(
        title=Text(title, style="bold") if title else None,
        title_justify="left",
        box=box.SIMPLE_HEAD,
        header_style="bold",
    )
    for index, header in enumerate(headers):
        result.add_column(
            header,
            style="cyan" if index == 0 else "",
            justify="right" if header in numeric else "left",
            overflow="fold",
        )
    for row in rows:
        result.add_row(*row)
    if result.row_count:
        console.print(result)
    else:
        console.print(empty, style="dim")


def fields(rows: Iterable[tuple[str, RenderableType]]) -> Table:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold")
    grid.add_column()
    for label, item in rows:
        grid.add_row(label, item)
    return grid


def listing(
    title: str, headers: dict[str, str], rows: list[dict], *, empty: str, numeric: tuple[str, ...] = ()
) -> None:
    table(
        terminal(),
        title,
        tuple(headers.values()),
        (tuple(status_text(row[key]) if key == "status" else value(row.get(key)) for key in headers) for row in rows),
        numeric=numeric,
        empty=empty,
    )


def accelerator(row: dict) -> str:
    name = row["accelerator"].removeprefix("NVIDIA_").removeprefix("TESLA_")
    name = name.removesuffix(f"_{row['gib_per_gpu']:g}GB").replace("_", " ")
    return f"{row['gpus']} × {row['gib_per_gpu']:g} GiB {name}"


def _preview_command(result: dict, row: dict, location: str | None, project: str | None) -> str | None:
    model = result["model_id"]
    workload = result["workload"]
    if workload["revision"] or workload["file"] or workload["format"] not in (None, "safetensors"):
        return None
    try:
        validate_repo_id(model)
    except HFValidationError:
        return None
    if Path(model).exists():
        return None

    args = ["hf", "gcp", "deploy", model, "--machine-type", row["instance_type"], "--dry-run"]
    if result["instances"] != 1:
        args.extend(["--instances", str(result["instances"])])
    if row["tensor_parallel_size"] != row["gpus"]:
        args.extend(["--tensor-parallel-size", str(row["tensor_parallel_size"])])
    if location:
        args.extend(["--location", location])
    if project:
        args.extend(["--project", project])
    return shlex.join(args)


def catalog(result: dict, *, location: str | None, project: str | None) -> None:
    console = terminal()
    is_estimate = "workload" in result
    instance_count, hours = result["instances"], result["hours"]
    catalog_region = result["pricing_region"] or "region unspecified"
    replicas = f"{instance_count} {'replica' if instance_count == 1 else 'replicas'}"
    matches = result["matches"]
    headers = ["Machine", "GPU layout"]
    show_memory = is_estimate and console.width >= 76
    separate_tp = is_estimate and any(row["tensor_parallel_size"] != row["gpus"] for row in matches)
    if is_estimate:
        console.print(Text(f"Vertex AI estimate for {result['model_id']}", style="bold"))
        workload = result["workload"]
        context = workload["max_model_len"]
        context_label = f"{context:,} tokens" if context else "model default"
        console.print(
            f"Context {context_label} · Batch {workload['batch_size']} · KV {workload['kv_cache_dtype']} "
            f"· Reserve {result['safety_margin'] * 100:g}%",
            style="dim",
        )
        if show_memory:
            headers.append("VRAM used / total")
        if separate_tp:
            headers.append("TP")
    else:
        headers.append("Startup timeout")
    rate_header = "Est. $/h" if hours == 1 else f"Est. {hours:g}h"
    headers.append(rate_header)
    price_key = "deployment_price_per_hour" if hours == 1 else "estimated_cost"

    rows = []
    for row in matches:
        cells = [row["instance_type"], accelerator(row)]
        if is_estimate:
            if show_memory:
                fit = row["fit"]
                required = fit["required_bytes"] / 1024**3
                total = fit["memory_per_gpu_bytes"] * fit["gpu_count"] / 1024**3
                cells.append(f"{required:,.2f} / {total:g}")
            if separate_tp:
                cells.append(str(row["tensor_parallel_size"]))
        else:
            timeout = row["startup_timeout_seconds"]
            cells.append(f"{timeout:g}s" if timeout is not None else "—")
        rows.append((*cells, usd(row[price_key])))
    table(
        console,
        "" if is_estimate else "Vertex AI GPU Machines",
        tuple(headers),
        rows,
        numeric=("VRAM used / total", "TP", "Startup timeout", rate_header),
        empty="No compatible Vertex AI GPU machines found.",
    )

    if is_estimate:
        if matches:
            memory_units = " · VRAM in GiB" if show_memory else ""
            console.print(
                f"{len(matches)} of {result['total_matches']} compatible machines · cheapest first{memory_units}",
                style="dim",
            )
            best = matches[0]
            if best["deployment_price_per_hour"] is not None:
                detail = Text(f"{usd(best['deployment_price_per_hour'])}/h · {best['instance_type']}")
                if hours != 1 or instance_count != 1:
                    detail.append(f"\n{usd(best['estimated_cost'])} for {hours:g}h × {replicas}")
                console.print(
                    Panel.fit(detail, title="Cheapest compatible machine", title_align="left", border_style="green")
                )
                if command := _preview_command(result, best, location, project):
                    console.print("Preview deployment", style="bold")
                    console.print(Text(command, style="cyan"), soft_wrap=True)
            else:
                console.print("Catalog prices are unavailable for the compatible machines.", style="yellow")
        console.print("Runtime support, Vertex AI quota, and capacity are unverified.", style="dim")
        if result["warnings"]:
            stderr = terminal(stderr=True)
            stderr.print("Notes", style="bold yellow")
            for warning in result["warnings"]:
                stderr.print(Text(f"• {warning}", style="dim"))
    elif instance_count != 1 or hours != 1:
        console.print(f"Projected compute for {hours:g}h × {replicas}", style="dim")
    console.print(f"Price snapshot · {result['catalog_date']} · {catalog_region} · USD", style="dim")


def endpoint(title: str, info: dict) -> None:
    labels = {
        "name": "Endpoint",
        "status": "Status",
        "model_id": "Hub model",
        "location": "Location",
        "engine": "Runtime",
        "machine_type": "Machine",
        "endpoint": "Resource",
        "model": "Model Registry",
        "deployed_model_id": "Deployed model ID",
        "operation": "Operation",
        "predict_url": "Predict URL",
        "created_at": "Created",
        "updated_at": "Updated",
    }
    rows = [
        (label, status_text(info[key]) if key == "status" else value(info[key]))
        for key, label in labels.items()
        if info.get(key) is not None
    ]
    if tags := info.get("labels"):
        rows.append(("Labels", value(", ".join(f"{key}={item}" for key, item in sorted(tags.items())))))
    border = STATUS_STYLES.get(info.get("status"), "cyan")
    terminal().print(Panel(fields(rows), title=title, title_align="left", border_style=border))

    if models := info.get("models"):
        listing(
            "Deployed models",
            {
                "id": "ID",
                "model": "Model",
                "machine_type": "Machine",
                "gpu_layout": "GPUs",
                "replicas": "Min–max replicas",
                "traffic": "Traffic",
            },
            [
                {
                    **model,
                    "model": model.get("model_id") or model["model"],
                    "gpu_layout": f"{model['gpus']} × {model['accelerator']}" if model.get("accelerator") else None,
                    "replicas": (
                        f"{model['min_replicas']}–{model['max_replicas']}" if model.get("min_replicas") else None
                    ),
                    "traffic": f"{model['traffic_percent']}%",
                }
                for model in models
            ],
            empty="No deployed models.",
            numeric=("Min–max replicas", "Traffic"),
        )
