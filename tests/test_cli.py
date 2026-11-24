import json
import sys
from unittest.mock import Mock

import pytest

from hf_gcp import cli, planning
from hf_gcp.hf_mem import GpuFit, HfMemEstimate

PREVIEW = [
    "deploy",
    "openai-community/gpt2",
    "--name",
    "gpt2-demo",
    "--machine-type",
    "g2-standard-4",
    "--env",
    "HF_TASK=text-generation",
]


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["hf-gcp", *args])
    return cli.main()


def test_catalog_cost_and_json(monkeypatch, capsys):
    assert run(monkeypatch, "-o", "json", "instances", "g2-standard-4", "--instances", "2", "--hours", "24") == 0
    output = json.loads(capsys.readouterr().out)
    assert output["pricing_region"] == "us-central1"
    assert output["matches"][0]["estimated_cost"] == 39.02064


def test_estimate_batches_layouts_and_filters(monkeypatch, capsys):
    def fake_estimate(model_id, layouts, **kw):
        assert model_id == "openai-community/gpt2"
        assert layouts == [(1, 24)]
        assert kw["options"].max_model_len == 1024
        return [HfMemEstimate(GpuFit(True, 100, 1, 24 * 1024**3, 1, 1024), ())]

    monkeypatch.setattr(planning, "run_hf_mem_batch", fake_estimate)
    assert (
        run(
            monkeypatch,
            "estimate",
            "openai-community/gpt2",
            "--accelerator-type",
            "nvidia-l4",
            "--accelerator-count",
            "1",
            "--max-model-len",
            "1024",
            "--limit",
            "1",
            "-o",
            "json",
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["total_matches"] > 1
    assert result["matches"][0]["instance_type"] == "g2-standard-4"


def test_auto_hardware_and_context(monkeypatch, capsys):
    fit = {"instance_type": "g2-standard-48", "price_per_hour": 4, "fit": {"max_model_len": 16384}}
    estimate = Mock(return_value={"matches": [fit]})
    monkeypatch.setattr(planning, "estimate", estimate)
    assert run(monkeypatch, "deploy", "org/model", "--env", "HF_TASK=text-generation", "--dry-run") == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["machine_type"] == "g2-standard-48"
    assert plan["tensor_parallel_size"] == 4
    assert "--max-model-len=16384" in plan["container_args"]
    estimate.assert_called_once()


@pytest.mark.parametrize("mode", ["--dry-run", "--print-code"])
def test_preview_is_offline_and_omits_secrets(monkeypatch, capsys, tmp_path, mode):
    monkeypatch.setattr(planning, "HfApi", Mock(side_effect=AssertionError("Hub accessed")))
    monkeypatch.setattr(cli, "apply_plan", Mock(side_effect=AssertionError("Vertex accessed")))
    env_file = tmp_path / "deploy.env"
    env_file.write_text("LABEL=from-file\nCUSTOM_PASSWORD=container-secret\nHF_TOKEN=hub-secret\n")
    assert (
        run(
            monkeypatch,
            "--project",
            "project",
            *PREVIEW,
            "--location",
            "europe-west4",
            "--env-file",
            str(env_file),
            "--env",
            "LABEL=from-cli",
            "--min-replicas",
            "2",
            "--max-replicas",
            "4",
            mode,
        )
        == 0
    )
    result = capsys.readouterr().out
    assert "container-secret" not in result
    assert "hub-secret" not in result
    if mode == "--print-code":
        namespace = {}
        exec(compile(result, "deploy.py", "exec"), namespace)
        assert "hf_gcp" not in result
        plan = namespace["PLAN"]
    else:
        plan = json.loads(result)
    assert plan["environment"]["LABEL"] == "from-cli"
    assert plan["secret_environment"] == ["CUSTOM_PASSWORD"]
    assert plan["use_hf_token"]
    assert plan["project"] == "project"
    assert plan["location"] == "europe-west4"
    assert plan["max_replicas"] == 4
    assert plan["accelerator_type"] == "NVIDIA_L4"


def test_cpu_preview(monkeypatch, capsys):
    assert (
        run(
            monkeypatch,
            "deploy",
            "org/model",
            "--machine-type",
            "n1-standard-4",
            "--env",
            "HF_TASK=text-classification",
            "--dry-run",
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["engine"] == "pytorch"
    assert plan["accelerator_count"] == 0
    assert plan["environment"]["HF_MODEL_ID"] == "org/model"
    assert plan["predict_route"] == "/predict"


def test_json_parse_error(monkeypatch, capsys):
    with pytest.raises(SystemExit, match="2"):
        run(monkeypatch, *PREVIEW, "--accelerator-type", "wrong", "-o", "json")
    result = capsys.readouterr()
    assert not result.out
    assert "Vertex AI GPU" in json.loads(result.err)["error"]


def test_estimate_terminal_summary_and_json_alias(monkeypatch, capsys):
    worker = Mock(
        side_effect=lambda model, layouts, **kwargs: [
            HfMemEstimate(GpuFit(True, 2 * 1024**3, count, int(gib * 1024**3), 1, 1024), ("Test note",))
            for count, gib in layouts
        ]
    )
    monkeypatch.setattr(planning, "run_hf_mem_batch", worker)
    monkeypatch.setenv("COLUMNS", "100")
    options = (
        "estimate",
        "openai-community/gpt2",
        "--instances",
        "2",
        "--hours",
        "24",
        "--limit",
        "1",
        "--project",
        "test-project",
        "--location",
        "europe-west4",
    )
    assert run(monkeypatch, *options) == 0
    result = capsys.readouterr()
    assert "Vertex AI estimate for openai-community/gpt2" in result.out
    assert "Cheapest compatible machine" in result.out
    assert "VRAM used / total" in result.out
    assert "$39.02 for 24h × 2 replicas" in result.out
    assert "hf gcp deploy openai-community/gpt2 --machine-type g2-standard-4 --dry-run" in result.out
    assert "--location europe-west4 --project test-project" in result.out
    assert "Notes" in result.err and "Test note" in result.err
    assert run(monkeypatch, "--json", *options) == 0
    result = capsys.readouterr()
    assert json.loads(result.out)["matches"][0]["estimated_cost"] == pytest.approx(39.02064)
    assert result.err == ""


def test_endpoint_panel_renders_vertex_model_fields(monkeypatch, capsys):
    data = {
        "name": "test[endpoint]",
        "status": "DEPLOYED",
        "endpoint": "projects/demo/locations/us-central1/endpoints/test",
        "models": [
            {
                "id": "1",
                "model": "projects/demo/locations/us-central1/models/model",
                "model_id": "org/model",
                "machine_type": "g2-standard-4",
                "accelerator": "NVIDIA_L4",
                "gpus": 1,
                "min_replicas": 1,
                "max_replicas": 2,
                "traffic_percent": 100,
            }
        ],
        "labels": {"hf-gcp": "true"},
    }
    monkeypatch.setenv("COLUMNS", "100")
    monkeypatch.setattr(cli, "Deployments", Mock(return_value=Mock(describe=Mock(return_value=data))))
    options = ("deployments", "describe", "test")
    assert run(monkeypatch, *options) == 0
    output = capsys.readouterr().out
    assert "Vertex AI endpoint" in output
    assert "test[endpoint]" in output
    assert "DEPLOYED" in output
    assert "Deployed models" in output
    assert "org/model" in output
    assert "1 × NVIDIA_L4" in output
    assert "1–2" in output
    assert run(monkeypatch, *options, "-j") == 0
    assert json.loads(capsys.readouterr().out) == data


def test_json_alias_applies_to_nested_argument_errors(monkeypatch, capsys):
    with pytest.raises(SystemExit) as error:
        run(monkeypatch, "deployments", "logs", "test", "--tail", "0", "--json")
    assert error.value.code == 2
    output = capsys.readouterr()
    assert not output.out
    assert json.loads(output.err)["error"].endswith("expected a positive integer")
