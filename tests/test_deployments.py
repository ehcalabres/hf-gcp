from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from google.api_core.exceptions import NotFound
from google.cloud import aiplatform_v1 as vertex
from google.longrunning.operations_pb2 import Operation
from google.protobuf.json_format import ParseDict
from google.rpc.status_pb2 import Status

from hf_gcp import _serve, deployments
from hf_gcp.errors import GcpError

PARENT = "projects/demo/locations/us-central1"
ENDPOINT = f"{PARENT}/endpoints/example"
MODEL = f"{PARENT}/models/example"
OPERATION = f"{ENDPOINT}/operations/123"


def operation(name, result):
    op = Mock()
    op.operation.name = name
    op.result.return_value = result
    return op


def mock_scope(monkeypatch):
    endpoint_client, model_client, prediction_client = Mock(), Mock(), Mock()
    clients = {
        vertex.EndpointServiceClient: endpoint_client,
        vertex.ModelServiceClient: model_client,
        vertex.PredictionServiceClient: prediction_client,
    }
    scope = SimpleNamespace(
        project="demo",
        location="us-central1",
        parent=PARENT,
        api_host="us-central1-aiplatform.googleapis.com",
        client=clients.__getitem__,
    )
    for module in (_serve, deployments):
        monkeypatch.setattr(module, "VertexScope", Mock(return_value=scope))
    return endpoint_client, model_client, prediction_client


@pytest.mark.parametrize("wait", [False, True])
def test_apply_plan_submits_typed_vertex_requests(monkeypatch, wait):
    endpoints, models, _ = mock_scope(monkeypatch)
    endpoints.create_endpoint.return_value = operation("create-op", vertex.Endpoint(name=ENDPOINT))
    models.upload_model.return_value = operation("upload-op", vertex.UploadModelResponse(model=MODEL))
    deploy = operation(OPERATION, vertex.DeployModelResponse(deployed_model={"id": "123"}))
    endpoints.deploy_model.return_value = deploy
    plan = {
        "name": "example",
        "project": "demo",
        "location": "us-central1",
        "model_id": "org/model",
        "engine": "vllm",
        "machine_type": "g2-standard-24",
        "accelerator_type": "NVIDIA_L4",
        "accelerator_count": 2,
        "min_replicas": 1,
        "max_replicas": 3,
        "service_account": None,
        "image_uri": "us-docker.pkg.dev/demo/container/server:1",
        "container_args": ["--tensor-parallel-size=2"],
        "predict_route": "/generate",
        "health_route": "/ping",
        "startup_timeout_seconds": 7200,
        "environment": {},
        "secret_environment": [],
        "use_hf_token": True,
        "wait": wait,
    }
    result = _serve.apply_plan(plan, token="hf-test")
    create = vertex.CreateEndpointRequest(endpoints.create_endpoint.call_args.kwargs["request"])
    assert create.endpoint_id == "example"
    upload = vertex.UploadModelRequest(models.upload_model.call_args.kwargs["request"])
    assert upload.model.container_spec.env[0].value == "hf-test"
    assert upload.model.container_spec.deployment_timeout.total_seconds() == 7200
    request = vertex.DeployModelRequest(endpoints.deploy_model.call_args.kwargs["request"])
    assert request.deployed_model.dedicated_resources.machine_spec.accelerator_count == 2
    assert request.deployed_model.dedicated_resources.max_replica_count == 3
    assert result["operation"] == OPERATION
    assert "hf-test" not in str(result)
    assert deploy.result.called == wait


def test_resolve_existing_hf_cloud_display_name(monkeypatch):
    endpoints, _, _ = mock_scope(monkeypatch)
    endpoints.get_endpoint.side_effect = NotFound("missing")
    endpoints.list_endpoints.return_value = [vertex.Endpoint(name=ENDPOINT, display_name="old-name")]
    manager = deployments.Deployments("demo", "us-central1")
    assert manager.resolve("old-name").name == ENDPOINT
    request = vertex.ListEndpointsRequest(endpoints.list_endpoints.call_args.kwargs["request"])
    assert request.filter == 'display_name="old-name"'


def test_prediction_body_and_replicas(monkeypatch):
    endpoints, _, prediction = mock_scope(monkeypatch)
    endpoints.get_endpoint.return_value = vertex.Endpoint(
        name=ENDPOINT,
        labels={"hf-engine": "vllm"},
        deployed_models=[
            {
                "id": "123",
                "model": MODEL,
                "dedicated_resources": {
                    "machine_spec": {"machine_type": "g2-standard-4", "accelerator_type": "NVIDIA_L4"},
                    "min_replica_count": 1,
                    "max_replica_count": 2,
                },
            }
        ],
    )
    wire_response = ParseDict({"predictions": [{"generated_text": "world"}]}, vertex.PredictResponse.pb()())
    prediction.predict.return_value = vertex.PredictResponse.wrap(wire_response)
    manager = deployments.Deployments(None, None, ENDPOINT)
    result = manager.invoke(ENDPOINT, text="hello", payload=None, raw=False, max_tokens=4)
    request = vertex.PredictRequest(prediction.predict.call_args.kwargs["request"])
    assert dict(request.instances[0]) == {"prompt": "hello", "max_tokens": 4}
    assert result["predictions"] == [{"generated_text": "world"}]
    manager.scale(ENDPOINT, 2, 3, None)
    mutate = vertex.MutateDeployedModelRequest(endpoints.mutate_deployed_model.call_args.kwargs["request"])
    assert mutate.deployed_model.id == "123"
    assert mutate.deployed_model.dedicated_resources.max_replica_count == 3
    assert list(mutate.update_mask.paths) == [
        "dedicated_resources.min_replica_count",
        "dedicated_resources.max_replica_count",
    ]


def test_wait_reports_operation_failure(monkeypatch):
    endpoints, _, _ = mock_scope(monkeypatch)
    endpoints.transport.operations_client.get_operation.return_value = Operation(
        name=OPERATION,
        done=True,
        error=Status(code=8, message="GPU quota exhausted"),
    )
    manager = deployments.Deployments(None, None, OPERATION)
    with pytest.raises(GcpError, match="GPU quota exhausted"):
        manager.wait(OPERATION, timeout=60)


def test_delete_keeps_valid_traffic_splits(monkeypatch):
    endpoints, _, _ = mock_scope(monkeypatch)
    endpoint = vertex.Endpoint(
        name=ENDPOINT,
        deployed_models=[{"id": name} for name in ("1", "2", "3")],
        traffic_split={"1": 30, "2": 30, "3": 40},
    )
    endpoints.delete_endpoint.return_value = operation(OPERATION, None)
    manager = deployments.Deployments("demo", "us-central1")
    assert manager.delete(endpoint, wait=True)["status"] == "DELETED"
    splits = [call.kwargs["traffic_split"] for call in endpoints.undeploy_model.call_args_list]
    assert splits == [{"2": 30, "3": 70}, {"3": 100}, {}]
