"""Vertex AI endpoint lifecycle and inference."""

from __future__ import annotations

import json
import re
from time import monotonic, sleep

from google.api.httpbody_pb2 import HttpBody
from google.api_core.exceptions import NotFound
from google.cloud import aiplatform_v1 as vertex
from google.cloud import logging as cloud_logging
from google.protobuf.json_format import MessageToDict, ParseDict

from hf_gcp._serve import MANAGED_LABEL, VertexScope
from hf_gcp.errors import GcpError

SCOPE = r"projects/(?P<project>[^/]+)/locations/(?P<location>[^/]+)"
ENDPOINT = re.compile(rf"{SCOPE}/endpoints/[^/]+")
OPERATION = re.compile(rf"{SCOPE}/(?:endpoints/[^/]+/)?operations/[^/]+")
MANAGED_FILTER = f'labels.{MANAGED_LABEL}="true" OR labels.hf-cloud="true"'


def endpoint_summary(endpoint: vertex.Endpoint) -> dict:
    managed = endpoint.labels.get(MANAGED_LABEL) == "true"
    models = []
    for model in endpoint.deployed_models:
        resources = model.dedicated_resources
        spec = resources.machine_spec
        models.append(
            {
                "id": model.id,
                "model": model.model,
                "model_id": model.display_name if managed else None,
                "machine_type": spec.machine_type or None,
                "accelerator": spec.accelerator_type.name if spec.accelerator_type else None,
                "gpus": spec.accelerator_count,
                "min_replicas": resources.min_replica_count or None,
                "max_replicas": resources.max_replica_count or None,
                "traffic_percent": endpoint.traffic_split.get(model.id, 0),
            }
        )
    return {
        "name": endpoint.display_name,
        "endpoint": endpoint.name,
        "status": "DEPLOYED" if models else "EMPTY",
        "engine": endpoint.labels.get("hf-engine"),
        "created_at": endpoint.create_time,
        "updated_at": endpoint.update_time,
        "models": models,
    }


def poll_until(check, timeout: int, description: str):
    deadline = monotonic() + timeout
    while True:
        result = check()
        if result is not None:
            return result
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise GcpError(f"Timed out waiting for {description}")
        sleep(min(10, remaining))


class Deployments:
    def __init__(self, project: str | None, location: str | None, reference: str | None = None):
        if reference and reference.startswith("projects/"):
            match = ENDPOINT.fullmatch(reference) or OPERATION.fullmatch(reference)
            if not match:
                raise GcpError("Use a full Vertex AI endpoint or operation resource name")
            if location and location != match["location"]:
                raise GcpError("--location differs from the resource name")
            project, location = match["project"], match["location"]
        self.scope = VertexScope(project, location)
        self.client = self.scope.client(vertex.EndpointServiceClient)

    def resolve(self, reference: str) -> vertex.Endpoint:
        if ENDPOINT.fullmatch(reference):
            return self.client.get_endpoint(name=reference, timeout=30)
        # Prefer the endpoint ID for extension-managed resources and numeric IDs.
        if re.fullmatch(r"[a-z0-9-]+", reference):
            try:
                return self.client.get_endpoint(name=f"{self.scope.parent}/endpoints/{reference}", timeout=30)
            except NotFound:
                pass
        filter_ = f"display_name={json.dumps(reference, ensure_ascii=False)}"
        matches = list(
            self.client.list_endpoints(request={"parent": self.scope.parent, "filter": filter_}, timeout=30)
        )
        if not matches:
            raise GcpError(f"Endpoint {reference!r} was not found in {self.scope.parent}")
        if len(matches) > 1:
            raise GcpError(f"Multiple endpoints named {reference!r}. Use a full endpoint resource name")
        return matches[0]

    def list(self, *, all_endpoints: bool) -> list[dict]:
        endpoints = self.client.list_endpoints(
            request={"parent": self.scope.parent, "filter": "" if all_endpoints else MANAGED_FILTER}, timeout=30
        )
        return [endpoint_summary(endpoint) for endpoint in endpoints]

    def describe(self, reference: str) -> dict:
        endpoint = self.resolve(reference)
        if endpoint.dedicated_endpoint_dns:
            url = f"https://{endpoint.dedicated_endpoint_dns}/v1/{endpoint.name}:predict"
        else:
            url = f"https://{self.scope.api_host}/v1/{endpoint.name}:predict"
        return {**endpoint_summary(endpoint), "predict_url": url, "labels": dict(endpoint.labels)}

    def wait(self, reference: str, timeout: int) -> dict:
        if OPERATION.fullmatch(reference):
            operations = self.client.transport.operations_client

            def check():
                operation = operations.get_operation(reference, timeout=30)
                if not operation.done:
                    return None
                if operation.HasField("error"):
                    raise GcpError(f"Vertex operation {reference} failed: {operation.error.message}")
                return {"operation": reference, "status": "SUCCEEDED"}

            return poll_until(check, timeout, reference)

        name = self.resolve(reference).name

        def check():
            endpoint = self.client.get_endpoint(name=name, timeout=30)
            return endpoint_summary(endpoint) if endpoint.deployed_models else None

        return poll_until(check, timeout, f"a model to deploy on {name}")

    def logs(self, reference: str, tail: int) -> list[str]:
        endpoint = self.resolve(reference)
        endpoint_id = endpoint.name.rsplit("/", 1)[-1]
        client = cloud_logging.Client(project=self.scope.project, credentials=self.scope.credentials)
        entries = client.list_entries(
            filter_=(
                'resource.type="aiplatform.googleapis.com/Endpoint" '
                f'AND resource.labels.endpoint_id="{endpoint_id}" '
                f'AND resource.labels.location="{self.scope.location}"'
            ),
            order_by=cloud_logging.DESCENDING,
            max_results=tail,
        )
        lines = []
        for entry in entries:
            timestamp = entry.timestamp.isoformat() if entry.timestamp else ""
            payload = entry.payload if isinstance(entry.payload, str) else json.dumps(entry.payload, default=str)
            lines.append(f"{timestamp} {payload}".strip())
        return list(reversed(lines))

    def invoke(self, reference: str, *, text: str | None, payload: dict | None, raw: bool, max_tokens: int) -> object:
        endpoint = self.resolve(reference)
        if payload is None:
            if endpoint.labels.get("hf-engine") == "vllm":
                prompt = {"prompt": text, "max_tokens": max_tokens}
                payload = prompt if raw else {"instances": [prompt]}
            else:
                item = {"inputs": text}
                payload = item if raw else {"instances": [item]}
        client = self.scope.client(vertex.PredictionServiceClient)
        metadata = ()
        if endpoint.dedicated_endpoint_dns:
            client = vertex.PredictionServiceClient(
                credentials=self.scope.credentials, client_options={"api_endpoint": endpoint.dedicated_endpoint_dns}
            )
            metadata = (("x-vertex-ai-endpoint-id", endpoint.name.rsplit("/", 1)[-1]),)
        if raw:
            response = client.raw_predict(
                endpoint=endpoint.name,
                http_body=HttpBody(content_type="application/json", data=json.dumps(payload).encode()),
                metadata=metadata,
                timeout=300,
            )
            return json.loads(response.data)
        instances = payload.get("instances")
        if not isinstance(instances, list) or not instances:
            raise GcpError("Prediction payload must contain a nonempty instances array. Use --raw for container JSON")
        payload = {key: payload[key] for key in ("instances", "parameters") if key in payload}
        wire_request = ParseDict({"endpoint": endpoint.name, **payload}, vertex.PredictRequest.pb()())
        response = client.predict(request=vertex.PredictRequest.wrap(wire_request), metadata=metadata, timeout=300)
        return MessageToDict(vertex.PredictResponse.pb(response), preserving_proto_field_name=True)

    def scale(self, reference: str, minimum: int, maximum: int, model_id: str | None) -> dict:
        endpoint = self.resolve(reference)
        matches = [model for model in endpoint.deployed_models if model_id is None or model.id == model_id]
        if len(matches) != 1:
            raise GcpError("Select a deployed model with --deployed-model-id. See deployments describe for IDs")
        model = matches[0]
        if not model.dedicated_resources.machine_spec.machine_type:
            raise GcpError(f"Model {model.id} does not use dedicated resources")
        operation = self.client.mutate_deployed_model(
            request={
                "endpoint": endpoint.name,
                "deployed_model": {
                    "id": model.id,
                    "dedicated_resources": {"min_replica_count": minimum, "max_replica_count": maximum},
                },
                "update_mask": {
                    "paths": ["dedicated_resources.min_replica_count", "dedicated_resources.max_replica_count"]
                },
            },
            timeout=60,
        )
        operation.result(timeout=7200)
        return self.describe(endpoint.name)

    def delete(self, endpoint: vertex.Endpoint, *, wait: bool) -> dict:
        # Keep one receiving deployment until last so Vertex always has a valid traffic split.
        models = sorted(endpoint.deployed_models, key=lambda model: bool(endpoint.traffic_split.get(model.id)))
        traffic = dict(endpoint.traffic_split)
        for model in models:
            share = traffic.pop(model.id, 0)
            if share and model.id != models[-1].id:
                replacement = models[-1].id
                traffic[replacement] = traffic.get(replacement, 0) + share
            self.client.undeploy_model(
                endpoint=endpoint.name, deployed_model_id=model.id, traffic_split=traffic.copy(), timeout=60
            ).result(timeout=3600)
        operation = self.client.delete_endpoint(name=endpoint.name, timeout=60)
        result = {"endpoint": endpoint.name, "operation": operation.operation.name, "status": "DELETING"}
        if wait:
            operation.result(timeout=3600)
            result["status"] = "DELETED"
        return result
