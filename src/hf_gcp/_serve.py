"""Vertex AI deployment runtime, also included verbatim in exported scripts."""

import os
import re

import google.auth
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import aiplatform_v1 as vertex
from huggingface_hub import get_token

DEFAULT_LOCATION = "us-central1"
MANAGED_LABEL = "hf-gcp"
CLOUD_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


class VertexScope:
    def __init__(self, project=None, location=None):
        self.location = location or default_location()
        self.project = project or default_project()
        validate_location(self.location)
        if self.project:
            validate_project(self.project)
        try:
            self.credentials, adc_project = google.auth.default(scopes=[CLOUD_SCOPE])
        except DefaultCredentialsError as error:
            raise RuntimeError("Configure Google ADC with gcloud auth application-default login") from error
        self.project = self.project or adc_project
        if not self.project:
            raise ValueError("Set --project or GOOGLE_CLOUD_PROJECT")
        validate_project(self.project)
        self.parent = f"projects/{self.project}/locations/{self.location}"
        self.api_host = f"{self.location}-aiplatform.googleapis.com"

    def client(self, client_type):
        return client_type(credentials=self.credentials, client_options={"api_endpoint": self.api_host})


def validate_project(project):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9.:-]*", project):
        raise ValueError("Use a Google Cloud project ID or number")


def validate_location(location):
    if not re.fullmatch(r"[a-z]+(?:-[a-z]+)+[0-9]+", location):
        raise ValueError("Use a Vertex AI region such as us-central1")


def default_project():
    return os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")


def default_location():
    return os.environ.get("GOOGLE_CLOUD_LOCATION") or os.environ.get("CLOUD_ML_REGION") or DEFAULT_LOCATION


def deployment_environment(plan, token):
    env = dict(plan["environment"])
    for key in plan["secret_environment"]:
        if key not in env:
            value = os.environ.get(key)
            if value is None:
                raise ValueError(f"Set {key} in the environment to run this deployment")
            env[key] = value
    if plan["use_hf_token"]:
        resolved_token = token or get_token()
        if not resolved_token:
            raise ValueError("Set HF_TOKEN or run hf auth login for this model")
        env["HF_TOKEN"] = resolved_token
    return env


def apply_plan(plan, *, token=None):
    env = deployment_environment(plan, token)
    scope = VertexScope(plan["project"], plan["location"])
    endpoints = scope.client(vertex.EndpointServiceClient)
    models = scope.client(vertex.ModelServiceClient)
    name = plan["name"]
    labels = {MANAGED_LABEL: "true", "hf-engine": plan["engine"]}
    progress = []
    try:
        # An explicit endpoint ID lets Vertex reject concurrent creation before model upload.
        create = endpoints.create_endpoint(
            request={
                "parent": scope.parent,
                "endpoint_id": name,
                "endpoint": {"display_name": name, "labels": labels},
            },
            timeout=60,
        )
        progress.append(f"create operation {create.operation.name}")
        endpoint = create.result(timeout=600)
        progress.append(f"endpoint {endpoint.name}")
        container = {
            "image_uri": plan["image_uri"],
            "args": plan["container_args"],
            "env": [{"name": key, "value": value} for key, value in env.items()],
            "ports": [{"container_port": 8080}],
            "predict_route": plan["predict_route"],
            "health_route": plan["health_route"],
            "deployment_timeout": {"seconds": plan["startup_timeout_seconds"]},
        }
        if plan["engine"] == "vllm":
            container["shared_memory_size_mb"] = 16 * 1024
        upload = models.upload_model(
            request={
                "parent": scope.parent,
                "model": {"display_name": name, "labels": labels, "container_spec": container},
            },
            timeout=60,
        )
        progress.append(f"upload operation {upload.operation.name}")
        model = upload.result(timeout=600)
        progress.append(f"model {model.model}")
        machine = {"machine_type": plan["machine_type"]}
        if plan["accelerator_type"]:
            machine.update(accelerator_type=plan["accelerator_type"], accelerator_count=plan["accelerator_count"])
        deployed = {
            "display_name": plan["model_id"][:128],
            "model": model.model,
            "dedicated_resources": {
                "machine_spec": machine,
                "min_replica_count": plan["min_replicas"],
                "max_replica_count": plan["max_replicas"],
            },
        }
        if plan["service_account"]:
            deployed["service_account"] = plan["service_account"]
        operation = endpoints.deploy_model(
            request={"endpoint": endpoint.name, "deployed_model": deployed, "traffic_split": {"0": 100}},
            timeout=60,
        )
        progress.append(f"deploy operation {operation.operation.name}")
        result = {
            "name": name,
            "endpoint": endpoint.name,
            "model": model.model,
            "model_id": plan["model_id"],
            "machine_type": plan["machine_type"],
            "location": scope.location,
            "operation": operation.operation.name,
            "status": "DEPLOYING",
        }
        if plan["wait"]:
            response = operation.result(timeout=plan["startup_timeout_seconds"] + 1800)
            result.update(status="DEPLOYED", deployed_model_id=response.deployed_model.id)
        return result
    except Exception as error:
        message = str(error)
        for key in [*plan["secret_environment"], "HF_TOKEN"]:
            if env.get(key):
                message = message.replace(env[key], "[redacted]")
        if progress:
            message += ". Vertex resources and operations created so far: " + ", ".join(progress)
        raise RuntimeError(message) from None
