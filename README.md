# hf-gcp

`hf-gcp` is a Hugging Face CLI extension for planning and deploying Hub
models on Google Cloud Vertex AI online prediction endpoints. It extracts the
Vertex AI workflow from [hf-cloud](https://github.com/ehcalabres/hf-cloud) and
follows the command structure of [hf-aws](https://github.com/ehcalabres/hf-aws)
and [hf-msft-foundry](https://github.com/alvarobartt/hf-msft-foundry).

## Installation

```bash
hf extensions install ehcalabres/hf-gcp
hf gcp --help
```

From a checkout, run `uv run hf-gcp --help` or install with `uv tool install .`
and run `hf-gcp`. Python 3.10 or newer is required. Memory estimation uses
`uvx` to run a pinned hf-mem worker. Deployments with an explicit machine type
can run without the worker.

## Authentication

Authenticate with Application Default Credentials (ADC) and select your project.

```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=my-project
export GOOGLE_CLOUD_LOCATION=us-central1
```

The CLI accepts `--project` and `--location` at any command level. The project
defaults to `GOOGLE_CLOUD_PROJECT` or the ADC project, and the location defaults
to `GOOGLE_CLOUD_LOCATION` or `us-central1`. Enable the Vertex AI API for
deployments, Cloud Logging for logs, and the Cloud Quotas API for quota
lookups. With a project ID, quota lookup also uses Cloud Resource Manager to
resolve the project number. You can pass the project number directly.

For gated or private models, accept the model's terms on the Hub and use
`hf auth login`, `HF_TOKEN`, or `--token`. Vertex AI receives the token in the
model's container environment. Use a read token scoped to the required models
and restrict access to the model registry. To run the serving container as a
specific service account, pass `--service-account`. The caller needs permission
to act as that account.

## Plan a deployment

```bash
hf gcp models list --search gpt2
hf gcp models openai-community/gpt2
hf gcp instances g2-standard-4 --instances 2 --hours 24
hf gcp estimate openai-community/gpt2 --max-model-len 1024 --instances 2 --hours 24
hf gcp quota --machine-type g2-standard-4
```

`estimate` runs hf-mem across distinct GPU layouts and ranks matching machines
by price. It accepts Hub IDs, URLs, local paths, and weight selectors supported
by hf-mem. Deployment takes a Hub model ID. Use `--accelerator-type`,
`--accelerator-count`, or `--tensor-parallel-size` to narrow the estimate.

The bundled catalog uses public on-demand USD rates for Vertex AI online
prediction in Iowa (`us-central1`). Rates cover one machine with all its GPUs.
Estimated compute cost multiplies that rate by replica count and hours. Check
current regional rates before deploying. [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing#prediction-and-explanation)

Quota shows applied limits returned by Cloud Quotas. Google reserves compute
quota against the maximum replica count on autoscaling deployments. Check
runtime compatibility and regional capacity for the selected machine.
[Cloud Quotas](https://docs.cloud.google.com/docs/quotas/view-manage) ·
[Vertex AI autoscaling](https://cloud.google.com/vertex-ai/docs/predictions/autoscaling)

```bash
hf gcp deploy openai-community/gpt2 --name gpt2-demo --machine-type g2-standard-4 \
  --env HF_TASK=text-generation --max-model-len 1024 --dry-run

hf gcp deploy openai-community/gpt2 --name gpt2-demo --machine-type g2-standard-4 \
  --env HF_TASK=text-generation --max-model-len 1024 --print-code > deploy.py
```

Both previews resolve the same plan used by deploy and make no Google Cloud
API calls. Set `HF_TASK` and specify the machine type to skip Hub metadata
lookup and hf-mem. Credential-like values are omitted from previews. Supply
those values through your environment when running exported code.

Without `--machine-type`, deployment selects the cheapest priced GPU memory fit
from the catalog. The default engine is vLLM for GPU text generation and the
Hugging Face PyTorch inference container for other tasks or an explicit CPU
machine. vLLM defaults tensor parallelism to the machine's GPU count. Use
`--tensor-parallel-size` to override it and `--max-model-len` to set the context
length. Override the selected engine or image with `--engine` or `--image-uri`.
Use repeatable `--env KEY=VALUE` and `--env-file PATH` for container settings.
Inline values override files.
[Model Garden vLLM](https://cloud.google.com/vertex-ai/generative-ai/docs/open-models/vllm/use-vllm) ·
[Hugging Face inference containers](https://huggingface.co/docs/google-cloud/containers/available)

## Deploy and manage an endpoint

```bash
hf gcp deploy openai-community/gpt2 --name gpt2-demo --machine-type g2-standard-4 \
  --max-model-len 1024
hf gcp deployments list
hf gcp deployments status gpt2-demo
hf gcp deployments describe gpt2-demo
hf gcp deployments invoke gpt2-demo --input "Hello world"
hf gcp deployments logs gpt2-demo --tail 20
hf gcp deployments scale gpt2-demo --min-replicas 1 --max-replicas 2
hf gcp deployments delete gpt2-demo --yes --wait
```

Deploy waits for Vertex AI by default. `--no-wait` returns once the model
deployment has been submitted. Pass its full operation name to
`deployments wait` to see whether it succeeded. You can also wait for a
model to appear with `deployments wait ENDPOINT`.

List includes endpoints created by `hf-gcp` and `hf-cloud`. Add `--all` for
all endpoints in the selected region. Management commands accept an endpoint
ID, display name, or full resource name. Use the full name when display names
are ambiguous. For custom requests, pass `--payload @request.json` and add
`--raw` when the container expects a body without Vertex's `instances`
wrapper.

Deleting an endpoint undeploys its models to release serving compute. Model
Registry entries have a separate lifecycle. If a deployment fails partway
through, the error includes available resource and operation names for
inspection and cleanup.

The terminal view compares GPU memory and cost, highlights the cheapest
compatible machine, and shows endpoint status and deployed models. Pass
`--json` or `-o json` at any command level for machine-readable output. Errors
go to stderr in the selected format. Dry runs emit JSON and code export emits
standalone Python.

## Development

```bash
uv sync --locked
uv run ruff check
uv run ruff format --check
uv run pytest
uv build
```

Tests stub cloud operations. The [agent skill](skills/hf-gcp/SKILL.md)
documents the planning and deployment workflow.
