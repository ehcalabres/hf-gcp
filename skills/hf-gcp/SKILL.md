---
name: hf-gcp
description: Plan, deploy, and manage Hugging Face Hub models on Google Cloud Vertex AI using the hf gcp CLI extension.
---

Use `hf gcp` if installed as an extension, or `hf-gcp` if installed as a
standalone tool. Use `-o json` to read structured output.

1. Find a Hub model with `hf gcp models list --search QUERY`.
2. Compare memory fit and snapshot compute cost with
   `hf gcp estimate MODEL --max-model-len TOKENS`.
3. Check the target project's applied limits with
   `hf gcp quota --project PROJECT --location REGION --machine-type MACHINE`.
   Memory fit and quota do not establish regional capacity.
4. Preview with `hf gcp deploy MODEL --name NAME --project PROJECT --location REGION
   --machine-type MACHINE --dry-run`. Substitute `--print-code` to export the SDK
   script. Set `--env HF_TASK=TASK` to preview without Hub metadata lookup.
5. Run deploy with the same arguments. With `--no-wait`, pass the returned
   operation name to `hf gcp deployments wait OPERATION`.
6. Check the endpoint with `hf gcp deployments describe ENDPOINT`. Test with
   `deployments invoke ENDPOINT --input TEXT` or `--payload @request.json`. Read
   the serving logs with `deployments logs ENDPOINT`.
7. To release serving compute when finished, use
   `hf gcp deployments delete ENDPOINT --yes --wait`. The model registry entry
   can be managed separately.

Use Google Application Default Credentials for API calls. For gated models,
accept the model's terms on the Hub and use a scoped HF read token.
