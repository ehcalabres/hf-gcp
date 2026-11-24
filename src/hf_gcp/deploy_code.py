"""Export the resolved plan and its Vertex AI runtime as standalone Python."""

from importlib.resources import files
from pprint import pformat

from hf_gcp.planning import DeploymentPlan


def deployment_code(plan: DeploymentPlan) -> str:
    runtime = files("hf_gcp").joinpath("_serve.py").read_text(encoding="utf-8")
    return (
        "# Install: pip install 'google-cloud-aiplatform>=1.135,<3' 'huggingface-hub>=1.3.4,<3'\nimport json\n"
        f"{runtime}\n\n"
        f"PLAN = {pformat(plan.public(), sort_dicts=False)}\n\n"
        'if __name__ == "__main__":\n'
        "    try:\n"
        "        print(json.dumps(apply_plan(PLAN), indent=2))\n"
        "    except (ValueError, RuntimeError) as error:\n"
        "        raise SystemExit(str(error)) from None\n"
    )
