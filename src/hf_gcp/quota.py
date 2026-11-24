"""Read applied Vertex AI serving limits from Cloud Quotas."""

from google.auth.transport.requests import AuthorizedSession

from hf_gcp._serve import VertexScope
from hf_gcp.errors import GcpError

GPU_ALIASES = {
    "NVIDIA_TESLA_A100": ("a100",),
    "NVIDIA_A100_80GB": ("a10080gb", "a10080"),
    "NVIDIA_H100_80GB": ("h100", "h10080gb"),
    "NVIDIA_H100_MEGA_80GB": ("h100mega", "h100", "h10080gb"),
    "NVIDIA_H200_141GB": ("h200", "h200141gb"),
}


def _get(session: AuthorizedSession, url: str, **params) -> dict:
    response = session.get(url, params=params, timeout=30)
    if not response.ok:
        try:
            detail = response.json()["error"]["message"]
        except (KeyError, TypeError, ValueError):
            detail = response.reason
        raise GcpError(f"Google Cloud API HTTP {response.status_code}: {detail}")
    return response.json()


def _matches_accelerator(info: dict, expected: str) -> bool:
    identifiers = [info.get("quotaId", ""), info.get("metric", ""), info.get("metricDisplayName", "")]
    names = GPU_ALIASES.get(expected, (expected.lower().removeprefix("nvidia_").removeprefix("tesla_"),))
    normalized = "".join(c for c in " ".join(identifiers).lower() if c.isalnum())
    if expected == "NVIDIA_TESLA_A100" and "a10080" in normalized:
        return False
    return any(name in normalized for name in names)


def quotas(scope: VertexScope, accelerator_type: str | None) -> list[dict]:
    with AuthorizedSession(scope.credentials) as session:
        project_number = scope.project
        if not project_number.isdecimal():
            project = _get(session, f"https://cloudresourcemanager.googleapis.com/v3/projects/{scope.project}")
            project_number = project["name"].removeprefix("projects/")
        url = (
            f"https://cloudquotas.googleapis.com/v1/projects/{project_number}"
            "/locations/global/services/aiplatform.googleapis.com/quotaInfos"
        )
        rows = []
        params = {}
        while True:
            page = _get(session, url, **params)
            for info in page.get("quotaInfos", []):
                metric = info.get("metric", "")
                text = " ".join((metric, info.get("metricDisplayName", ""), info.get("quotaDisplayName", ""))).lower()
                if not any(term in text.replace("_", " ") for term in ("custom model serving", "online prediction")):
                    continue
                if accelerator_type and not _matches_accelerator(info, accelerator_type):
                    continue
                for entry in info.get("dimensionsInfo", []):
                    dimensions = entry.get("dimensions", {})
                    locations = entry.get("applicableLocations", [])
                    if dimensions.get("region", scope.location) != scope.location:
                        continue
                    if locations and scope.location not in locations and "global" not in locations:
                        continue
                    limit = entry.get("details", {}).get("value")
                    region = dimensions.get("region") or (scope.location if scope.location in locations else None)
                    rows.append(
                        {
                            "quota_id": info["quotaId"],
                            "metric": metric,
                            "name": info.get("quotaDisplayName"),
                            "limit": int(limit) if limit is not None else None,
                            "region": region,
                            "dimensions": dimensions,
                            "applicable_locations": locations,
                        }
                    )
            next_page = page.get("nextPageToken")
            if not next_page:
                return sorted(rows, key=lambda row: row["quota_id"])
            params = {"pageToken": next_page}
