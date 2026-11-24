from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

from hf_gcp import quota


def test_quota_reads_regional_gpu_limits_across_pages(monkeypatch):
    session = MagicMock()
    session.return_value.__enter__.return_value = Mock()
    monkeypatch.setattr(quota, "AuthorizedSession", session)
    l4 = {
        "quotaId": "CustomModelServingNvidiaL4Gpus",
        "metric": "aiplatform.googleapis.com/custom_model_serving_nvidia_l4_gpus",
        "quotaDisplayName": "Custom model serving Nvidia L4 GPUs per region",
        "dimensionsInfo": [
            {"dimensions": {"region": "us-central1"}, "details": {"value": "12"}},
            {"dimensions": {"region": "us-east1"}, "details": {"value": "4"}},
        ],
    }
    get = Mock(side_effect=[{"name": "projects/123"}, {"nextPageToken": "two"}, {"quotaInfos": [l4]}])
    monkeypatch.setattr(quota, "_get", get)
    scope = SimpleNamespace(project="demo", location="us-central1", credentials=object())
    result = quota.quotas(scope, "NVIDIA_L4")
    assert len(result) == 1
    assert result[0]["limit"] == 12
    assert result[0]["region"] == "us-central1"
    expected_url = "https://cloudquotas.googleapis.com/v1/projects/123/locations/global/services/aiplatform.googleapis.com/quotaInfos"
    assert get.call_args_list[-1] == call(session.return_value.__enter__.return_value, expected_url, pageToken="two")
