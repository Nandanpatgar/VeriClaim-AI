import pytest
from fastapi import HTTPException

import backend_api
from main import PipelineError


def test_health_endpoint_reports_healthy():
    result = backend_api.health_check()

    assert result["status"] == "healthy"
    assert result["application"] == "Veri-Claim-AI"


def test_verify_endpoint_wraps_pipeline_result(monkeypatch):
    pipeline_result = {"status": "success", "final_decisions": {"decisions": []}}
    monkeypatch.setattr(backend_api, "run_pipeline", lambda text: pipeline_result)

    result = backend_api.verify_text(
        backend_api.VerificationRequest(text="A factual statement for testing.")
    )

    assert result["status"] == "success"
    assert result["data"] == pipeline_result


def test_retrieval_failure_is_returned_as_service_unavailable(monkeypatch):
    def fail_pipeline(_text):
        raise PipelineError("Evidence Retrieval", "All providers are offline.")

    monkeypatch.setattr(backend_api, "run_pipeline", fail_pipeline)

    with pytest.raises(HTTPException) as error:
        backend_api.verify_text(
            backend_api.VerificationRequest(text="A factual statement for testing.")
        )

    assert error.value.status_code == 503
    assert error.value.detail["stage"] == "Evidence Retrieval"
