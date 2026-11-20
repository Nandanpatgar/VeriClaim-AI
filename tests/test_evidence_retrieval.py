
import pytest
import evidence_retrieval as er


# ==================================================
# MOCK RESPONSE
# ==================================================

class MockResponse:

    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


# ==================================================
# WIKIPEDIA TESTS
# ==================================================

def test_wikipedia_retrieval(monkeypatch):

    mock_data = {
        "query": {
            "pages": [
                {
                    "title": "Albert Einstein",
                    "extract": "Albert Einstein was a physicist.",
                    "fullurl": "https://en.wikipedia.org/wiki/Albert_Einstein"
                }
            ]
        }
    }

    def mock_get(*args, **kwargs):
        return MockResponse(mock_data)

    monkeypatch.setattr(er.requests, "get", mock_get)

    result = er.retrieve_wikipedia_evidence("Albert Einstein")

    assert len(result) == 1
    assert result[0]["source"] == "Wikipedia"
    assert result[0]["title"] == "Albert Einstein"


def test_wikipedia_empty_results(monkeypatch):

    def mock_get(*args, **kwargs):
        return MockResponse({"query": {"pages": []}})

    monkeypatch.setattr(er.requests, "get", mock_get)

    result = er.retrieve_wikipedia_evidence("Unknown topic")

    assert result == []


def test_wikipedia_network_failure_is_not_reported_as_no_evidence(monkeypatch):
    def fail_request(*args, **kwargs):
        raise er.requests.ConnectionError("offline")

    monkeypatch.setattr(er.requests, "get", fail_request)

    with pytest.raises(er.EvidenceRetrievalError, match="Wikipedia retrieval failed"):
        er.retrieve_wikipedia_evidence("Unknown topic")


def test_wikipedia_api_error_payload_is_not_treated_as_empty_results(monkeypatch):
    monkeypatch.setattr(er.requests, "get", lambda *args, **kwargs: MockResponse({"error": "upstream"}))

    with pytest.raises(er.EvidenceRetrievalError, match="without a valid query/pages"):
        er.retrieve_wikipedia_evidence("Unknown topic")


# ==================================================
# TAVILY TESTS
# ==================================================

def test_tavily_retrieval(monkeypatch):

    mock_data = {
        "results": [
            {
                "title": "Einstein Biography",
                "content": "Einstein developed relativity.",
                "url": "https://example.com/einstein",
                "score": 0.92
            }
        ]
    }

    def mock_post(*args, **kwargs):
        return MockResponse(mock_data)

    monkeypatch.setattr(er.requests, "post", mock_post)

    result = er.retrieve_tavily_evidence(
        "Albert Einstein",
        api_key="test-key"
    )

    assert len(result) == 1
    assert result[0]["source"] == "Tavily"
    assert result[0]["score"] == 0.92


def test_tavily_missing_api_key():

    result = er.retrieve_tavily_evidence(
        "Albert Einstein",
        api_key=""
    )

    assert result == []


def test_tavily_malformed_success_payload_is_reported(monkeypatch):
    monkeypatch.setattr(er.requests, "post", lambda *args, **kwargs: MockResponse({"error": "invalid key"}))

    with pytest.raises(er.EvidenceRetrievalError, match="without a valid results list"):
        er.retrieve_tavily_evidence("A claim", api_key="test-key")


# ==================================================
# DUPLICATE REMOVAL TEST
# ==================================================

def test_duplicate_evidence_removal():

    evidence = [
        {
            "source": "Wikipedia",
            "url": "https://example.com"
        },
        {
            "source": "Tavily",
            "url": "https://example.com"
        },
        {
            "source": "Tavily",
            "url": "https://another.com"
        }
    ]

    result = er.remove_duplicate_evidence(evidence)

    assert len(result) == 2


# ==================================================
# COMBINED RETRIEVAL TEST
# ==================================================

def test_combined_evidence_retrieval(monkeypatch):

    wikipedia_mock = [
        {
            "source": "Wikipedia",
            "title": "Albert Einstein",
            "content": "Einstein was a physicist.",
            "url": "https://en.wikipedia.org/wiki/Albert_Einstein",
            "score": None
        }
    ]

    tavily_mock = [
        {
            "source": "Tavily",
            "title": "Einstein Biography",
            "content": "Einstein developed relativity.",
            "url": "https://example.com/einstein",
            "score": 0.95
        }
    ]

    monkeypatch.setattr(
        er,
        "retrieve_wikipedia_evidence",
        lambda claim: wikipedia_mock
    )

    monkeypatch.setattr(
        er,
        "retrieve_tavily_evidence",
        lambda claim: tavily_mock
    )

    claims = {
        "claims": [
            {
                "claim_id": 1,
                "claim": "Albert Einstein developed relativity.",
                "source_text": "Einstein developed relativity."
            }
        ]
    }

    result = er.retrieve_evidence(claims)

    assert len(result["claims"]) == 1
    assert result["claims"][0]["total_evidence"] == 2
    assert result["claims"][0]["evidence"][0]["evidence_id"] == 1


# ==================================================
# INVALID INPUT TEST
# ==================================================

def test_invalid_input():

    with pytest.raises(ValueError):
        er.retrieve_evidence("invalid input")


# ==================================================
# EMPTY CLAIMS TEST
# ==================================================

def test_empty_claims():

    result = er.retrieve_evidence({"claims": []})

    assert result == {"claims": []}


def test_partial_provider_failure_is_exposed(monkeypatch):
    def fail_wikipedia(claim):
        raise er.EvidenceRetrievalError("Wikipedia retrieval failed: offline")

    monkeypatch.setattr(er, "retrieve_wikipedia_evidence", fail_wikipedia)
    monkeypatch.setattr(
        er,
        "retrieve_tavily_evidence",
        lambda claim, **kwargs: [{
            "source": "Tavily",
            "title": "Example",
            "content": "Evidence text.",
            "url": "https://example.com/article"
        }]
    )

    evidence, status = er.retrieve_evidence_for_claim(
        "A test claim.", api_key="test-key", include_status=True
    )

    assert len(evidence) == 1
    assert status["state"] == "partial"
    assert status["wikipedia"] == "failed"


def test_all_configured_provider_failures_raise_error(monkeypatch):
    def fail_provider(*args, **kwargs):
        raise er.EvidenceRetrievalError("provider offline")

    monkeypatch.setattr(er, "retrieve_wikipedia_evidence", fail_provider)
    monkeypatch.setattr(er, "retrieve_tavily_evidence", fail_provider)

    with pytest.raises(er.EvidenceRetrievalError, match="All configured evidence providers failed"):
        er.retrieve_evidence_for_claim("A test claim.", api_key="test-key")
