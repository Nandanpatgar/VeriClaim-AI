
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
