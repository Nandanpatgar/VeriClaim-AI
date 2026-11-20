
import pytest

import source_reliability as sr


def test_domain_extraction():

    result = sr.get_source_domain(
        "https://www.bbc.com/news"
    )

    assert result == "bbc.com"


def test_subdomains_group_under_same_publisher_domain():
    assert sr.get_source_domain("https://en.wikipedia.org/wiki/Claim") == "wikipedia.org"
    assert sr.get_source_domain("https://news.bbc.co.uk/article") == "bbc.co.uk"


def test_invalid_url():

    result = sr.get_source_domain("")

    assert result == ""


def test_government_source_classification():

    result = sr.classify_source_type(
        "https://www.nasa.gov/"
    )

    assert result == "government"


def test_academic_source_classification():

    result = sr.classify_source_type(
        "https://www.harvard.edu/"
    )

    assert result == "academic"


def test_encyclopedia_classification():

    result = sr.classify_source_type(
        "https://en.wikipedia.org/wiki/Einstein"
    )

    assert result == "encyclopedia"


def test_https_validation():

    assert sr.uses_https(
        "https://example.com"
    )

    assert not sr.uses_https(
        "http://example.com"
    )


def test_reliability_score_range():

    score = sr.calculate_reliability_score(
        "https://www.nasa.gov/"
    )

    assert 0.0 <= score <= 1.0


def test_trusted_source():

    assert sr.is_trusted_source(0.85)

    assert not sr.is_trusted_source(0.40)


def test_single_source_evaluation():

    evidence = {
        "evidence_id": 1,
        "source": "Wikipedia",
        "content": "Einstein was a physicist.",
        "url": "https://en.wikipedia.org/wiki/Einstein"
    }

    result = sr.evaluate_single_source(evidence)

    assert "reliability_score" in result

    assert "domain" in result

    assert "source_type" in result

    assert "is_trusted" in result


def test_multiple_source_evaluation():

    evidence = [
        {
            "content": "Evidence one.",
            "url": "https://www.nasa.gov/"
        },
        {
            "content": "Evidence two.",
            "url": "https://en.wikipedia.org/wiki/Einstein"
        }
    ]

    result = sr.evaluate_source_reliability(evidence)

    assert len(result) == 2

    assert all(
        "reliability_score" in item
        for item in result
    )


def test_source_diversity():

    evidence = [
        {
            "url": "https://www.bbc.com/news/one"
        },
        {
            "url": "https://www.bbc.com/news/two"
        },
        {
            "url": "https://www.nasa.gov/article"
        }
    ]

    result = sr.calculate_source_diversity(evidence)

    assert result["unique_domains"] == 2


def test_duplicate_source_removal():

    evidence = [
        {
            "url": "https://example.com/article"
        },
        {
            "url": "https://example.com/article"
        },
        {
            "url": "https://another.com/article"
        }
    ]

    result = sr.remove_duplicate_sources(evidence)

    assert len(result) == 2


def test_complete_source_analysis():

    evidence = [
        {
            "source": "NASA",
            "content": "Evidence one.",
            "url": "https://www.nasa.gov/"
        },
        {
            "source": "Wikipedia",
            "content": "Evidence two.",
            "url": "https://en.wikipedia.org/wiki/Einstein"
        }
    ]

    result = sr.analyze_sources(evidence)

    assert result["total_sources"] == 2

    assert result["unique_domains"] == 2

    assert 0.0 <= result["average_reliability"] <= 1.0


def test_empty_evidence():

    result = sr.analyze_sources([])

    assert result["total_sources"] == 0

    assert result["trusted_sources"] == 0

    assert result["unique_domains"] == 0

    assert result["average_reliability"] == 0.0


def test_invalid_evidence_input():

    with pytest.raises(ValueError):

        sr.analyze_sources("invalid")


def test_empty_claims():

    result = sr.evaluate_claim_sources({
        "claims": []
    })

    assert result == {
        "claims": []
    }


def test_invalid_claim_input():

    with pytest.raises(ValueError):

        sr.evaluate_claim_sources("invalid")


def test_claim_source_evaluation():

    data = {
        "claims": [
            {
                "claim_id": 1,
                "claim": "Einstein was a physicist.",
                "evidence": [
                    {
                        "source": "NASA",
                        "content": "Evidence.",
                        "url": "https://www.nasa.gov/"
                    }
                ]
            }
        ]
    }

    result = sr.evaluate_claim_sources(data)

    claim = result["claims"][0]

    assert "source_analysis" in claim

    assert len(claim["evidence"]) == 1

    assert "reliability_score" in claim["evidence"][0]
