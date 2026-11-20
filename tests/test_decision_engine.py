
import pytest

from decision_engine import (
    decide_claim,
    decide_claims,
    filter_strong_evidence,
    get_source_domain
)


# ============================================================
# TEST DATA HELPERS
# ============================================================

def make_evidence(
    evidence_id,
    domain,
    label,
    confidence=0.95,
    similarity=0.80
):

    return {

        "evidence_id": evidence_id,

        "source": domain,

        "title": "Test evidence",

        "url": f"https://{domain}/article",

        "evidence_text": "Test evidence passage.",

        "nli_label": label,

        "nli_confidence": confidence,

        "similarity_score": similarity

    }


def make_claim(evidence):

    return {

        "claim_id": 1,

        "claim": "The Earth revolves around the Sun.",

        "verification_results": evidence

    }


# ============================================================
# TEST 1: SUPPORTED CLAIM
# ============================================================

def test_supported_claim():

    evidence = [

        make_evidence(1, "source1.com", "entailment"),

        make_evidence(2, "source2.com", "entailment")

    ]

    result = decide_claim(make_claim(evidence))

    assert result["decision"] == "SUPPORTED"


# ============================================================
# TEST 2: CONTRADICTED CLAIM
# ============================================================

def test_contradicted_claim():

    evidence = [

        make_evidence(
            1,
            "source1.com",
            "contradiction",
            confidence=0.95
        ),

        make_evidence(
            2,
            "source2.com",
            "contradiction",
            confidence=0.96
        )

    ]

    result = decide_claim(make_claim(evidence))

    assert result["decision"] == "CONTRADICTED"


# ============================================================
# TEST 3: SINGLE-SOURCE CLAIM
# ============================================================

def test_single_source_is_insufficient_for_definitive_verdict():

    evidence = [

        make_evidence(
            1,
            "source1.com",
            "entailment"
        )

    ]

    result = decide_claim(make_claim(evidence))

    assert result["decision"] == "UNVERIFIABLE"

    assert (
        result["evidence_summary"][
            "independent_supporting_sources"
        ] == 1
    )


# ============================================================
# TEST 4: CONFLICTING EVIDENCE
# ============================================================

def test_conflicting_evidence():

    evidence = [

        make_evidence(1, "source1.com", "entailment"),

        make_evidence(
            2,
            "source2.com",
            "contradiction",
            confidence=0.95
        )

    ]

    result = decide_claim(make_claim(evidence))

    assert result["decision"] == "UNVERIFIABLE"


# ============================================================
# TEST 5: LOW CONFIDENCE EVIDENCE
# ============================================================

def test_low_confidence_evidence():

    evidence = [

        make_evidence(
            1,
            "source1.com",
            "entailment",
            confidence=0.50
        ),

        make_evidence(
            2,
            "source2.com",
            "entailment",
            confidence=0.60
        )

    ]

    result = decide_claim(make_claim(evidence))

    assert result["decision"] == "UNVERIFIABLE"


# ============================================================
# TEST 6: LOW SIMILARITY EVIDENCE
# ============================================================

def test_low_similarity_evidence():

    evidence = [

        make_evidence(
            1,
            "source1.com",
            "entailment",
            similarity=0.20
        ),

        make_evidence(
            2,
            "source2.com",
            "entailment",
            similarity=0.30
        )

    ]

    result = decide_claim(make_claim(evidence))

    assert result["decision"] == "UNVERIFIABLE"


# ============================================================
# TEST 7: SAME DOMAIN EVIDENCE
# ============================================================

def test_same_domain_evidence_is_not_independent_corroboration():

    evidence = [

        make_evidence(1, "source1.com", "entailment"),

        make_evidence(2, "source1.com", "entailment")

    ]

    result = decide_claim(make_claim(evidence))

    assert result["decision"] == "UNVERIFIABLE"

    assert (
        result["evidence_summary"][
            "independent_supporting_sources"
        ] == 1
    )


def test_publisher_subdomains_count_as_one_source():
    evidence = [
        make_evidence(1, "en.wikipedia.org", "entailment"),
        make_evidence(2, "www.wikipedia.org", "entailment"),
    ]

    result = decide_claim(make_claim(evidence))

    assert result["decision"] == "UNVERIFIABLE"
    assert result["evidence_summary"]["independent_supporting_sources"] == 1


# ============================================================
# TEST 8: INVALID CLAIM
# ============================================================

def test_invalid_claim():

    with pytest.raises(ValueError):

        decide_claim({

            "claim": "",

            "verification_results": []

        })


# ============================================================
# TEST 9: EMPTY EVIDENCE
# ============================================================

def test_empty_evidence():

    result = decide_claim(make_claim([]))

    assert result["decision"] == "UNVERIFIABLE"


# ============================================================
# TEST 10: NEUTRAL EVIDENCE
# ============================================================

def test_neutral_evidence():

    evidence = [

        make_evidence(1, "source1.com", "neutral"),

        make_evidence(2, "source2.com", "neutral")

    ]

    result = decide_claim(make_claim(evidence))

    assert result["decision"] == "UNVERIFIABLE"


# ============================================================
# TEST 11: MULTIPLE CLAIMS
# ============================================================

def test_multiple_claims():

    data = {

        "claims": [

            make_claim([

                make_evidence(1, "source1.com", "entailment"),

                make_evidence(2, "source2.com", "entailment")

            ]),

            make_claim([

                make_evidence(
                    3,
                    "source3.com",
                    "contradiction",
                    confidence=0.95
                ),

                make_evidence(
                    4,
                    "source4.com",
                    "contradiction",
                    confidence=0.96
                )

            ])

        ]

    }

    result = decide_claims(data)

    assert result["summary"]["total_claims"] == 2

    assert result["summary"]["supported"] == 1

    assert result["summary"]["contradicted"] == 1


# ============================================================
# TEST 12: INVALID PIPELINE INPUT
# ============================================================

def test_invalid_pipeline_input():

    with pytest.raises(ValueError):

        decide_claims([])


# ============================================================
# TEST 13: STRONG EVIDENCE FILTERING
# ============================================================

def test_filter_strong_evidence():

    evidence = [

        make_evidence(1, "source1.com", "entailment"),

        make_evidence(
            2,
            "source2.com",
            "entailment",
            confidence=0.40
        )

    ]

    result = filter_strong_evidence(evidence)

    assert len(result) == 1


# ============================================================
# TEST 14: SOURCE DOMAIN EXTRACTION
# ============================================================

def test_source_domain():

    evidence = {

        "url": "https://www.wikipedia.org/wiki/example"

    }

    result = get_source_domain(evidence)

    assert result == "wikipedia.org"


# ============================================================
# TEST 15: EVIDENCE TEXT PRESERVATION
# ============================================================

def test_evidence_text_preserved():

    evidence = [

        make_evidence(1, "source1.com", "entailment"),

        make_evidence(2, "source2.com", "entailment")

    ]

    result = decide_claim(make_claim(evidence))

    assert "evidence_details" in result

    assert len(
        result["evidence_details"]["supporting"]
    ) == 2


def test_filtered_passages_are_not_presented_as_verdict_evidence():
    evidence = [
        make_evidence(1, "source1.com", "entailment", similarity=0.20),
        make_evidence(2, "source2.com", "entailment"),
        make_evidence(3, "source3.com", "entailment"),
    ]

    result = decide_claim(make_claim(evidence))

    assert result["decision"] == "SUPPORTED"
    assert len(result["evidence_details"]["supporting"]) == 2
    assert len(result["excluded_evidence_details"]["supporting"]) == 1
    assert result["evidence_summary"]["filtered_evidence_count"] == 1
