import pytest

from decision_engine import (
    decide_claim,
    decide_claims,
    filter_strong_evidence,
    get_source_domain
)


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


def test_supported_claim():

    evidence = [

        make_evidence(1, "source1.com", "entailment"),

        make_evidence(2, "source2.com", "entailment")

    ]

    result = decide_claim(make_claim(evidence))

    assert result["decision"] == "SUPPORTED"


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


def test_unverifiable_claim():

    evidence = [

        make_evidence(1, "source1.com", "entailment")

    ]

    result = decide_claim(make_claim(evidence))

    assert result["decision"] == "UNVERIFIABLE"


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


def test_same_domain_not_independent():

    evidence = [

        make_evidence(1, "source1.com", "entailment"),

        make_evidence(2, "source1.com", "entailment")

    ]

    result = decide_claim(make_claim(evidence))

    assert result["decision"] == "UNVERIFIABLE"


def test_invalid_claim():

    with pytest.raises(ValueError):

        decide_claim({

            "claim": "",

            "verification_results": []

        })


def test_empty_evidence():

    result = decide_claim(make_claim([]))

    assert result["decision"] == "UNVERIFIABLE"


def test_neutral_evidence():

    evidence = [

        make_evidence(1, "source1.com", "neutral"),

        make_evidence(2, "source2.com", "neutral")

    ]

    result = decide_claim(make_claim(evidence))

    assert result["decision"] == "UNVERIFIABLE"


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


def test_invalid_pipeline_input():

    with pytest.raises(ValueError):

        decide_claims([])


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


def test_source_domain():

    evidence = {

        "url": "https://www.wikipedia.org/wiki/example"

    }

    result = get_source_domain(evidence)

    assert result == "wikipedia.org"


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