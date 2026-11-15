import pytest

import nli_verifier as nv


# ==================================================
# MOCK NLI PREDICTIONS
# ==================================================

def mock_nli_batch(premises, hypotheses, bundle=None):

    results = []

    for premise, hypothesis in zip(
        premises,
        hypotheses
    ):

        text = premise.lower()

        if "contradiction test" in text:

            label = "contradiction"

        elif "neutral test" in text:

            label = "neutral"

        else:

            label = "entailment"

        results.append({

            "label": label,

            "confidence": 0.95,

            "probabilities": {

                "entailment": (
                    0.95 if label == "entailment" else 0.02
                ),

                "contradiction": (
                    0.95 if label == "contradiction" else 0.02
                ),

                "neutral": (
                    0.95 if label == "neutral" else 0.03
                )

            }

        })

    return results


# ==================================================
# SINGLE CLAIM VERIFICATION TESTS
# ==================================================

def test_entailment_prediction(monkeypatch):

    monkeypatch.setattr(
        nv,
        "run_nli_batch",
        mock_nli_batch
    )

    evidence = [
        {
            "evidence_id": 1,
            "source": "Wikipedia",
            "content": "Albert Einstein was a physicist."
        }
    ]

    result = nv.verify_claim_evidence(
        "Albert Einstein was a physicist.",
        evidence
    )

    assert len(result) == 1

    assert result[0]["nli_label"] == "entailment"


def test_contradiction_prediction(monkeypatch):

    monkeypatch.setattr(
        nv,
        "run_nli_batch",
        mock_nli_batch
    )

    evidence = [
        {
            "evidence_id": 1,
            "content": "Contradiction test."
        }
    ]

    result = nv.verify_claim_evidence(
        "Test claim.",
        evidence
    )

    assert result[0]["nli_label"] == "contradiction"


def test_neutral_prediction(monkeypatch):

    monkeypatch.setattr(
        nv,
        "run_nli_batch",
        mock_nli_batch
    )

    evidence = [
        {
            "evidence_id": 1,
            "content": "Neutral test."
        }
    ]

    result = nv.verify_claim_evidence(
        "Test claim.",
        evidence
    )

    assert result[0]["nli_label"] == "neutral"


# ==================================================
# INPUT VALIDATION TESTS
# ==================================================

def test_empty_evidence():

    result = nv.verify_claim_evidence(
        "Test claim.",
        []
    )

    assert result == []


def test_invalid_claim():

    with pytest.raises(ValueError):

        nv.verify_claim_evidence(
            "",
            []
        )


def test_invalid_evidence():

    with pytest.raises(ValueError):

        nv.verify_claim_evidence(
            "Test claim.",
            "invalid"
        )


def test_invalid_evidence_items(monkeypatch):

    monkeypatch.setattr(
        nv,
        "run_nli_batch",
        mock_nli_batch
    )

    evidence = [
        {"content": ""},
        {"content": None}
    ]

    result = nv.verify_claim_evidence(
        "Test claim.",
        evidence
    )

    assert result == []


# ==================================================
# TOP K TEST
# ==================================================

def test_top_k_selection(monkeypatch):

    monkeypatch.setattr(
        nv,
        "run_nli_batch",
        mock_nli_batch
    )

    evidence = [
        {"content": "Evidence one."},
        {"content": "Evidence two."},
        {"content": "Evidence three."}
    ]

    result = nv.verify_claim_evidence(
        "Test claim.",
        evidence,
        top_k=2
    )

    assert len(result) == 2


# ==================================================
# COMPLETE VERIFICATION PIPELINE TEST
# ==================================================

def test_verify_claims_pipeline(monkeypatch):

    monkeypatch.setattr(
        nv,
        "run_nli_batch",
        mock_nli_batch
    )

    ranked_result = {

        "claims": [

            {
                "claim_id": 1,

                "claim": "Albert Einstein was a physicist.",

                "source_text": "Einstein was a physicist.",

                "ranked_evidence": [

                    {
                        "evidence_id": 1,
                        "source": "Wikipedia",
                        "content": "Albert Einstein was a physicist.",
                        "url": "https://example.com/einstein",
                        "similarity_score": 0.95
                    }

                ]

            }

        ]

    }

    result = nv.verify_claims(ranked_result)

    assert len(result["claims"]) == 1

    assert (
        result["claims"][0]["total_evidence_verified"]
        == 1
    )

    assert (
        result["claims"][0]["verification_results"][0]["nli_label"]
        == "entailment"
    )


# ==================================================
# EMPTY CLAIMS TEST
# ==================================================

def test_empty_claims():

    result = nv.verify_claims({
        "claims": []
    })

    assert result == {
        "claims": []
    }


# ==================================================
# INVALID INPUT TEST
# ==================================================

def test_invalid_pipeline_input():

    with pytest.raises(ValueError):

        nv.verify_claims("invalid input")