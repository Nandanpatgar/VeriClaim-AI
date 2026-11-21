import numpy as np
import pytest

from evidence_ranker import (
    calculate_similarity,
    calculate_lexical_overlap,
    rank_claim_evidence,
    rank_evidence
)


# ==================================================
# MOCK MODEL
# ==================================================

class MockEmbeddingModel:

    def encode(
        self,
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True
    ):

        embeddings = []

        for sentence in sentences:

            text = sentence.lower()

            if "einstein" in text or "relativity" in text:
                embeddings.append([1.0, 0.0])

            elif "football" in text:
                embeddings.append([0.0, 1.0])

            else:
                embeddings.append([0.5, 0.5])

        return np.array(embeddings, dtype=float)


# ==================================================
# COSINE SIMILARITY TESTS
# ==================================================

def test_identical_embeddings():

    claim = np.array([1.0, 0.0])

    evidence = np.array([
        [1.0, 0.0]
    ])

    result = calculate_similarity(claim, evidence)

    assert result[0] == pytest.approx(1.0)


def test_orthogonal_embeddings():

    claim = np.array([1.0, 0.0])

    evidence = np.array([
        [0.0, 1.0]
    ])

    result = calculate_similarity(claim, evidence)

    assert result[0] == pytest.approx(0.0)


# ==================================================
# EVIDENCE RANKING TESTS
# ==================================================

def test_rank_evidence():

    model = MockEmbeddingModel()

    claim = "Albert Einstein developed relativity."

    evidence = [
        {
            "content": "Football is played worldwide.",
            "source": "Test",
            "url": "https://example.com/football"
        },
        {
            "content": "Albert Einstein developed relativity.",
            "source": "Test",
            "url": "https://example.com/einstein"
        }
    ]

    result = rank_claim_evidence(
        claim,
        evidence,
        model=model
    )

    assert len(result) == 2

    assert result[0]["similarity_score"] >= result[1]["similarity_score"]

    assert result[0]["rank"] == 1


def test_empty_evidence():

    result = rank_claim_evidence(
        "Albert Einstein developed relativity.",
        [],
        model=MockEmbeddingModel()
    )

    assert result == []


def test_invalid_evidence_content():

    evidence = [
        {"content": ""},
        {"content": None}
    ]

    result = rank_claim_evidence(
        "Albert Einstein developed relativity.",
        evidence,
        model=MockEmbeddingModel()
    )

    assert result == []


def test_top_k_selection():

    model = MockEmbeddingModel()

    evidence = [
        {"content": "Albert Einstein developed relativity."},
        {"content": "Football is a sport."},
        {"content": "Einstein studied physics."}
    ]

    result = rank_claim_evidence(
        "Albert Einstein developed relativity.",
        evidence,
        model=model,
        top_k=2
    )

    assert len(result) == 2


def test_lexical_overlap_rewards_exact_terms():
    exact = calculate_lexical_overlap(
        "Einstein published the 1905 paper",
        "Einstein published the 1905 paper on relativity"
    )
    paraphrase = calculate_lexical_overlap(
        "Einstein published the 1905 paper",
        "A physicist released a famous work in the early twentieth century"
    )

    assert exact > paraphrase


def test_hybrid_ranking_keeps_semantic_and_lexical_provenance():
    claim = "Einstein studied physics in 1905."
    evidence = [
        {"content": "Einstein wrote about relativity in 1905."},
        {"content": "Einstein studied physics in 1905."},
    ]

    ranked = rank_claim_evidence(claim, evidence, model=MockEmbeddingModel())

    assert ranked[0]["content"] == claim
    assert ranked[0]["semantic_similarity_score"] == pytest.approx(1.0)
    assert ranked[0]["lexical_overlap_score"] > ranked[1]["lexical_overlap_score"]
    assert ranked[0]["similarity_score"] == pytest.approx(
        0.8 * ranked[0]["semantic_similarity_score"]
        + 0.2 * ranked[0]["lexical_overlap_score"]
    )


# ==================================================
# COMPLETE PIPELINE TEST
# ==================================================

def test_rank_evidence_pipeline():

    model = MockEmbeddingModel()

    retrieval_result = {
        "claims": [
            {
                "claim_id": 1,
                "claim": "Albert Einstein developed relativity.",
                "source_text": "Einstein developed relativity.",
                "retrieval_status": {"state": "partial", "warnings": ["Tavily unavailable"]},
                "evidence": [
                    {
                        "source": "Wikipedia",
                        "content": "Albert Einstein developed relativity.",
                        "url": "https://example.com/einstein"
                    },
                    {
                        "source": "Tavily",
                        "content": "Football is a popular sport.",
                        "url": "https://example.com/football"
                    }
                ],
                "total_evidence": 2
            }
        ]
    }

    result = rank_evidence(
        retrieval_result,
        model=model
    )

    assert len(result["claims"]) == 1

    assert result["claims"][0]["retrieval_status"]["state"] == "partial"

    ranked = result["claims"][0]["ranked_evidence"]

    assert len(ranked) == 2

    assert ranked[0]["similarity_score"] >= ranked[1]["similarity_score"]


def test_empty_claims():

    result = rank_evidence(
        {"claims": []},
        model=MockEmbeddingModel()
    )

    assert result == {"claims": []}


def test_invalid_input():

    with pytest.raises(ValueError):

        rank_evidence("invalid input")


# ==================================================
# END
# ==================================================
