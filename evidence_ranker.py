
import re
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5
EPSILON = 1e-10

# Semantic similarity handles paraphrases; lexical overlap protects exact
# names, dates, quantities, and technical terms from being lost in the mix.
SEMANTIC_WEIGHT = 0.8
LEXICAL_WEIGHT = 1.0 - SEMANTIC_WEIGHT

_model = None


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_top_k(top_k):

    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError("top_k must be a non-negative integer.")

    if top_k < 0:
        raise ValueError("top_k cannot be negative.")

    return top_k


# ============================================================
# LOAD MODEL
# ============================================================

def get_model():
    """
    Load MiniLM once and reuse the same model instance.
    """

    global _model

    if _model is None:

        print("\nLoading MiniLM embedding model...")

        _model = SentenceTransformer(MODEL_NAME)

        print("MiniLM model loaded successfully.")

    return _model


# ============================================================
# EMBEDDING VALIDATION
# ============================================================

def validate_embeddings(claim_embedding, evidence_embeddings):
    """
    Validate embedding dimensions and numerical values.
    """

    claim_embedding = np.asarray(
        claim_embedding,
        dtype=np.float64
    )

    evidence_embeddings = np.asarray(
        evidence_embeddings,
        dtype=np.float64
    )

    if claim_embedding.ndim != 1:
        raise ValueError(
            "Claim embedding must be a one-dimensional vector."
        )

    if evidence_embeddings.ndim == 1:

        evidence_embeddings = evidence_embeddings.reshape(1, -1)

    if evidence_embeddings.ndim != 2:
        raise ValueError(
            "Evidence embeddings must be a two-dimensional matrix."
        )

    if claim_embedding.shape[0] != evidence_embeddings.shape[1]:
        raise ValueError(
            "Claim and evidence embedding dimensions do not match."
        )

    if not np.all(np.isfinite(claim_embedding)):
        raise ValueError(
            "Claim embedding contains invalid numerical values."
        )

    if not np.all(np.isfinite(evidence_embeddings)):
        raise ValueError(
            "Evidence embeddings contain invalid numerical values."
        )

    return claim_embedding, evidence_embeddings


# ============================================================
# COSINE SIMILARITY
# ============================================================

def calculate_similarity(claim_embedding, evidence_embeddings):
    """
    Calculate cosine similarity between one claim embedding
    and multiple evidence embeddings.

    Returns a NumPy array containing similarity scores.

    Handles:
    - Normalized embeddings
    - Non-normalized embeddings
    - Zero vectors
    - Empty evidence embeddings
    """

    claim_embedding = np.asarray(
        claim_embedding,
        dtype=np.float64
    )

    evidence_embeddings = np.asarray(
        evidence_embeddings,
        dtype=np.float64
    )

    if evidence_embeddings.size == 0:

        return np.array([], dtype=np.float64)

    claim_embedding, evidence_embeddings = validate_embeddings(
        claim_embedding,
        evidence_embeddings
    )

    claim_norm = np.linalg.norm(claim_embedding)

    evidence_norms = np.linalg.norm(
        evidence_embeddings,
        axis=1
    )

    denominator = (
        evidence_norms * claim_norm
    )

    similarities = np.zeros(
        len(evidence_embeddings),
        dtype=np.float64
    )

    valid_mask = denominator > EPSILON

    if claim_norm <= EPSILON:
        return similarities

    similarities[valid_mask] = (
        np.dot(
            evidence_embeddings[valid_mask],
            claim_embedding
        )
        / denominator[valid_mask]
    )

    # Protect against floating-point precision errors.
    similarities = np.clip(
        similarities,
        -1.0,
        1.0
    )

    return similarities


# ============================================================
# PREPARE EVIDENCE
# ============================================================

def prepare_evidence(evidence):
    """
    Remove invalid evidence records.

    Only records containing non-empty textual content
    are considered for semantic ranking.
    """

    if not isinstance(evidence, list):
        raise ValueError("Evidence must be provided as a list.")

    valid_evidence = []

    for item in evidence:

        if not isinstance(item, dict):
            continue

        content = item.get("content")

        if not isinstance(content, str):
            continue

        if not content.strip():
            continue

        valid_evidence.append(item)

    return valid_evidence


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

def generate_embeddings(texts, model):
    """
    Generate normalized sentence embeddings.

    Returns a two-dimensional NumPy matrix.
    """

    if not texts:
        return np.empty((0, 0), dtype=np.float64)

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float64
    )

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    if embeddings.ndim != 2:
        raise ValueError(
            "Model returned invalid embedding dimensions."
        )

    if embeddings.shape[0] != len(texts):
        raise ValueError(
            "Number of generated embeddings does not match input texts."
        )

    if not np.all(np.isfinite(embeddings)):
        raise ValueError(
            "Generated embeddings contain invalid numerical values."
        )

    return embeddings


def calculate_lexical_overlap(claim, evidence_text):
    """Return a bounded token-overlap F1 score for exact term matching."""
    tokenize = lambda text: re.findall(r"\b[\w]+\b", text.casefold())
    claim_tokens = tokenize(claim)
    evidence_tokens = tokenize(evidence_text)
    if not claim_tokens or not evidence_tokens:
        return 0.0

    claim_set = set(claim_tokens)
    evidence_set = set(evidence_tokens)
    overlap = len(claim_set & evidence_set)
    precision = overlap / len(evidence_set)
    recall = overlap / len(claim_set)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


# ============================================================
# RANK EVIDENCE FOR ONE CLAIM
# ============================================================

def rank_claim_evidence(
    claim,
    evidence,
    model=None,
    top_k=TOP_K
):
    """
    Rank evidence passages with a hybrid semantic and lexical score.

    Pipeline:

        Claim
          |
          v
        Claim Embedding
          |
          v
        Evidence Embeddings
          |
          v
        Cosine Similarity
          |
          v
        Descending Ranking
          |
          v
        Top-K Evidence
    """

    top_k = validate_top_k(top_k)

    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("Claim must be a non-empty string.")

    valid_evidence = prepare_evidence(evidence)

    if not valid_evidence or top_k == 0:
        return []

    if model is None:
        model = get_model()

    claim = claim.strip()

    # Generate claim embedding.
    claim_embedding = generate_embeddings(
        [claim],
        model
    )[0]

    # Generate evidence embeddings.
    evidence_texts = [
        item["content"].strip()
        for item in valid_evidence
    ]

    evidence_embeddings = generate_embeddings(
        evidence_texts,
        model
    )

    # Calculate cosine similarity.
    similarities = calculate_similarity(
        claim_embedding,
        evidence_embeddings
    )

    ranked_evidence = []

    for item, score in zip(
        valid_evidence,
        similarities
    ):

        ranked_item = item.copy()

        ranked_item["similarity_score"] = float(score)

        ranked_evidence.append(ranked_item)

    # Preserve the raw embedding score for inspection, and use a small
    # lexical contribution to favor passages containing the claim's exact
    # entities and numbers. Neither score is a truth probability.
    for item in ranked_evidence:
        semantic_score = item["similarity_score"]
        lexical_score = calculate_lexical_overlap(claim, item["content"])
        item["semantic_similarity_score"] = semantic_score
        item["lexical_overlap_score"] = lexical_score
        item["similarity_score"] = (
            SEMANTIC_WEIGHT * semantic_score + LEXICAL_WEIGHT * lexical_score
        )

    # Highest hybrid relevance first.
    ranked_evidence.sort(
        key=lambda item: item["similarity_score"],
        reverse=True
    )

    # Select Top-K passages.
    ranked_evidence = ranked_evidence[:top_k]

    # Assign ranking positions.
    for index, item in enumerate(
        ranked_evidence,
        start=1
    ):

        item["rank"] = index

    return ranked_evidence


# ============================================================
# RANK EVIDENCE FOR ALL CLAIMS
# ============================================================

def rank_evidence(
    retrieval_result,
    model=None,
    top_k=TOP_K
):
    """
    Rank retrieved evidence for every extracted claim.

    Expected input:

    {
        "claims": [
            {
                "claim_id": 1,
                "claim": "Example claim",
                "source_text": "Original text",
                "evidence": [...]
            }
        ]
    }

    Returns ranked evidence for each claim.
    """

    top_k = validate_top_k(top_k)

    if not isinstance(retrieval_result, dict):
        raise ValueError(
            "Input must be a dictionary."
        )

    claims = retrieval_result.get("claims")

    if not isinstance(claims, list):
        raise ValueError(
            "Input must contain a claims list."
        )

    if not claims:
        return {"claims": []}

    ranked_claims = []

    for claim_item in claims:

        if not isinstance(claim_item, dict):
            continue

        claim = claim_item.get("claim", "")

        evidence = claim_item.get("evidence", [])

        if not isinstance(claim, str) or not claim.strip():
            continue

        if not isinstance(evidence, list):
            evidence = []

        print(
            f"\nRanking evidence for: {claim}"
        )

        ranked = rank_claim_evidence(
            claim=claim,
            evidence=evidence,
            model=model,
            top_k=top_k
        )

        ranked_claim = {

            "claim_id": claim_item.get("claim_id"),

            "claim": claim,

            "source_text": claim_item.get(
                "source_text",
                ""
            ),

            "source_start": claim_item.get("source_start"),

            "source_end": claim_item.get("source_end"),

            "retrieval_status": claim_item.get(
                "retrieval_status",
                {"state": "unknown", "warnings": []}
            ),

            "ranked_evidence": ranked,

            "total_evidence": len(evidence),

            "selected_evidence": len(ranked)

        }

        ranked_claims.append(ranked_claim)

        print(
            f"Total evidence: {len(evidence)}"
        )

        print(
            f"Evidence selected: {len(ranked)}"
        )

    return {
        "claims": ranked_claims
    }


# ============================================================
# TERMINAL INTERFACE
# ============================================================

def main():

    import json

    from evidence_retrieval import retrieve_evidence

    print("\n" + "=" * 55)

    print("              VERI-CLAIM-AI")

    print("              EVIDENCE RANKER")

    print("=" * 55)

    claim = input("\nEnter Claim: ").strip()

    if not claim:

        print("No claim provided.")

        return

    claims = {

        "claims": [

            {
                "claim_id": 1,

                "claim": claim,

                "source_text": claim
            }

        ]

    }

    try:

        # Step 1: Retrieve evidence.
        print("\nRetrieving evidence...")

        retrieval_result = retrieve_evidence(claims)

        # Step 2: Rank evidence.
        print("\nRanking evidence...")

        result = rank_evidence(retrieval_result)

        print("\n" + "=" * 55)

        print("             RANKING RESULTS")

        print("=" * 55)

        print(
            json.dumps(
                result,
                indent=4,
                ensure_ascii=False
            )
        )

    except (ValueError, RuntimeError) as error:

        print(f"\nRanking failed: {error}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
