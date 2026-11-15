import numpy as np
from sentence_transformers import SentenceTransformer


# ==================================================
# CONFIGURATION
# ==================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5

_model = None


# ==================================================
# LOAD MODEL
# ==================================================

def get_model():
    """
    Load MiniLM once and reuse it.
    """

    global _model

    if _model is None:
        print("\nLoading MiniLM embedding model...")
        _model = SentenceTransformer(MODEL_NAME)
        print("MiniLM model loaded successfully.")

    return _model


# ==================================================
# COSINE SIMILARITY
# ==================================================

def calculate_similarity(claim_embedding, evidence_embeddings):
    """
    Calculate cosine similarity between one claim
    embedding and multiple evidence embeddings.

    Embeddings are normalized before dot product.
    """

    claim_embedding = np.asarray(claim_embedding)
    evidence_embeddings = np.asarray(evidence_embeddings)

    claim_norm = np.linalg.norm(claim_embedding)

    evidence_norms = np.linalg.norm(
        evidence_embeddings,
        axis=1
    )

    if claim_norm == 0:
        return np.zeros(len(evidence_embeddings))

    similarities = np.dot(
        evidence_embeddings,
        claim_embedding
    ) / (evidence_norms * claim_norm + 1e-10)

    return similarities


# ==================================================
# RANK EVIDENCE FOR ONE CLAIM
# ==================================================

def rank_claim_evidence(claim, evidence, model=None, top_k=TOP_K):
    """
    Rank evidence passages according to their
    semantic similarity with a claim.
    """

    if not evidence:
        return []

    if model is None:
        model = get_model()

    valid_evidence = []

    for item in evidence:

        content = item.get("content", "")

        if isinstance(content, str) and content.strip():
            valid_evidence.append(item)

    if not valid_evidence:
        return []

    claim_embedding = model.encode(
        [claim],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    evidence_texts = [
        item["content"]
        for item in valid_evidence
    ]

    evidence_embeddings = model.encode(
        evidence_texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    similarities = calculate_similarity(
        claim_embedding,
        evidence_embeddings
    )

    ranked_evidence = []

    for item, score in zip(valid_evidence, similarities):

        ranked_item = item.copy()

        ranked_item["similarity_score"] = float(score)

        ranked_evidence.append(ranked_item)

    # Highest similarity first.
    ranked_evidence.sort(
        key=lambda x: x["similarity_score"],
        reverse=True
    )

    # Keep only the requested number of passages.
    ranked_evidence = ranked_evidence[:top_k]

    # Assign ranking positions.
    for index, item in enumerate(ranked_evidence, start=1):

        item["rank"] = index

    return ranked_evidence


# ==================================================
# RANK EVIDENCE FOR ALL CLAIMS
# ==================================================

def rank_evidence(retrieval_result, model=None, top_k=TOP_K):
    """
    Rank retrieved evidence for every claim.

    Accepts the output of evidence_retrieval.py.
    """

    if not isinstance(retrieval_result, dict):
        raise ValueError("Input must be a dictionary.")

    claims = retrieval_result.get("claims")

    if not isinstance(claims, list):
        raise ValueError("Input must contain a claims list.")

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

        print(f"\nRanking evidence for: {claim}")

        ranked = rank_claim_evidence(
            claim,
            evidence,
            model=model,
            top_k=top_k
        )

        ranked_claims.append({

            "claim_id": claim_item.get("claim_id"),

            "claim": claim,

            "source_text": claim_item.get("source_text", ""),

            "ranked_evidence": ranked,

            "total_evidence": len(evidence),

            "selected_evidence": len(ranked)

        })

        print(f"Evidence ranked: {len(ranked)}")

    return {
        "claims": ranked_claims
    }


# ==================================================
# TERMINAL INTERFACE
# ==================================================

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

    # Retrieve evidence.
    retrieval_result = retrieve_evidence(claims)

    # Rank retrieved evidence.
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


if __name__ == "__main__":
    main()