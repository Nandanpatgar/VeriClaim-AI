
"""
VERI-CLAIM-AI
NLI Verification Module

Model: cross-encoder/nli-deberta-v3-base

Responsibilities:
1. Load and cache the DeBERTa NLI model.
2. Compare evidence passages against factual claims.
3. Classify evidence as entailment, contradiction or neutral.
4. Preserve evidence text and source metadata.
5. Support efficient batch inference.
6. Return structured verification results.
"""

import math
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

MAX_LENGTH = 512

BATCH_SIZE = 8

TOP_K_EVIDENCE = 5

EPSILON = 1e-10


# ============================================================
# MODEL CACHE
# ============================================================

_model = None
_tokenizer = None
_device = None
_label_mapping = None


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_nli_label(label):
    """
    Normalize different model label formats.

    Supported labels:
        entailment
        contradiction
        neutral

    Unknown labels are treated as neutral.
    """

    label = str(label).lower().strip()

    if "entail" in label:
        return "entailment"

    if "contrad" in label:
        return "contradiction"

    if "neutral" in label:
        return "neutral"

    return "neutral"


# ============================================================
# LABEL MAPPING
# ============================================================

def get_label_mapping(model):
    """
    Extract NLI label mapping from model configuration.

    Falls back to the known label order of the configured
    DeBERTa NLI model when generic LABEL_x names are returned.
    """

    id2label = getattr(
        model.config,
        "id2label",
        {}
    )

    mapping = {}

    for index, label in id2label.items():

        try:
            index = int(index)
        except (ValueError, TypeError):
            continue

        mapping[index] = normalize_nli_label(label)

    expected = {
        "entailment",
        "contradiction",
        "neutral"
    }

    if set(mapping.values()) != expected:

        mapping = {
            0: "contradiction",
            1: "entailment",
            2: "neutral"
        }

    return mapping


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Load DeBERTa and tokenizer once.

    Uses GPU when available; otherwise uses CPU.
    """

    global _model
    global _tokenizer
    global _device
    global _label_mapping

    if (
        _model is not None
        and _tokenizer is not None
        and _device is not None
        and _label_mapping is not None
    ):

        return {
            "model": _model,
            "tokenizer": _tokenizer,
            "device": _device,
            "label_mapping": _label_mapping
        }

    print("\nLoading DeBERTa NLI model...")

    _device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    _model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    _model.to(_device)

    _model.eval()

    _label_mapping = get_label_mapping(_model)

    print(
        f"DeBERTa loaded successfully on {_device}."
    )

    return {
        "model": _model,
        "tokenizer": _tokenizer,
        "device": _device,
        "label_mapping": _label_mapping
    }


# ============================================================
# PROBABILITY NORMALIZATION
# ============================================================

def normalize_probabilities(probabilities):
    """
    Ensure all three NLI classes are present.

    Returns a normalized probability dictionary.
    """

    labels = [
        "entailment",
        "contradiction",
        "neutral"
    ]

    normalized = {}

    for label in labels:

        try:
            value = float(probabilities.get(label, 0.0))
        except (ValueError, TypeError):
            value = 0.0

        if not math.isfinite(value):
            value = 0.0

        normalized[label] = max(0.0, value)

    total = sum(normalized.values())

    if total <= EPSILON:

        return {
            "entailment": 0.0,
            "contradiction": 0.0,
            "neutral": 1.0
        }

    return {
        label: value / total
        for label, value in normalized.items()
    }


# ============================================================
# RUN BATCH NLI
# ============================================================

def run_nli_batch(
    premises,
    hypotheses,
    bundle=None
):
    """
    Perform batch NLI inference.

    Premise:
        Retrieved evidence passage.

    Hypothesis:
        Factual claim being verified.

    Returns:
        [
            {
                "label": "entailment",
                "confidence": 0.95,
                "probabilities": {
                    "entailment": 0.95,
                    "contradiction": 0.02,
                    "neutral": 0.03
                }
            }
        ]
    """

    if not isinstance(premises, list):
        raise ValueError("Premises must be a list.")

    if not isinstance(hypotheses, list):
        raise ValueError("Hypotheses must be a list.")

    if len(premises) != len(hypotheses):

        raise ValueError(
            "Premises and hypotheses must have equal lengths."
        )

    if not premises:
        return []

    for premise, hypothesis in zip(
        premises,
        hypotheses
    ):

        if not isinstance(premise, str) or not premise.strip():
            raise ValueError("Premises must contain non-empty strings.")

        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise ValueError("Hypotheses must contain non-empty strings.")

    if bundle is None:
        bundle = load_model()

    required_keys = {
        "model",
        "tokenizer",
        "device",
        "label_mapping"
    }

    if not isinstance(bundle, dict) or not required_keys.issubset(bundle):
        raise ValueError("Invalid NLI model bundle.")

    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    device = bundle["device"]
    label_mapping = bundle["label_mapping"]

    if not isinstance(label_mapping, dict):
        raise ValueError("Invalid NLI label mapping.")

    results = []

    # Process passages in manageable batches.
    for start in range(0, len(premises), BATCH_SIZE):

        batch_premises = premises[
            start:start + BATCH_SIZE
        ]

        batch_hypotheses = hypotheses[
            start:start + BATCH_SIZE
        ]

        encoded = tokenizer(
            batch_premises,
            batch_hypotheses,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.no_grad():

            outputs = model(**encoded)

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )

        for probability_row in probabilities:

            probabilities_dict = {}

            for index, score in enumerate(probability_row):

                label = label_mapping.get(
                    index,
                    label_mapping.get(str(index), "neutral")
                )

                label = normalize_nli_label(label)

                probabilities_dict[label] = float(
                    score.item()
                )

            probabilities_dict = normalize_probabilities(
                probabilities_dict
            )

            predicted_label = max(
                probabilities_dict,
                key=probabilities_dict.get
            )

            confidence = probabilities_dict[
                predicted_label
            ]

            results.append({
                "label": predicted_label,
                "confidence": confidence,
                "probabilities": probabilities_dict
            })

    return results


# ============================================================
# VERIFY ONE CLAIM AGAINST EVIDENCE
# ============================================================

def verify_claim_evidence(
    claim,
    evidence,
    model_bundle=None,
    top_k=TOP_K_EVIDENCE
):
    """
    Verify one claim against multiple evidence passages.

    Each evidence passage is independently classified as:
        entailment
        contradiction
        neutral

    The function does not determine the final claim verdict.
    That responsibility belongs to decision_engine.py.
    """

    if not isinstance(claim, str) or not claim.strip():

        raise ValueError(
            "Claim must be a non-empty string."
        )

    if not isinstance(evidence, list):

        raise ValueError(
            "Evidence must be provided as a list."
        )

    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k < 1
    ):

        raise ValueError(
            "top_k must be a positive integer."
        )

    claim = claim.strip()

    valid_evidence = []

    for item in evidence:

        if not isinstance(item, dict):
            continue

        content = item.get(
            "content",
            item.get("evidence_text", "")
        )

        if not isinstance(content, str):
            continue

        content = content.strip()

        if not content:
            continue

        valid_item = item.copy()

        valid_item["content"] = content

        valid_evidence.append(valid_item)

    selected_evidence = valid_evidence[:top_k]

    if not selected_evidence:
        return []

    premises = [
        item["content"]
        for item in selected_evidence
    ]

    hypotheses = [
        claim
        for _ in selected_evidence
    ]

    predictions = run_nli_batch(
        premises,
        hypotheses,
        model_bundle
    )

    if len(predictions) != len(selected_evidence):

        raise ValueError(
            "NLI prediction count does not match evidence count."
        )

    verification_results = []

    for item, prediction in zip(
        selected_evidence,
        predictions
    ):

        if not isinstance(prediction, dict):
            raise ValueError("Invalid NLI prediction.")

        label = normalize_nli_label(
            prediction.get("label", "neutral")
        )

        probabilities = normalize_probabilities(
            prediction.get("probabilities", {})
        )

        try:
            confidence = float(
                prediction.get(
                    "confidence",
                    probabilities[label]
                )
            )
        except (ValueError, TypeError):
            confidence = probabilities[label]

        if not math.isfinite(confidence):
            confidence = probabilities[label]

        confidence = max(
            0.0,
            min(1.0, confidence)
        )

        verification_results.append({

            "evidence_id": item.get("evidence_id"),

            "source": item.get("source", "Unknown"),

            "title": item.get("title", ""),

            "url": item.get("url", ""),

            "evidence_text": item.get("content", ""),

            "similarity_score": item.get(
                "similarity_score",
                item.get("score", 0)
            ),

            "semantic_similarity_score": item.get(
                "semantic_similarity_score"
            ),

            "lexical_overlap_score": item.get(
                "lexical_overlap_score"
            ),

            "retrieval_provider": item.get("retrieval_provider"),

            "published_date": item.get("published_date"),

            "nli_label": label,

            "nli_confidence": confidence,

            "nli_probabilities": probabilities

        })

    return verification_results


# ============================================================
# VERIFY MULTIPLE CLAIMS
# ============================================================

def verify_claims(
    ranked_result,
    model_bundle=None
):
    """
    Verify all claims in a ranked evidence result.

    Expected input:

    {
        "claims": [
            {
                "claim_id": 1,
                "claim": "Example claim",
                "ranked_evidence": [...]
            }
        ]
    }
    """

    if not isinstance(ranked_result, dict):

        raise ValueError(
            "Ranked result must be a dictionary."
        )

    claims = ranked_result.get("claims")

    if not isinstance(claims, list):

        raise ValueError(
            "Claims must be a list."
        )

    if not claims:
        return {"claims": []}

    verified_claims = []

    for claim_data in claims:

        if not isinstance(claim_data, dict):
            continue

        claim = claim_data.get("claim", "")

        if not isinstance(claim, str) or not claim.strip():
            continue

        ranked_evidence = claim_data.get(
            "ranked_evidence",
            []
        )

        if not isinstance(ranked_evidence, list):
            ranked_evidence = []

        print(
            f"\nVerifying Claim: {claim}"
        )

        results = verify_claim_evidence(
            claim,
            ranked_evidence,
            model_bundle=model_bundle
        )

        verified_claims.append({

            "claim_id": claim_data.get("claim_id"),

            "claim": claim,

            "source_text": claim_data.get(
                "source_text",
                claim
            ),

            "source_start": claim_data.get("source_start"),

            "source_end": claim_data.get("source_end"),

            "retrieval_status": claim_data.get(
                "retrieval_status",
                {"state": "unknown", "warnings": []}
            ),

            "verification_results": results,

            "total_evidence_verified": len(results)

        })

        print(
            f"Evidence verified: {len(results)}"
        )

    return {
        "claims": verified_claims
    }


# ============================================================
# TERMINAL TESTING
# ============================================================

def main():

    import json

    from evidence_retrieval import retrieve_evidence
    from evidence_ranker import rank_evidence

    print("\n" + "=" * 55)

    print("              VERI-CLAIM-AI")

    print("              NLI VERIFICATION")

    print("=" * 55)

    claim = input("\nEnter Claim: ").strip()

    if not claim:

        print("Claim cannot be empty.")

        return

    try:

        # Step 1: Retrieve evidence.
        retrieval_result = retrieve_evidence({

            "claims": [

                {
                    "claim_id": 1,
                    "claim": claim,
                    "source_text": claim
                }

            ]

        })

        # Step 2: Rank evidence.
        ranked_result = rank_evidence(
            retrieval_result
        )

        # Step 3: Verify evidence.
        verified_result = verify_claims(
            ranked_result
        )

        print("\nNLI VERIFICATION RESULTS")

        print(
            json.dumps(
                verified_result,
                indent=4,
                ensure_ascii=False
            )
        )

    except (ValueError, RuntimeError) as error:

        print(f"\nVerification failed: {error}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
