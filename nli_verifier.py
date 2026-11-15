"""
VERI-CLAIM-AI
NLI Verification Module

Model: cross-encoder/nli-deberta-v3-base

Responsibilities:
1. Load the DeBERTa NLI model.
2. Compare evidence passages against factual claims.
3. Classify evidence as entailment, contradiction or neutral.
4. Preserve evidence text and source metadata.
5. Return structured verification results.
"""

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ==================================================
# CONFIGURATION
# ==================================================

MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

MAX_LENGTH = 512

BATCH_SIZE = 8

TOP_K_EVIDENCE = 5


# ==================================================
# MODEL CACHE
# ==================================================

_model = None

_tokenizer = None

_device = None

_label_mapping = None


# ==================================================
# LABEL NORMALIZATION
# ==================================================

def normalize_nli_label(label):

    label = str(label).lower().strip()

    if "entail" in label:
        return "entailment"

    if "contrad" in label:
        return "contradiction"

    if "neutral" in label:
        return "neutral"

    return "neutral"


def get_label_mapping(model):

    mapping = {}

    id2label = model.config.id2label

    for index, label in id2label.items():

        mapping[int(index)] = normalize_nli_label(label)

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


# ==================================================
# LOAD MODEL
# ==================================================

def load_model():

    global _model
    global _tokenizer
    global _device
    global _label_mapping

    if (
        _model is not None
        and _tokenizer is not None
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

    print(f"DeBERTa loaded successfully on {_device}.")

    return {

        "model": _model,

        "tokenizer": _tokenizer,

        "device": _device,

        "label_mapping": _label_mapping

    }


# ==================================================
# RUN BATCH NLI
# ==================================================

def run_nli_batch(
    premises,
    hypotheses,
    bundle=None
):

    if len(premises) != len(hypotheses):

        raise ValueError(
            "Premises and hypotheses must have equal lengths."
        )

    if not premises:

        return []

    if bundle is None:

        bundle = load_model()

    model = bundle["model"]

    tokenizer = bundle["tokenizer"]

    device = bundle["device"]

    label_mapping = bundle["label_mapping"]

    encoded = tokenizer(

        premises,

        hypotheses,

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

    results = []

    for probability_row in probabilities:

        probabilities_dict = {}

        for index, score in enumerate(probability_row):

            label = label_mapping.get(
                index,
                "neutral"
            )

            probabilities_dict[label] = float(
                score.item()
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


# ==================================================
# VERIFY ONE CLAIM AGAINST EVIDENCE
# ==================================================

def verify_claim_evidence(
    claim,
    evidence,
    model_bundle=None,
    top_k=TOP_K_EVIDENCE
):

    if not isinstance(claim, str) or not claim.strip():

        raise ValueError(
            "Claim must be a non-empty string."
        )

    if not isinstance(evidence, list):

        raise ValueError(
            "Evidence must be provided as a list."
        )

    if top_k < 1:

        raise ValueError(
            "top_k must be greater than zero."
        )

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

    verification_results = []

    for item, prediction in zip(
        selected_evidence,
        predictions
    ):

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

            "nli_label": prediction.get("label"),

            "nli_confidence": prediction.get(
                "confidence",
                0
            ),

            "nli_probabilities": prediction.get(
                "probabilities",
                {}
            )

        })

    return verification_results


# ==================================================
# VERIFY MULTIPLE CLAIMS
# ==================================================

def verify_claims(
    ranked_result,
    model_bundle=None
):

    if not isinstance(ranked_result, dict):

        raise ValueError(
            "Ranked result must be a dictionary."
        )

    claims = ranked_result.get("claims", [])

    if not isinstance(claims, list):

        raise ValueError(
            "Claims must be a list."
        )

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

            "verification_results": results,

            "total_evidence_verified": len(results)

        })

        print(
            f"Evidence verified: {len(results)}"
        )

    return {

        "claims": verified_claims

    }


# ==================================================
# TERMINAL TESTING
# ==================================================

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

    retrieval_result = retrieve_evidence({

        "claims": [

            {

                "claim_id": 1,

                "claim": claim,

                "source_text": claim

            }

        ]

    })

    ranked_result = rank_evidence(
        retrieval_result
    )

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


if __name__ == "__main__":

    main()