import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ==================================================
# CONFIGURATION
# ==================================================

MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

MAX_LENGTH = 512

BATCH_SIZE = 8

TOP_K_EVIDENCE = 5

_model = None
_tokenizer = None
_device = None


# ==================================================
# MODEL LOADING
# ==================================================

def load_model():
    """
    Load the DeBERTa NLI model once and reuse it.
    """

    global _model, _tokenizer, _device

    if _model is None:

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

        print(f"DeBERTa loaded successfully on {_device}.")

    return _tokenizer, _model, _device


# ==================================================
# LABEL MAPPING
# ==================================================

def get_label_mapping(model):
    """
    Map model output indexes to NLI labels.
    """

    id2label = model.config.id2label

    mapping = {}

    for index, label in id2label.items():

        normalized = str(label).lower()

        if "entail" in normalized:
            mapping[int(index)] = "entailment"

        elif "contrad" in normalized:
            mapping[int(index)] = "contradiction"

        elif "neutral" in normalized:
            mapping[int(index)] = "neutral"

    # Fallback for the specified pretrained model.
    if len(mapping) != 3:

        mapping = {
            0: "contradiction",
            1: "entailment",
            2: "neutral"
        }

    return mapping


# ==================================================
# NLI BATCH PREDICTION
# ==================================================

def run_nli_batch(premises, hypotheses, bundle=None):
    """
    Compare multiple evidence passages against claims.

    Premise = Evidence
    Hypothesis = Claim
    """

    if bundle is None:
        bundle = load_model()

    tokenizer, model, device = bundle

    inputs = tokenizer(
        premises,
        hypotheses,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )

    probabilities = probabilities.cpu().numpy()

    label_mapping = get_label_mapping(model)

    results = []

    for probability_row in probabilities:

        scores = {}

        for index, score in enumerate(probability_row):

            label = label_mapping[index]

            scores[label] = float(score)

        predicted_label = max(
            scores,
            key=scores.get
        )

        results.append({

            "label": predicted_label,

            "probabilities": scores,

            "confidence": scores[predicted_label]

        })

    return results


# ==================================================
# VERIFY EVIDENCE FOR ONE CLAIM
# ==================================================

def verify_claim_evidence(
    claim,
    evidence,
    model_bundle=None,
    top_k=TOP_K_EVIDENCE
):
    """
    Perform NLI verification for one claim
    against its ranked evidence.
    """

    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("Claim must be a non-empty string.")

    if not isinstance(evidence, list):
        raise ValueError("Evidence must be a list.")

    if not evidence:
        return []

    valid_evidence = []

    for item in evidence:

        if not isinstance(item, dict):
            continue

        content = item.get("content")

        if isinstance(content, str) and content.strip():
            valid_evidence.append(item)

    valid_evidence = valid_evidence[:top_k]

    if not valid_evidence:
        return []

    premises = [
        item["content"]
        for item in valid_evidence
    ]

    hypotheses = [
        claim
        for _ in valid_evidence
    ]

    predictions = run_nli_batch(
        premises,
        hypotheses,
        bundle=model_bundle
    )

    results = []

    for item, prediction in zip(
        valid_evidence,
        predictions
    ):

        results.append({

            "evidence_id": item.get("evidence_id"),

            "source": item.get("source"),

            "title": item.get("title"),

            "url": item.get("url"),

            "similarity_score": item.get(
                "similarity_score"
            ),

            "nli_label": prediction["label"],

            "nli_confidence": prediction["confidence"],

            "nli_probabilities": prediction["probabilities"]

        })

    return results


# ==================================================
# VERIFY ALL CLAIMS
# ==================================================

def verify_claims(ranked_result, model_bundle=None):
    """
    Perform NLI verification for all claims.

    Input:
        Output of evidence_ranker.py

    Output:
        Evidence-level NLI verification results.
    """

    if not isinstance(ranked_result, dict):

        raise ValueError(
            "Input must be a dictionary."
        )

    claims = ranked_result.get("claims")

    if not isinstance(claims, list):

        raise ValueError(
            "Input must contain a claims list."
        )

    verified_claims = []

    for claim_item in claims:

        if not isinstance(claim_item, dict):
            continue

        claim_id = claim_item.get("claim_id")

        claim = claim_item.get("claim", "")

        evidence = claim_item.get(
            "ranked_evidence",
            []
        )

        if not isinstance(claim, str) or not claim.strip():
            continue

        if not isinstance(evidence, list):
            evidence = []

        print(
            f"\nVerifying Claim {claim_id}: {claim}"
        )

        verification_results = verify_claim_evidence(
            claim,
            evidence,
            model_bundle=model_bundle
        )

        verified_claims.append({

            "claim_id": claim_id,

            "claim": claim,

            "source_text": claim_item.get(
                "source_text",
                ""
            ),

            "verification_results": verification_results,

            "total_evidence_verified": len(
                verification_results
            )

        })

        print(
            "Evidence verified:",
            len(verification_results)
        )

    return {
        "claims": verified_claims
    }


# ==================================================
# TERMINAL INTERFACE
# ==================================================

def main():

    import json

    from evidence_retrieval import retrieve_evidence
    from evidence_ranker import rank_evidence

    print("\n" + "=" * 55)

    print("              VERI-CLAIM-AI")

    print("             NLI VERIFICATION")

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

    # STEP 1: Retrieve evidence.
    retrieval_result = retrieve_evidence(claims)

    # STEP 2: Rank evidence.
    ranking_result = rank_evidence(retrieval_result)

    # STEP 3: Verify using DeBERTa.
    verification_result = verify_claims(
        ranking_result
    )

    print("\n" + "=" * 55)

    print("          NLI VERIFICATION RESULTS")

    print("=" * 55)

    print(
        json.dumps(
            verification_result,
            indent=4,
            ensure_ascii=False
        )
    )


# ==================================================
# APPLICATION ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()