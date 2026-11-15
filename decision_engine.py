"""
VERI-CLAIM-AI
Evidence-Based Decision Engine

Classifications:
1. SUPPORTED
2. CONTRADICTED
3. UNVERIFIABLE
"""

from urllib.parse import urlparse


# ==================================================
# CONFIGURATION
# ==================================================

MIN_SUPPORT_CONFIDENCE = 0.80

MIN_CONTRADICTION_CONFIDENCE = 0.90

MIN_SIMILARITY_SCORE = 0.40

MIN_SUPPORTING_EVIDENCE = 2

MIN_CONTRADICTING_EVIDENCE = 2

MIN_INDEPENDENT_SOURCES = 2


# ==================================================
# SOURCE DOMAIN EXTRACTION
# ==================================================

def get_source_domain(evidence):

    if isinstance(evidence, dict):

        url = evidence.get("url", "")

    else:

        url = evidence

    if not isinstance(url, str) or not url.strip():

        return None

    try:

        parsed = urlparse(url)

        domain = parsed.netloc.lower()

        if domain.startswith("www."):

            domain = domain[4:]

        return domain or None

    except Exception:

        return None


# ==================================================
# EVIDENCE VALIDATION
# ==================================================

def validate_evidence(evidence):

    if not isinstance(evidence, dict):

        return False

    label = evidence.get("nli_label")

    if label not in {

        "entailment",

        "contradiction",

        "neutral"

    }:

        return False

    try:

        confidence = float(
            evidence.get("nli_confidence")
        )

        similarity = float(
            evidence.get("similarity_score")
        )

    except (TypeError, ValueError):

        return False

    if not 0 <= confidence <= 1:

        return False

    if not 0 <= similarity <= 1:

        return False

    return True


# ==================================================
# FILTER STRONG EVIDENCE
# ==================================================

def filter_strong_evidence(evidence):

    strong_evidence = []

    for item in evidence:

        if not validate_evidence(item):

            continue

        label = item["nli_label"]

        confidence = float(
            item["nli_confidence"]
        )

        similarity = float(
            item["similarity_score"]
        )

        if similarity < MIN_SIMILARITY_SCORE:

            continue

        if (
            label == "entailment"
            and confidence < MIN_SUPPORT_CONFIDENCE
        ):

            continue

        if (
            label == "contradiction"
            and confidence < MIN_CONTRADICTION_CONFIDENCE
        ):

            continue

        if (
            label == "neutral"
            and confidence < MIN_SUPPORT_CONFIDENCE
        ):

            continue

        strong_evidence.append(item)

    return strong_evidence


# ==================================================
# COUNT INDEPENDENT SOURCES
# ==================================================

def count_independent_sources(evidence):

    domains = set()

    for item in evidence:

        domain = get_source_domain(item)

        if domain:

            domains.add(domain)

    return len(domains)


# ==================================================
# DECISION CONFIDENCE
# ==================================================

def calculate_decision_confidence(evidence):

    if not evidence:

        return 0.0

    scores = [

        float(item["nli_confidence"])

        for item in evidence

    ]

    return round(
        sum(scores) / len(scores),
        4
    )


# ==================================================
# PREPARE EVIDENCE DETAILS
# ==================================================

def prepare_evidence_details(evidence):

    details = {

        "supporting": [],

        "contradicting": [],

        "neutral": []

    }

    for item in evidence:

        label = item.get("nli_label")

        information = {

            "evidence_id": item.get("evidence_id"),

            "source": item.get("source"),

            "title": item.get("title"),

            "url": item.get("url"),

            "evidence_text": item.get(
                "evidence_text",
                item.get("content", "")
            ),

            "similarity_score": item.get(
                "similarity_score"
            ),

            "nli_confidence": item.get(
                "nli_confidence"
            )

        }

        if label == "entailment":

            details["supporting"].append(
                information
            )

        elif label == "contradiction":

            details["contradicting"].append(
                information
            )

        elif label == "neutral":

            details["neutral"].append(
                information
            )

    return details


# ==================================================
# DECISION EXPLANATION
# ==================================================

def generate_decision_explanation(decision):

    explanations = {

        "SUPPORTED": (
            "Multiple independent sources provide "
            "strong evidence supporting the claim."
        ),

        "CONTRADICTED": (
            "Multiple independent sources provide "
            "strong evidence contradicting the claim."
        ),

        "UNVERIFIABLE": (
            "Available evidence is insufficient, "
            "conflicting, or lacks adequate independent "
            "source support."
        )

    }

    return explanations.get(
        decision,
        "Decision could not be established."
    )


# ==================================================
# DECIDE ONE CLAIM
# ==================================================

def decide_claim(claim_data):

    if not isinstance(claim_data, dict):

        raise ValueError(
            "Claim data must be a dictionary."
        )

    claim = claim_data.get("claim", "")

    if not isinstance(claim, str) or not claim.strip():

        raise ValueError(
            "Claim must be a non-empty string."
        )

    evidence = claim_data.get(
        "verification_results",
        []
    )

    if not isinstance(evidence, list):

        evidence = []

    strong_evidence = filter_strong_evidence(
        evidence
    )

    supporting = [

        item for item in strong_evidence

        if item["nli_label"] == "entailment"

    ]

    contradicting = [

        item for item in strong_evidence

        if item["nli_label"] == "contradiction"

    ]

    neutral = [

        item for item in strong_evidence

        if item["nli_label"] == "neutral"

    ]

    support_sources = count_independent_sources(
        supporting
    )

    contradiction_sources = count_independent_sources(
        contradicting
    )

    # Default decision
    decision = "UNVERIFIABLE"

    # Evidence used for confidence calculation
    selected_evidence = []

    # ----------------------------------------------
    # CONFLICTING EVIDENCE
    # ----------------------------------------------

    if supporting and contradicting:

        decision = "UNVERIFIABLE"

        selected_evidence = strong_evidence

    # ----------------------------------------------
    # SUPPORTED
    # ----------------------------------------------

    elif (

        len(supporting) >= MIN_SUPPORTING_EVIDENCE

        and support_sources >= MIN_INDEPENDENT_SOURCES

    ):

        decision = "SUPPORTED"

        selected_evidence = supporting

    # ----------------------------------------------
    # CONTRADICTED
    # ----------------------------------------------

    elif (

        len(contradicting) >= MIN_CONTRADICTING_EVIDENCE

        and contradiction_sources >= MIN_INDEPENDENT_SOURCES

    ):

        decision = "CONTRADICTED"

        selected_evidence = contradicting

    # ----------------------------------------------
    # INSUFFICIENT EVIDENCE
    # ----------------------------------------------

    else:

        decision = "UNVERIFIABLE"

        selected_evidence = strong_evidence

    # ----------------------------------------------
    # CONFIDENCE
    # ----------------------------------------------

    evidence_confidence = calculate_decision_confidence(
        selected_evidence
    )

    # ----------------------------------------------
    # EVIDENCE IDS
    # ----------------------------------------------

    supporting_ids = [

        item.get("evidence_id")

        for item in supporting

    ]

    contradicting_ids = [

        item.get("evidence_id")

        for item in contradicting

    ]

    # ----------------------------------------------
    # FINAL RESULT
    # ----------------------------------------------

    return {

        "claim_id": claim_data.get("claim_id"),

        "claim": claim,

        "decision": decision,

        "evidence_confidence": evidence_confidence,

        "decision_reason": generate_decision_explanation(
            decision
        ),

        "evidence_summary": {

            "total_evidence_received": len(evidence),

            "strong_evidence_count": len(
                strong_evidence
            ),

            "supporting_evidence_count": len(
                supporting
            ),

            "contradicting_evidence_count": len(
                contradicting
            ),

            "neutral_evidence_count": len(
                neutral
            ),

            "independent_supporting_sources": (
                support_sources
            ),

            "independent_contradicting_sources": (
                contradiction_sources
            )

        },

        "supporting_evidence_ids": supporting_ids,

        "contradicting_evidence_ids": contradicting_ids,

        "evidence_details": prepare_evidence_details(
            evidence
        )

    }


# ==================================================
# DECIDE MULTIPLE CLAIMS
# ==================================================

def decide_claims(verification_result):

    if not isinstance(verification_result, dict):

        raise ValueError(
            "Verification result must be a dictionary."
        )

    claims = verification_result.get(
        "claims",
        []
    )

    if not isinstance(claims, list):

        raise ValueError(
            "Claims must be a list."
        )

    decisions = []

    for claim_data in claims:

        if not isinstance(claim_data, dict):

            continue

        try:

            result = decide_claim(claim_data)

            decisions.append(result)

        except ValueError:

            continue

    summary = {

        "total_claims": len(decisions),

        "supported": sum(

            item["decision"] == "SUPPORTED"

            for item in decisions

        ),

        "contradicted": sum(

            item["decision"] == "CONTRADICTED"

            for item in decisions

        ),

        "unverifiable": sum(

            item["decision"] == "UNVERIFIABLE"

            for item in decisions

        )

    }

    return {

        "summary": summary,

        "decisions": decisions

    }


# ==================================================
# TERMINAL TESTING
# ==================================================

def main():

    import json

    sample = {

        "claims": [

            {

                "claim_id": 1,

                "claim": "The Earth revolves around the Sun.",

                "verification_results": [

                    {

                        "evidence_id": 1,

                        "source": "NASA",

                        "title": "Earth and the Sun",

                        "url": "https://www.nasa.gov/example",

                        "evidence_text": (
                            "The Earth revolves around the Sun."
                        ),

                        "similarity_score": 0.90,

                        "nli_label": "entailment",

                        "nli_confidence": 0.98

                    },

                    {

                        "evidence_id": 2,

                        "source": "Britannica",

                        "title": "Earth",

                        "url": "https://www.britannica.com/example",

                        "evidence_text": (
                            "Earth orbits the Sun."
                        ),

                        "similarity_score": 0.85,

                        "nli_label": "entailment",

                        "nli_confidence": 0.96

                    }

                ]

            }

        ]

    }

    result = decide_claims(sample)

    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False
        )
    )


if __name__ == "__main__":

    main()