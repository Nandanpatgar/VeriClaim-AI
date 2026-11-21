
"""
VERI-CLAIM-AI
Evidence-Based Decision Engine

Classifications:
1. SUPPORTED
2. CONTRADICTED
3. UNVERIFIABLE

Integrates:
- NLI verification
- Semantic similarity
- Source reliability
- Independent domain verification
"""

import math

from source_reliability import (
    evaluate_single_source,
    get_source_domain as reliability_get_source_domain
)


# ============================================================
# CONFIGURATION
# ============================================================

MIN_SUPPORT_CONFIDENCE = 0.80

MIN_CONTRADICTION_CONFIDENCE = 0.90

MIN_SIMILARITY_SCORE = 0.40

MIN_SOURCE_RELIABILITY = 0.50

# Minimum evidence requirements
MIN_SUPPORTING_EVIDENCE = 1

MIN_CONTRADICTING_EVIDENCE = 1

# Require independent corroboration before assigning a definitive verdict.
MIN_INDEPENDENT_SOURCES = 2
# ============================================================
# SOURCE DOMAIN EXTRACTION
# ============================================================

def get_source_domain(evidence):
    """
    Extract normalized domain from evidence or URL.

    Example:
        https://www.wikipedia.org/wiki/example
        -> wikipedia.org
    """

    url = evidence.get("url", "") if isinstance(evidence, dict) else evidence
    return reliability_get_source_domain(url) or None


# ============================================================
# EVIDENCE RELIABILITY ENRICHMENT
# ============================================================

def enrich_evidence_reliability(evidence):
    """
    Add source reliability information to evidence.

    Preserves existing evidence fields.
    """

    if not isinstance(evidence, dict):
        return None

    enriched = evidence.copy()

    try:
        reliability_data = evaluate_single_source(enriched)

        enriched.update({

            "domain": reliability_data.get("domain", ""),

            "source_type": reliability_data.get(
                "source_type",
                "unknown"
            ),

            "reliability_score": reliability_data.get(
                "reliability_score",
                0.0
            ),

            "is_trusted": reliability_data.get(
                "is_trusted",
                False
            ),

            "uses_https": reliability_data.get(
                "uses_https",
                False
            )

        })

    except (ValueError, TypeError):
        enriched.update({

            "domain": get_source_domain(enriched) or "",

            "source_type": "unknown",

            "reliability_score": 0.0,

            "is_trusted": False,

            "uses_https": False

        })

    return enriched


# ============================================================
# EVIDENCE VALIDATION
# ============================================================

def validate_evidence(evidence):
    """
    Validate NLI prediction and semantic similarity.
    """

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

    if not math.isfinite(confidence):
        return False

    if not math.isfinite(similarity):
        return False

    if not 0 <= confidence <= 1:
        return False

    if not 0 <= similarity <= 1:
        return False

    return True


# ============================================================
# FILTER STRONG EVIDENCE
# ============================================================

def filter_strong_evidence(evidence):
    """
    Select evidence satisfying:

    1. Valid NLI classification.
    2. Minimum NLI confidence.
    3. Minimum semantic similarity.
    4. Minimum source reliability.
    """

    if not isinstance(evidence, list):
        raise ValueError("Evidence must be a list.")

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

        # Semantic relevance filter.
        if similarity < MIN_SIMILARITY_SCORE:
            continue

        # NLI confidence filter.
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

        # Source reliability evaluation.
        enriched = enrich_evidence_reliability(item)

        if enriched is None:
            continue

        reliability = enriched.get(
            "reliability_score",
            0.0
        )

        if reliability < MIN_SOURCE_RELIABILITY:
            continue

        strong_evidence.append(enriched)

    return strong_evidence


# ============================================================
# COUNT INDEPENDENT SOURCES
# ============================================================

def count_independent_sources(evidence):
    """
    Count unique source domains.

    Multiple articles from the same domain count as
    one independent source.
    """

    domains = set()

    for item in evidence:

        domain = get_source_domain(item)

        if domain:
            domains.add(domain)

    return len(domains)


# ============================================================
# DECISION CONFIDENCE
# ============================================================

def calculate_decision_confidence(evidence):
    """
    Calculate confidence using both NLI confidence
    and source reliability.

    Per-evidence confidence:

        NLI confidence * source reliability

    Final confidence:
        Mean adjusted confidence across selected evidence.

    This is a heuristic decision score, not a calibrated
    probability that the claim is true.
    """

    if not evidence:
        return 0.0

    scores = []

    for item in evidence:

        try:
            nli_confidence = float(
                item["nli_confidence"]
            )

            reliability = item.get(
                "reliability_score"
            )

            if reliability is None:
                enriched = enrich_evidence_reliability(item)

                reliability = (
                    enriched.get("reliability_score", 0.0)
                    if enriched
                    else 0.0
                )

            reliability = float(reliability)

            if not math.isfinite(nli_confidence):
                continue

            if not math.isfinite(reliability):
                continue

            adjusted_score = (
                nli_confidence * reliability
            )

            scores.append(adjusted_score)

        except (KeyError, TypeError, ValueError):
            continue

    if not scores:
        return 0.0

    return round(
        sum(scores) / len(scores),
        4
    )


# ============================================================
# PREPARE EVIDENCE DETAILS
# ============================================================

def prepare_evidence_details(evidence):
    """
    Organize evidence by NLI classification.

    Includes source reliability metadata.
    """

    details = {
        "supporting": [],
        "contradicting": [],
        "neutral": []
    }

    for item in evidence:

        if not isinstance(item, dict):
            continue

        label = item.get("nli_label")

        enriched = enrich_evidence_reliability(item)

        if enriched is None:
            continue

        information = {

            "evidence_id": enriched.get("evidence_id"),

            "source": enriched.get("source"),

            "title": enriched.get("title"),

            "url": enriched.get("url"),

            "domain": enriched.get("domain"),

            "source_type": enriched.get("source_type"),

            "reliability_score": enriched.get(
                "reliability_score"
            ),

            "is_trusted": enriched.get("is_trusted"),

            "evidence_text": enriched.get(
                "evidence_text",
                enriched.get("content", "")
            ),

            "similarity_score": enriched.get(
                "similarity_score"
            ),

            "semantic_similarity_score": enriched.get(
                "semantic_similarity_score"
            ),

            "lexical_overlap_score": enriched.get(
                "lexical_overlap_score"
            ),

            "retrieval_provider": enriched.get(
                "retrieval_provider"
            ),

            "published_date": enriched.get(
                "published_date"
            ),

            "nli_confidence": enriched.get(
                "nli_confidence"
            ),

            "nli_probabilities": enriched.get(
                "nli_probabilities",
                {}
            ),

            "nli_label": label

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


# ============================================================
# DECISION EXPLANATION
# ============================================================

def generate_decision_explanation(decision):

    explanations = {

        "SUPPORTED": (
            "Multiple independent sources provide "
            "strong, sufficiently reliable evidence "
            "supporting the claim."
        ),

        "CONTRADICTED": (
            "Multiple independent sources provide "
            "strong, sufficiently reliable evidence "
            "contradicting the claim."
        ),

        "UNVERIFIABLE": (
            "Available evidence is insufficient, conflicting, "
            "or lacks adequate relevance, reliability, "
            "or independent source support."
        )

    }

    return explanations.get(
        decision,
        "Decision could not be established."
    )


# ============================================================
# DECIDE ONE CLAIM
# ============================================================

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

    # Evaluate and filter evidence.
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

    # Default decision.
    decision = "UNVERIFIABLE"

    selected_evidence = []

    # --------------------------------------------------------
    # CONFLICTING EVIDENCE
    # --------------------------------------------------------

    if supporting and contradicting:

        decision = "UNVERIFIABLE"

        selected_evidence = strong_evidence

    # --------------------------------------------------------
    # SUPPORTED
    # --------------------------------------------------------

    elif (

        len(supporting) >= MIN_SUPPORTING_EVIDENCE

        and support_sources >= MIN_INDEPENDENT_SOURCES

    ):

        decision = "SUPPORTED"

        selected_evidence = supporting

    # --------------------------------------------------------
    # CONTRADICTED
    # --------------------------------------------------------

    elif (

        len(contradicting) >= MIN_CONTRADICTING_EVIDENCE

        and contradiction_sources >= MIN_INDEPENDENT_SOURCES

    ):

        decision = "CONTRADICTED"

        selected_evidence = contradicting

    # --------------------------------------------------------
    # INSUFFICIENT EVIDENCE
    # --------------------------------------------------------

    else:

        decision = "UNVERIFIABLE"

        selected_evidence = strong_evidence

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    evidence_confidence = calculate_decision_confidence(
        selected_evidence
    )

    # --------------------------------------------------------
    # EVIDENCE IDS
    # --------------------------------------------------------

    supporting_ids = [

        item.get("evidence_id")

        for item in supporting

    ]

    contradicting_ids = [

        item.get("evidence_id")

        for item in contradicting

    ]

    # --------------------------------------------------------
    # SOURCE RELIABILITY SUMMARY
    # --------------------------------------------------------

    trusted_sources = sum(

        1

        for item in strong_evidence

        if item.get("is_trusted", False)

    )

    unique_domains = count_independent_sources(
        strong_evidence
    )

    selected_source_ids = {
        item.get("evidence_id")
        for item in strong_evidence
        if item.get("evidence_id") is not None
    }
    excluded_evidence = [
        item for item in evidence
        if isinstance(item, dict)
        and item.get("evidence_id") not in selected_source_ids
    ]

    if supporting and contradicting:
        decision_reason = (
            "Relevant evidence supports and contradicts this claim, "
            "so the available sources do not establish one conclusion."
        )
    elif decision == "UNVERIFIABLE" and (supporting or contradicting):
        available_sources = max(support_sources, contradiction_sources)
        decision_reason = (
            "Some relevant evidence was found, but it is supported by "
            f"only {available_sources} independent source(s); at least "
            f"{MIN_INDEPENDENT_SOURCES} are required for a definitive verdict."
        )
    elif decision == "UNVERIFIABLE":
        decision_reason = (
            "No evidence passed the relevance, NLI-confidence, and source "
            "reliability checks required for a definitive verdict."
        )
    else:
        decision_reason = generate_decision_explanation(decision)

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    return {

        "claim_id": claim_data.get("claim_id"),

        "claim": claim,

        "source_text": claim_data.get("source_text", ""),

        "source_start": claim_data.get("source_start"),

        "source_end": claim_data.get("source_end"),

        "decision": decision,

        "retrieval_status": claim_data.get(
            "retrieval_status",
            {"state": "unknown", "warnings": []}
        ),

        "evidence_confidence": evidence_confidence,

        "decision_reason": decision_reason,

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
            ),

            "trusted_sources": trusted_sources,

            "unique_domains": unique_domains,

            "filtered_evidence_count": (
                len(evidence) - len(strong_evidence)
            )

        },

        "supporting_evidence_ids": supporting_ids,

        "contradicting_evidence_ids": contradicting_ids,

        "evidence_details": prepare_evidence_details(
            strong_evidence
        ),

        "excluded_evidence_details": prepare_evidence_details(
            excluded_evidence
        )

    }


# ============================================================
# DECIDE MULTIPLE CLAIMS
# ============================================================

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


# ============================================================
# TERMINAL TESTING
# ============================================================

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


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
