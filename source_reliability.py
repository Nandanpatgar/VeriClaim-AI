
"""
VERI-CLAIM-AI
Source Reliability Module

Responsibilities:
1. Identify evidence source domains.
2. Classify source types.
3. Calculate transparent heuristic reliability scores.
4. Detect duplicate sources.
5. Measure independent source diversity.
6. Preserve original evidence metadata.
"""

import re
from urllib.parse import urlparse


# ============================================================
# CONFIGURATION
# ============================================================

RELIABILITY_THRESHOLD = 0.60

SOURCE_WEIGHTS = {
    "government": 0.95,
    "academic": 0.90,
    "scientific": 0.90,
    "encyclopedia": 0.85,
    "established_news": 0.78,
    "organization": 0.70,
    "commercial": 0.58,
    "blog": 0.40,
    "unknown": 0.50
}

HTTPS_BONUS = 0.02


# ============================================================
# DOMAIN EXTRACTION
# ============================================================

def get_source_domain(url):
    """
    Extract normalized domain from a URL.

    Examples:
        https://www.bbc.com/news -> bbc.com
        https://en.wikipedia.org/wiki/example
            -> en.wikipedia.org
    """

    if not isinstance(url, str) or not url.strip():
        return ""

    try:
        parsed = urlparse(url.strip())

        domain = parsed.netloc.lower()

        if not domain:
            return ""

        if "@" in domain:
            domain = domain.rsplit("@", 1)[-1]

        if domain.startswith("["):
            return domain.split("]")[0] + "]"

        if domain.count(":") == 1:
            domain = domain.split(":")[0]

        domain = domain.removeprefix("www.")

        return domain.rstrip(".")

    except (ValueError, AttributeError):
        return ""


# ============================================================
# DOMAIN VALIDATION
# ============================================================

def is_valid_domain(domain):
    if not isinstance(domain, str) or not domain:
        return False

    if len(domain) > 253:
        return False

    if domain.startswith(".") or domain.endswith("."):
        return False

    if ".." in domain:
        return False

    labels = domain.split(".")

    if len(labels) < 2:
        return False

    for label in labels:

        if not label:
            return False

        if len(label) > 63:
            return False

        if label.startswith("-") or label.endswith("-"):
            return False

        if not re.fullmatch(r"[a-z0-9-]+", label):
            return False

    return True


# ============================================================
# SOURCE TYPE CLASSIFICATION
# ============================================================

def classify_source_type(url, source=""):
    """
    Classify evidence sources using domain patterns.

    Classification is heuristic, not a guarantee of accuracy.
    """

    domain = get_source_domain(url)

    if not domain:
        return "unknown"

    # Government sources.
    if (
        domain.endswith(".gov")
        or domain.endswith(".gov.in")
        or domain.endswith(".gov.uk")
        or domain.endswith(".gc.ca")
    ):
        return "government"

    # Academic institutions.
    if (
        domain.endswith(".edu")
        or domain.endswith(".edu.in")
        or domain.endswith(".ac.in")
        or domain.endswith(".ac.uk")
        or domain.endswith(".edu.au")
    ):
        return "academic"

    # Scientific and research publishing platforms.
    scientific_domains = (
        "nature.com",
        "science.org",
        "sciencedirect.com",
        "springer.com",
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "arxiv.org",
        "ieee.org",
        "acm.org",
        "plos.org",
        "frontiersin.org",
        "tandfonline.com",
        "wiley.com"
    )

    if any(
        domain == item or domain.endswith("." + item)
        for item in scientific_domains
    ):
        return "scientific"

    # Encyclopedias.
    encyclopedia_domains = (
        "wikipedia.org",
        "britannica.com",
        "encyclopedia.com"
    )

    if any(
        domain == item or domain.endswith("." + item)
        for item in encyclopedia_domains
    ):
        return "encyclopedia"

    # Established news organizations.
    news_domains = (
        "reuters.com",
        "bbc.com",
        "bbc.co.uk",
        "apnews.com",
        "theguardian.com",
        "nytimes.com",
        "washingtonpost.com",
        "aljazeera.com",
        "npr.org",
        "indianexpress.com",
        "thehindu.com",
        "ndtv.com"
    )

    if any(
        domain == item or domain.endswith("." + item)
        for item in news_domains
    ):
        return "established_news"

    # Blogs and publishing platforms.
    blog_domains = (
        "medium.com",
        "blogspot.com",
        "wordpress.com",
        "substack.com",
        "tumblr.com"
    )

    if any(
        domain == item or domain.endswith("." + item)
        for item in blog_domains
    ):
        return "blog"

    # Organization domains.
    if domain.endswith(".org"):
        return "organization"

    # Commercial websites.
    if domain.endswith(".com") or domain.endswith(".net"):
        return "commercial"

    return "unknown"


# ============================================================
# HTTPS VALIDATION
# ============================================================

def uses_https(url):
    if not isinstance(url, str):
        return False

    try:
        return urlparse(url.strip()).scheme.lower() == "https"
    except ValueError:
        return False


# ============================================================
# RELIABILITY SCORE
# ============================================================

def calculate_reliability_score(url, source=""):
    """
    Calculate a transparent heuristic reliability score.

    Score range: 0.0 to 1.0.

    The score is based on:
        Source category
        HTTPS usage

    It is NOT a factual accuracy probability.
    """

    domain = get_source_domain(url)

    if not domain or not is_valid_domain(domain):
        return SOURCE_WEIGHTS["unknown"]

    source_type = classify_source_type(url, source)

    score = SOURCE_WEIGHTS.get(
        source_type,
        SOURCE_WEIGHTS["unknown"]
    )

    if uses_https(url):
        score += HTTPS_BONUS

    return round(min(score, 1.0), 4)


# ============================================================
# SOURCE TRUST EVALUATION
# ============================================================

def is_trusted_source(reliability_score):
    try:
        score = float(reliability_score)
    except (ValueError, TypeError):
        return False

    return score >= RELIABILITY_THRESHOLD


# ============================================================
# EVALUATE SINGLE EVIDENCE SOURCE
# ============================================================

def evaluate_single_source(evidence):
    """
    Add reliability information to one evidence record.
    """

    if not isinstance(evidence, dict):
        raise ValueError("Evidence must be a dictionary.")

    result = evidence.copy()

    url = result.get("url", "")
    source = result.get("source", "")

    domain = get_source_domain(url)

    source_type = classify_source_type(
        url,
        source
    )

    reliability_score = calculate_reliability_score(
        url,
        source
    )

    result.update({

        "domain": domain,

        "source_type": source_type,

        "reliability_score": reliability_score,

        "is_trusted": is_trusted_source(
            reliability_score
        ),

        "uses_https": uses_https(url)

    })

    return result


# ============================================================
# EVALUATE MULTIPLE SOURCES
# ============================================================

def evaluate_source_reliability(evidence):
    """
    Evaluate reliability of multiple evidence records.

    Returns a list preserving original evidence metadata.
    """

    if not isinstance(evidence, list):
        raise ValueError("Evidence must be provided as a list.")

    evaluated = []

    for item in evidence:

        if not isinstance(item, dict):
            continue

        evaluated.append(
            evaluate_single_source(item)
        )

    return evaluated


# Backward-compatible alias.
def assess_source_reliability(evidence):
    return evaluate_source_reliability(evidence)


# ============================================================
# SOURCE DIVERSITY
# ============================================================

def calculate_source_diversity(evidence):
    """
    Calculate the number of independent source domains.

    Multiple articles from the same domain count as one source.
    """

    if not isinstance(evidence, list):
        raise ValueError("Evidence must be a list.")

    domains = set()

    for item in evidence:

        if not isinstance(item, dict):
            continue

        domain = item.get("domain", "")

        if not domain:
            domain = get_source_domain(
                item.get("url", "")
            )

        if domain and is_valid_domain(domain):
            domains.add(domain)

    return {
        "unique_domains": len(domains),
        "domains": sorted(domains)
    }


# ============================================================
# REMOVE DUPLICATE SOURCES
# ============================================================

def remove_duplicate_sources(evidence):
    """
    Remove duplicate evidence URLs.

    Different URLs from the same domain are preserved because
    they may contain different evidence.
    """

    if not isinstance(evidence, list):
        raise ValueError("Evidence must be a list.")

    unique = []
    seen = set()

    for item in evidence:

        if not isinstance(item, dict):
            continue

        url = item.get("url", "")

        normalized_url = (
            url.strip().lower().rstrip("/")
            if isinstance(url, str)
            else ""
        )

        if normalized_url and normalized_url in seen:
            continue

        if normalized_url:
            seen.add(normalized_url)

        unique.append(item)

    return unique


# ============================================================
# COMPLETE SOURCE ANALYSIS
# ============================================================

def analyze_sources(evidence):
    """
    Perform complete source reliability analysis.

    Returns:
        {
            "evidence": [...],
            "total_sources": int,
            "trusted_sources": int,
            "unique_domains": int,
            "source_diversity": {...},
            "average_reliability": float
        }
    """

    if not isinstance(evidence, list):
        raise ValueError("Evidence must be a list.")

    unique_evidence = remove_duplicate_sources(
        evidence
    )

    evaluated = evaluate_source_reliability(
        unique_evidence
    )

    diversity = calculate_source_diversity(
        evaluated
    )

    trusted_count = sum(
        1
        for item in evaluated
        if item["is_trusted"]
    )

    if evaluated:

        average_reliability = round(
            sum(
                item["reliability_score"]
                for item in evaluated
            ) / len(evaluated),
            4
        )

    else:
        average_reliability = 0.0

    return {

        "evidence": evaluated,

        "total_sources": len(evaluated),

        "trusted_sources": trusted_count,

        "unique_domains": diversity["unique_domains"],

        "source_diversity": diversity,

        "average_reliability": average_reliability

    }


# ============================================================
# VERIFY SOURCE RELIABILITY FOR MULTIPLE CLAIMS
# ============================================================

def evaluate_claim_sources(retrieval_result):
    """
    Evaluate sources for every claim.

    Input:
        {
            "claims": [
                {
                    "claim_id": 1,
                    "claim": "...",
                    "evidence": [...]
                }
            ]
        }
    """

    if not isinstance(retrieval_result, dict):
        raise ValueError("Input must be a dictionary.")

    claims = retrieval_result.get("claims")

    if not isinstance(claims, list):
        raise ValueError("Input must contain a claims list.")

    if not claims:
        return {"claims": []}

    evaluated_claims = []

    for claim_data in claims:

        if not isinstance(claim_data, dict):
            continue

        evidence = claim_data.get("evidence", [])

        if not isinstance(evidence, list):
            evidence = []

        analysis = analyze_sources(evidence)

        evaluated_claims.append({

            **claim_data,

            "evidence": analysis["evidence"],

            "source_analysis": {

                "total_sources": analysis["total_sources"],

                "trusted_sources": analysis["trusted_sources"],

                "unique_domains": analysis["unique_domains"],

                "average_reliability": analysis["average_reliability"],

                "source_diversity": analysis["source_diversity"]

            }

        })

    return {
        "claims": evaluated_claims
    }


# ============================================================
# TERMINAL TESTING
# ============================================================

def main():

    import json

    print("\n" + "=" * 55)
    print("              VERI-CLAIM-AI")
    print("              SOURCE RELIABILITY")
    print("=" * 55)

    evidence = [

        {
            "evidence_id": 1,
            "source": "Wikipedia",
            "title": "Albert Einstein",
            "content": "Albert Einstein was a physicist.",
            "url": "https://en.wikipedia.org/wiki/Albert_Einstein"
        },

        {
            "evidence_id": 2,
            "source": "Scientific Journal",
            "title": "Research Article",
            "content": "Scientific research evidence.",
            "url": "https://www.nature.com/"
        }

    ]

    result = analyze_sources(evidence)

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
