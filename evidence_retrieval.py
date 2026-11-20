
import os
import re
import hashlib
from urllib.parse import urlparse

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ============================================================
# CONFIGURATION
# ============================================================

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
TAVILY_API_URL = "https://api.tavily.com/search"

REQUEST_TIMEOUT = 15
MAX_WIKIPEDIA_RESULTS = 5
MAX_TAVILY_RESULTS = 5

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


# ============================================================
# CUSTOM EXCEPTION
# ============================================================

class EvidenceRetrievalError(ValueError):
    pass


# ============================================================
# INPUT VALIDATION
# ============================================================

def normalize_claim(claim):
    if not isinstance(claim, str):
        return ""

    return re.sub(r"\s+", " ", claim).strip()


def validate_claim(claim):
    normalized = normalize_claim(claim)

    if not normalized:
        raise ValueError("Claim cannot be empty.")

    if len(normalized) < 3:
        raise ValueError("Claim is too short.")

    return normalized


def tokenize(text):
    if not isinstance(text, str):
        return []

    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


# ============================================================
# SEARCH QUERY GENERATION
# ============================================================

def generate_search_queries(claim):
    claim = validate_claim(claim)

    queries = [claim]

    cleaned = re.sub(
        r"\b(is|are|was|were|the|a|an)\b",
        " ",
        claim,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if cleaned and cleaned not in queries:
        queries.append(cleaned)

    return queries


# ============================================================
# RELEVANCE SCORING
# ============================================================

def calculate_relevance_score(claim, content):
    claim_tokens = set(tokenize(claim))
    content_tokens = set(tokenize(content))

    if not claim_tokens or not content_tokens:
        return 0.0

    overlap = len(claim_tokens.intersection(content_tokens))

    return round(overlap / len(claim_tokens), 4)


# ============================================================
# URL UTILITIES
# ============================================================

def normalize_url(url):
    if not isinstance(url, str):
        return ""

    url = url.strip()

    if not url:
        return ""

    parsed = urlparse(url)

    if not parsed.scheme or not parsed.netloc:
        return url.rstrip("/").lower()

    return (
        f"{parsed.scheme.lower()}://"
        f"{parsed.netloc.lower()}"
        f"{parsed.path.rstrip('/')}"
    )


def get_domain(url):
    if not isinstance(url, str):
        return ""

    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def source_priority(source):
    priorities = {
        "Wikipedia": 1,
        "Tavily": 2
    }

    return priorities.get(source, 99)


# ============================================================
# EVIDENCE IDENTIFICATION
# ============================================================

def make_evidence_id(content, url=""):
    raw = f"{content}|{url}"

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:16]


def create_evidence(
    source,
    title,
    content,
    url,
    score=None,
    retrieval_provider=None
):
    content = content.strip() if isinstance(content, str) else ""
    title = title.strip() if isinstance(title, str) else ""
    url = url.strip() if isinstance(url, str) else ""

    return {
        "evidence_id": make_evidence_id(content, url),
        "source": source,
        "title": title,
        "content": content,
        "url": url,
        "score": score,
        "retrieval_provider": retrieval_provider or source
    }


# ============================================================
# WIKIPEDIA RETRIEVAL
# ============================================================

def retrieve_wikipedia_evidence(claim):
    claim = validate_claim(claim)

    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": claim,
        "gsrlimit": MAX_WIKIPEDIA_RESULTS,
        "prop": "extracts|info",
        "exintro": True,
        "explaintext": True,
        "inprop": "url",
        "format": "json"
    }

    try:
        response = requests.get(
            WIKIPEDIA_API_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
            headers={
                "User-Agent": "Veri-Claim-AI/1.0"
            }
        )

        response.raise_for_status()

        data = response.json()

        if (
            not isinstance(data, dict)
            or not isinstance(data.get("query"), dict)
            or not isinstance(data["query"].get("pages"), (dict, list))
        ):
            raise EvidenceRetrievalError(
                "Wikipedia returned a response without a valid query/pages result."
            )

        pages = data.get("query", {}).get("pages", {})

        page_values = (
            pages if isinstance(pages, list)
            else pages.values()
        )

        evidence = []

        for page in page_values:
            title = page.get("title", "")
            content = page.get("extract", "")
            url = page.get("fullurl", "")

            if not content:
                continue

            evidence.append(
                create_evidence(
                    source="Wikipedia",
                    title=title,
                    content=content,
                    url=url,
                    score=None,
                    retrieval_provider="Wikipedia"
                )
            )

        return evidence

    except requests.RequestException as error:
        raise EvidenceRetrievalError(
            f"Wikipedia retrieval failed: {error}"
        ) from error

    except EvidenceRetrievalError:
        raise

    except (ValueError, TypeError, AttributeError) as error:
        raise EvidenceRetrievalError(
            f"Wikipedia returned an invalid response: {error}"
        ) from error


def retrieve_wikipedia(claim):
    return retrieve_wikipedia_evidence(claim)


# ============================================================
# TAVILY RETRIEVAL
# ============================================================

def retrieve_tavily_evidence(claim, api_key=None):
    claim = validate_claim(claim)

    key = api_key if api_key is not None else TAVILY_API_KEY

    if not key:
        return []

    payload = {
        "query": claim,
        "search_depth": "advanced",
        "topic": "general",
        "max_results": MAX_TAVILY_RESULTS,
        "include_answer": False,
        "include_raw_content": False
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }

    try:
        response = requests.post(
            TAVILY_API_URL,
            json=payload,
            headers=headers,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise EvidenceRetrievalError(
                "Tavily returned a response without a valid results list."
            )

        evidence = []

        for result in data.get("results", []):
            title = result.get("title", "")
            content = result.get("content", "")
            url = result.get("url", "")

            if not content:
                continue

            provider_score = result.get("score")

            try:
                if provider_score is not None:
                    provider_score = float(provider_score)
            except (ValueError, TypeError):
                provider_score = None

            evidence.append(
                create_evidence(
                    source="Tavily",
                    title=title,
                    content=content,
                    url=url,
                    score=provider_score,
                    retrieval_provider="Tavily"
                )
            )

        return evidence

    except requests.RequestException as error:
        raise EvidenceRetrievalError(
            f"Tavily retrieval failed: {error}"
        ) from error

    except EvidenceRetrievalError:
        raise

    except (ValueError, TypeError, AttributeError) as error:
        raise EvidenceRetrievalError(
            f"Tavily returned an invalid response: {error}"
        ) from error


def retrieve_tavily(claim, api_key=None):
    return retrieve_tavily_evidence(
        claim,
        api_key=api_key
    )


# ============================================================
# DUPLICATE REMOVAL
# ============================================================

def remove_duplicate_evidence(evidence):
    if not isinstance(evidence, list):
        return []

    unique_evidence = []
    seen_urls = set()
    seen_content = set()

    for item in evidence:
        if not isinstance(item, dict):
            continue

        url = normalize_url(item.get("url", ""))

        content = re.sub(
            r"\s+",
            " ",
            str(item.get("content", "")).lower()
        ).strip()

        if url and url in seen_urls:
            continue

        if content and content in seen_content:
            continue

        if url:
            seen_urls.add(url)

        if content:
            seen_content.add(content)

        unique_evidence.append(item)

    return unique_evidence


def deduplicate_evidence(evidence):
    return remove_duplicate_evidence(evidence)


# ============================================================
# SINGLE CLAIM RETRIEVAL
# ============================================================

def retrieve_evidence_for_claim(claim, api_key=None, include_status=False):
    claim = validate_claim(claim)

    provider_errors = []
    wikipedia_evidence = []
    tavily_evidence = []

    try:
        wikipedia_evidence = retrieve_wikipedia_evidence(claim)
    except EvidenceRetrievalError as error:
        provider_errors.append(str(error))

    # Maintain compatibility with one-argument test mocks
    # and existing callers when no explicit API key is supplied.
    key = api_key if api_key is not None else TAVILY_API_KEY
    tavily_enabled = bool(key)
    if tavily_enabled:
        try:
            if api_key is None:
                tavily_evidence = retrieve_tavily_evidence(claim)
            else:
                tavily_evidence = retrieve_tavily_evidence(
                    claim,
                    api_key=api_key
                )
        except EvidenceRetrievalError as error:
            provider_errors.append(str(error))

    successful_providers = 1 + int(tavily_enabled) - len(provider_errors)
    if successful_providers <= 0:
        raise EvidenceRetrievalError(
            "All configured evidence providers failed: "
            + "; ".join(provider_errors)
        )

    combined_evidence = (
        wikipedia_evidence + tavily_evidence
    )

    unique_evidence = remove_duplicate_evidence(
        combined_evidence
    )

    for index, item in enumerate(unique_evidence, start=1):
        item["evidence_id"] = index

        item["relevance_score"] = calculate_relevance_score(
            claim,
            item.get("content", "")
        )

    warnings = list(provider_errors)
    if not tavily_enabled:
        warnings.append(
            "Tavily is not configured; this claim was searched with Wikipedia only."
        )

    status = {
        "state": "partial" if warnings else "complete",
        "warnings": warnings,
        "wikipedia": "failed" if any("Wikipedia retrieval" in e or "Wikipedia returned" in e for e in provider_errors) else "available",
        "tavily": (
            "failed" if any("Tavily retrieval" in e or "Tavily returned" in e for e in provider_errors)
            else "available" if tavily_enabled else "disabled"
        ),
    }
    return (unique_evidence, status) if include_status else unique_evidence


# ============================================================
# MAIN RETRIEVAL FUNCTION
# ============================================================

def retrieve_evidence(data, api_key=None):

    # Public input must be a dictionary.
    if not isinstance(data, dict):
        raise ValueError(
            "Invalid input. Expected a dictionary containing 'claims'."
        )

    if "claims" not in data:
        raise ValueError(
            "Input dictionary must contain a 'claims' field."
        )

    claims = data["claims"]

    if not isinstance(claims, list):
        raise ValueError("'claims' must be a list.")

    if not claims:
        return {"claims": []}

    processed_claims = []

    for claim_data in claims:

        if not isinstance(claim_data, dict):
            raise ValueError(
                "Each claim must be represented as a dictionary."
            )

        claim_text = validate_claim(
            claim_data.get("claim")
        )

        evidence, retrieval_status = retrieve_evidence_for_claim(
            claim_text,
            api_key=api_key,
            include_status=True
        )

        processed_claim = {
            **claim_data,
            "claim": claim_text,
            "evidence": evidence,
            "total_evidence": len(evidence),
            "retrieval_status": retrieval_status
        }

        processed_claims.append(processed_claim)

    return {
        "claims": processed_claims
    }


# ============================================================
# END OF MODULE
# ============================================================
