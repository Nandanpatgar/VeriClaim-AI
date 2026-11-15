
import os
import json
import requests

from dotenv import load_dotenv


# ==================================================
# CONFIGURATION
# ==================================================

load_dotenv()

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

TAVILY_API_URL = "https://api.tavily.com/search"

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

REQUEST_TIMEOUT = 30

WIKIPEDIA_RESULTS = 3

TAVILY_RESULTS = 3


HEADERS = {
    "User-Agent": "Veri-Claim-AI/1.0 (Evidence Retrieval)"
}


# ==================================================
# WIKIPEDIA EVIDENCE RETRIEVAL
# ==================================================

def retrieve_wikipedia_evidence(
    claim,
    max_results=WIKIPEDIA_RESULTS
):
    """
    Search Wikipedia and retrieve relevant article extracts.
    """

    evidence = []

    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": claim,
        "gsrnamespace": 0,
        "gsrlimit": max_results,
        "prop": "extracts|info",
        "exintro": 1,
        "explaintext": 1,
        "exchars": 1200,
        "inprop": "url",
        "format": "json",
        "formatversion": 2
    }

    try:

        response = requests.get(
            WIKIPEDIA_API_URL,
            params=params,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        pages = data.get("query", {}).get("pages", [])

        for page in pages:

            title = page.get("title", "")

            content = page.get("extract", "")

            url = page.get("fullurl", "")

            if not content.strip():
                continue

            evidence.append({

                "source": "Wikipedia",

                "title": title,

                "content": content.strip(),

                "url": url,

                "score": None

            })

    except requests.exceptions.Timeout:

        print("Wikipedia request timed out.")

    except requests.exceptions.RequestException as error:

        print(f"Wikipedia API error: {error}")

    except (ValueError, TypeError) as error:

        print(f"Wikipedia response error: {error}")

    return evidence


# ==================================================
# TAVILY WEB SEARCH
# ==================================================

def retrieve_tavily_evidence(
    claim,
    api_key=None,
    max_results=TAVILY_RESULTS
):
    """
    Search the web using Tavily API.

    If api_key is None, use the key from .env.
    If api_key is an empty string, skip Tavily.
    """

    evidence = []

    # IMPORTANT:
    # Only use the environment key when no key argument
    # was explicitly provided.
    if api_key is None:
        api_key = TAVILY_API_KEY

    # Skip Tavily when the API key is missing.
    if not api_key or not api_key.strip():

        print(
            "WARNING: TAVILY_API_KEY is missing. "
            "Skipping Tavily search."
        )

        return evidence

    headers = {

        "Authorization": f"Bearer {api_key}",

        "Content-Type": "application/json"

    }

    payload = {

        "query": claim,

        "search_depth": "basic",

        "max_results": max_results,

        "include_answer": False,

        "include_raw_content": False,

        "include_images": False

    }

    try:

        response = requests.post(
            TAVILY_API_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        results = data.get("results", [])

        for item in results:

            title = item.get("title", "")

            content = item.get("content", "")

            url = item.get("url", "")

            score = item.get("score")

            if not isinstance(content, str) or not content.strip():
                continue

            evidence.append({

                "source": "Tavily",

                "title": title,

                "content": content.strip(),

                "url": url,

                "score": score

            })

    except requests.exceptions.Timeout:

        print("Tavily request timed out.")

    except requests.exceptions.RequestException as error:

        print(f"Tavily API error: {error}")

    except (ValueError, TypeError) as error:

        print(f"Tavily response error: {error}")

    return evidence


# ==================================================
# DUPLICATE EVIDENCE REMOVAL
# ==================================================

def remove_duplicate_evidence(evidence):
    """
    Remove duplicate evidence based on source URL.
    """

    unique_evidence = []

    seen_urls = set()

    for item in evidence:

        url = item.get("url", "").strip()

        if url and url in seen_urls:
            continue

        if url:
            seen_urls.add(url)

        unique_evidence.append(item)

    return unique_evidence


# ==================================================
# COMBINED EVIDENCE RETRIEVAL
# ==================================================

def retrieve_evidence(claims):
    """
    Retrieve evidence from Wikipedia and Tavily
    for every extracted claim.

    Accepts:
    1. A list of claims.
    2. The complete claim extractor output dictionary.
    """

    if isinstance(claims, dict):

        claims = claims.get("claims", [])

    if not isinstance(claims, list):

        raise ValueError(
            "Input must be a list of claims or a claims dictionary."
        )

    results = []

    print("\nStarting Evidence Retrieval...")

    print("=" * 55)

    for claim_item in claims:

        if not isinstance(claim_item, dict):
            continue

        claim_id = claim_item.get("claim_id")

        claim_text = claim_item.get("claim")

        source_text = claim_item.get("source_text", "")

        if not isinstance(claim_text, str) or not claim_text.strip():
            continue

        print(
            f"\nProcessing Claim {claim_id}: {claim_text}"
        )

        # ------------------------------------------
        # WIKIPEDIA SEARCH
        # ------------------------------------------

        print("Searching Wikipedia...")

        wikipedia_results = retrieve_wikipedia_evidence(
            claim_text
        )

        # ------------------------------------------
        # TAVILY SEARCH
        # ------------------------------------------

        print("Searching Tavily...")

        tavily_results = retrieve_tavily_evidence(
            claim_text
        )

        # ------------------------------------------
        # COMBINE EVIDENCE
        # ------------------------------------------

        combined_evidence = (
            wikipedia_results + tavily_results
        )

        # ------------------------------------------
        # REMOVE DUPLICATES
        # ------------------------------------------

        combined_evidence = remove_duplicate_evidence(
            combined_evidence
        )

        # ------------------------------------------
        # ASSIGN EVIDENCE IDs
        # ------------------------------------------

        for index, item in enumerate(
            combined_evidence,
            start=1
        ):

            item["evidence_id"] = index

        # ------------------------------------------
        # STORE CLAIM RESULTS
        # ------------------------------------------

        results.append({

            "claim_id": claim_id,

            "claim": claim_text,

            "source_text": source_text,

            "evidence": combined_evidence,

            "total_evidence": len(combined_evidence)

        })

        print(
            f"Evidence collected: {len(combined_evidence)}"
        )

    print("\nEvidence Retrieval Completed.")

    return {
        "claims": results
    }


# ==================================================
# TERMINAL INTERFACE
# ==================================================

def main():

    print("\n" + "=" * 55)

    print("              VERI-CLAIM-AI")

    print("             EVIDENCE RETRIEVAL")

    print("=" * 55)

    print("\nEnter a factual claim for testing.")

    claim = input("\nEnter Claim: ").strip()

    if not claim:

        print("\nNo claim provided.")

        return

    test_claims = {

        "claims": [

            {
                "claim_id": 1,

                "claim": claim,

                "source_text": claim

            }

        ]

    }

    result = retrieve_evidence(test_claims)

    print("\n" + "=" * 55)

    print("             RETRIEVAL RESULTS")

    print("=" * 55)

    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False
        )
    )


# ==================================================
# APPLICATION ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()
