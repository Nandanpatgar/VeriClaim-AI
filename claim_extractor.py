
import json
import re
import requests


# ==================================================
# CONFIGURATION
# ==================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"
REQUEST_TIMEOUT = 300


# ==================================================
# TEXT NORMALIZATION
# ==================================================

def normalize_text(text):
    """Normalize text for comparison."""

    text = text.casefold()
    text = re.sub(r"\s*([.,!?;:])\s*", r"\1 ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==================================================
# PROMPT GENERATION
# ==================================================

def build_prompt(response):

    return f"""
You are an expert factual claim extraction system.

Extract every objectively verifiable factual assertion
from the given text.

RULES:

1. FACTUAL CLAIM EXTRACTION

- Extract all objectively verifiable factual assertions.
- Include potentially true, false and unverifiable claims.
- Never fact-check or correct the original statements.
- Preserve negations and original meanings.
- Exclude subjective opinions and personal preferences.

2. ATOMIC CLAIM EXTRACTION

Each claim must express only ONE factual assertion.

Example:

Input:
Virat Kohli is Indian and plays cricket.

Output:
1. Virat Kohli is Indian.
2. Virat Kohli plays cricket.

Example:

Input:
Python was created by Guido van Rossum and released in 1991.

Output:
1. Python was created by Guido van Rossum.
2. Python was first released in 1991.

3. PRONOUN RESOLUTION

Resolve pronouns whenever their references are clear.

Example:

Input:
Albert Einstein was born in Germany in 1879.
He developed the theory of relativity.

Output:
1. Albert Einstein was born in Germany in 1879.
2. Albert Einstein developed the theory of relativity.

Do not leave identifiable pronouns unresolved.
Never guess ambiguous references.

4. OPINION FILTERING

Exclude statements such as:
- Python is the best programming language.
- Cricket is an amazing sport.
- Artificial intelligence is fascinating.

5. SOURCE TRACEABILITY

Every claim must contain source_text.

- Copy source_text from the original input.
- Do not invent source sentences.
- Preserve original wording.
- Use the sentence containing the claim.
- Multiple claims may share the same source sentence.

6. OUTPUT FORMAT

Return ONLY valid JSON.

{{
    "claims": [
        {{
            "claim": "An individual factual assertion.",
            "source_text": "Original sentence from input."
        }}
    ]
}}

If no factual claims exist:

{{
    "claims": []
}}

INPUT TEXT:

{response}
"""


# ==================================================
# JSON PARSER
# ==================================================

def parse_model_response(model_response):

    if isinstance(model_response, dict):
        data = model_response

    else:
        cleaned = model_response.strip()

        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE
        )

        cleaned = re.sub(r"\s*```$", "", cleaned)

        data = json.loads(cleaned)

    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object.")

    if "claims" not in data:
        raise ValueError("Missing claims field.")

    if not isinstance(data["claims"], list):
        raise ValueError("Claims must be a JSON array.")

    return data["claims"]


# ==================================================
# SOURCE MATCHING
# ==================================================

def source_exists(source_text, original_response):

    normalized_source = normalize_text(source_text)
    normalized_original = normalize_text(original_response)

    if normalized_source in normalized_original:
        return True

    source_words = normalized_source.split()
    original_words = normalized_original.split()

    if not source_words:
        return False

    for i in range(
        len(original_words) - len(source_words) + 1
    ):

        if original_words[i:i + len(source_words)] == source_words:
            return True

    return False


# ==================================================
# CLAIM VALIDATION
# ==================================================

def validate_claims(raw_claims, original_response):

    validated_claims = []
    seen_claims = set()

    for item in raw_claims:

        if not isinstance(item, dict):
            continue

        claim = item.get("claim")
        source_text = item.get("source_text")

        if not isinstance(claim, str):
            continue

        if not isinstance(source_text, str):
            continue

        claim = re.sub(r"\s+", " ", claim).strip()
        source_text = re.sub(r"\s+", " ", source_text).strip()

        if not claim or not source_text:
            continue

        if not source_exists(source_text, original_response):

            print(
                "WARNING: Claim skipped because its "
                "source text could not be matched."
            )

            continue

        normalized_claim = normalize_text(claim)

        if normalized_claim in seen_claims:
            continue

        seen_claims.add(normalized_claim)

        validated_claims.append({
            "claim_id": len(validated_claims) + 1,
            "claim": claim,
            "source_text": source_text
        })

    return validated_claims


# ==================================================
# SENTENCE SPLITTING
# ==================================================

def split_sentences(text):
    """Split text into individual sentences."""

    return [
        sentence.strip()
        for sentence in re.split(r'(?<=[.!?])\s+', text.strip())
        if sentence.strip()
    ]


# ==================================================
# ENTITY DETECTION
# ==================================================

def extract_named_entity(sentence):
    """
    Extract a simple multi-word proper name from a sentence.

    This is a lightweight heuristic, not a full
    named-entity recognition model.
    """

    pattern = (
        r"\b([A-Z][a-z]+"
        r"(?:\s+[A-Z][a-z]+)+)\b"
    )

    matches = re.findall(pattern, sentence)

    # Avoid treating common sentence openings as names.
    ignored = {
        "The Earth",
        "The Moon",
        "The Sun",
        "New Delhi",
        "United States"
    }

    for match in matches:

        if match not in ignored:
            return match

    return None


# ==================================================
# PRONOUN RESOLUTION
# ==================================================

def resolve_pronouns(claims, original_response):
    """
    Resolve clear subject pronouns using the most
    recent identifiable entity in the preceding text.

    This intentionally handles simple cases only.
    """

    sentences = split_sentences(original_response)

    for item in claims:

        claim = item["claim"]
        source_text = item["source_text"]

        normalized_source = normalize_text(source_text)

        # Find the sentence corresponding to source_text.
        source_index = None

        for index, sentence in enumerate(sentences):

            normalized_sentence = normalize_text(sentence)

            if (
                normalized_sentence == normalized_source
                or normalized_sentence in normalized_source
                or normalized_source in normalized_sentence
            ):
                source_index = index
                break

        if source_index is None or source_index == 0:
            continue

        # Search previous sentences for a named entity.
        entity = None

        for previous_sentence in reversed(
            sentences[:source_index]
        ):

            candidate = extract_named_entity(
                previous_sentence
            )

            if candidate:
                entity = candidate
                break

        if not entity:
            continue

        # Resolve clear subject pronouns only.
        pattern = r"^(He|She|They)\b"

        if re.match(pattern, claim, flags=re.IGNORECASE):

            claim = re.sub(
                pattern,
                entity,
                claim,
                count=1,
                flags=re.IGNORECASE
            )

            item["claim"] = claim

    return claims


# ==================================================
# MAIN EXTRACTION FUNCTION
# ==================================================

def extract_claims(response):

    if not isinstance(response, str):

        print("ERROR: Input must be a string.")

        return None

    response = response.strip()

    if not response:
        return {"claims": []}

    payload = {

        "model": MODEL_NAME,

        "prompt": build_prompt(response),

        "stream": False,

        "format": "json",

        "options": {

            "temperature": 0,

            "seed": 42,

            "num_ctx": 8192,

            "num_predict": 4096

        }
    }

    try:

        print("\nAnalyzing response using Llama 3.2...")

        model_response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        model_response.raise_for_status()

        response_data = model_response.json()

        generated_text = response_data.get("response", "")

        if not generated_text:
            raise ValueError("Ollama returned an empty response.")

        # Parse Llama output.
        raw_claims = parse_model_response(generated_text)

        # Validate source text and remove duplicates.
        validated_claims = validate_claims(
            raw_claims,
            response
        )

        # Resolve clear pronoun references.
        validated_claims = resolve_pronouns(
            validated_claims,
            response
        )

        # Reassign sequential IDs.
        for index, item in enumerate(validated_claims, start=1):
            item["claim_id"] = index

        return {
            "claims": validated_claims
        }

    except requests.exceptions.ConnectionError:

        print("\nERROR: Cannot connect to Ollama.")
        print("Make sure Ollama is running.")

        return None

    except requests.exceptions.Timeout:

        print("\nERROR: Ollama request timed out.")

        return None

    except requests.exceptions.RequestException as error:

        print(f"\nOLLAMA API ERROR: {error}")

        return None

    except (ValueError, KeyError, TypeError) as error:

        print(f"\nCLAIM EXTRACTION ERROR: {error}")

        return None

    except Exception as error:

        print(f"\nUNEXPECTED ERROR: {error}")

        return None


# ==================================================
# TERMINAL APPLICATION
# ==================================================

def main():

    print("\n" + "=" * 50)

    print("             VERI-CLAIM-AI")

    print("          CLAIM EXTRACTION MODULE")

    print("=" * 50)

    print("\nEnter the LLM-generated response.")

    response = input("\nEnter LLM Response: ").strip()

    if not response:

        print("\nNo input provided.")

        return

    result = extract_claims(response)

    if result is None:

        print("\nClaim extraction failed.")

        return

    print("\n" + "=" * 50)

    print("             EXTRACTED CLAIMS")

    print("=" * 50)

    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False
        )
    )

    print("\n" + "-" * 50)

    print("Total Claims:", len(result["claims"]))

    print("-" * 50)

    print("\nClaim extraction completed successfully.")


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()
