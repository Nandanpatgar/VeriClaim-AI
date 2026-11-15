
import pytest

from claim_extractor import (
    extract_claims,
    normalize_text,
    parse_model_response,
    validate_claims
)


# ==================================================
# 1. EMPTY INPUT TESTS
# ==================================================

def test_empty_input():
    result = extract_claims("")
    assert result == {"claims": []}


def test_whitespace_input():
    result = extract_claims("   ")
    assert result == {"claims": []}


def test_invalid_input():
    result = extract_claims(None)
    assert result is None


# ==================================================
# 2. JSON PARSING TESTS
# ==================================================

def test_valid_json_response():

    response = {
        "claims": [
            {
                "claim": "India gained independence in 1947.",
                "source_text": "India gained independence in 1947."
            }
        ]
    }

    result = parse_model_response(response)

    assert len(result) == 1
    assert result[0]["claim"] == "India gained independence in 1947."


def test_invalid_json_response():

    with pytest.raises(ValueError):
        parse_model_response("Invalid JSON")


def test_missing_claims_field():

    with pytest.raises(ValueError):
        parse_model_response({"data": []})


def test_claims_must_be_list():

    with pytest.raises(ValueError):
        parse_model_response({"claims": "invalid"})


# ==================================================
# 3. TEXT NORMALIZATION TESTS
# ==================================================

def test_text_normalization():

    result = normalize_text("  HELLO   WORLD  ")

    assert result == "hello world"


def test_normalization_case_insensitive():

    assert normalize_text("INDIA") == normalize_text("India")


# ==================================================
# 4. CLAIM VALIDATION TESTS
# ==================================================

def test_valid_claim():

    original = "India gained independence in 1947."

    raw_claims = [
        {
            "claim": "India gained independence in 1947.",
            "source_text": "India gained independence in 1947."
        }
    ]

    result = validate_claims(raw_claims, original)

    assert len(result) == 1
    assert result[0]["claim_id"] == 1


def test_invalid_source_text_rejected():

    original = "India gained independence in 1947."

    raw_claims = [
        {
            "claim": "India became independent in 1947.",
            "source_text": "India became independent in 1950."
        }
    ]

    result = validate_claims(raw_claims, original)

    assert len(result) == 0


def test_duplicate_claim_removal():

    original = "India gained independence in 1947."

    raw_claims = [
        {
            "claim": "India gained independence in 1947.",
            "source_text": original
        },
        {
            "claim": "India gained independence in 1947.",
            "source_text": original
        }
    ]

    result = validate_claims(raw_claims, original)

    assert len(result) == 1


def test_sequential_claim_ids():

    original = (
        "India gained independence in 1947. "
        "The Taj Mahal is located in Agra."
    )

    raw_claims = [
        {
            "claim": "India gained independence in 1947.",
            "source_text": "India gained independence in 1947."
        },
        {
            "claim": "The Taj Mahal is located in Agra.",
            "source_text": "The Taj Mahal is located in Agra."
        }
    ]

    result = validate_claims(raw_claims, original)

    ids = [item["claim_id"] for item in result]

    assert ids == [1, 2]


# ==================================================
# 5. FALSE CLAIM PRESERVATION
# ==================================================

def test_false_claim_is_not_removed():

    original = "The Earth is flat."

    raw_claims = [
        {
            "claim": "The Earth is flat.",
            "source_text": original
        }
    ]

    result = validate_claims(raw_claims, original)

    assert len(result) == 1
    assert result[0]["claim"] == "The Earth is flat."


# ==================================================
# 6. SOURCE TRACEABILITY
# ==================================================

def test_source_traceability():

    original = (
        "Python was created by Guido van Rossum. "
        "Java was released in 1995."
    )

    raw_claims = [
        {
            "claim": "Python was created by Guido van Rossum.",
            "source_text": "Python was created by Guido van Rossum."
        },
        {
            "claim": "Java was released in 1995.",
            "source_text": "Java was released in 1995."
        }
    ]

    result = validate_claims(raw_claims, original)

    assert len(result) == 2

    for item in result:
        assert normalize_text(item["source_text"]) in normalize_text(original)


# ==================================================
# 7. INVALID MODEL OUTPUT
# ==================================================

def test_invalid_claim_item_is_ignored():

    original = "India gained independence in 1947."

    raw_claims = [
        "Invalid item",
        {
            "claim": "India gained independence in 1947.",
            "source_text": original
        }
    ]

    result = validate_claims(raw_claims, original)

    assert len(result) == 1


# ==================================================
# TEST EXECUTION
# ==================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
