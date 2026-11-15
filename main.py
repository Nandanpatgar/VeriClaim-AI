"""
VERI-CLAIM-AI
End-to-End Hallucination Detection System

Pipeline:
1. Claim Extraction - Llama 3.2
2. Evidence Retrieval - Wikipedia + Tavily
3. Evidence Ranking - MiniLM
4. NLI Verification - DeBERTa
5. Decision Engine
6. Verification Report Generation
"""

import json
import traceback

from claim_extractor import extract_claims
from evidence_retrieval import retrieve_evidence
from evidence_ranker import rank_evidence
from nli_verifier import verify_claims
from decision_engine import decide_claims


# ==================================================
# CONFIGURATION
# ==================================================

REPORT_FILE = "verification_report.json"


# ==================================================
# INPUT VALIDATION
# ==================================================

def validate_input(text):

    if not isinstance(text, str):
        raise ValueError("Input must be a string.")

    text = text.strip()

    if not text:
        raise ValueError("Input cannot be empty.")

    if len(text) < 10:
        raise ValueError(
            "Please enter a longer factual statement."
        )

    return text


# ==================================================
# NORMALIZE CLAIM EXTRACTION OUTPUT
# ==================================================

def normalize_claims(result):

    if isinstance(result, dict):

        claims = result.get("claims", [])

    elif isinstance(result, list):

        claims = result

    else:

        raise ValueError(
            "Unexpected claim extraction output."
        )

    if not isinstance(claims, list) or not claims:

        raise ValueError(
            "No factual claims could be extracted."
        )

    return claims


# ==================================================
# MAIN VERIFICATION PIPELINE
# ==================================================

def run_pipeline(text):

    text = validate_input(text)

    print("\n" + "=" * 65)
    print("                 VERI-CLAIM-AI")
    print("          HALLUCINATION DETECTION SYSTEM")
    print("=" * 65)

    # ----------------------------------------------
    # STEP 1: CLAIM EXTRACTION
    # ----------------------------------------------

    print("\n[1/5] Extracting factual claims...")

    extracted_result = extract_claims(text)

    claims = normalize_claims(extracted_result)

    print(f"Claims extracted: {len(claims)}")

    # ----------------------------------------------
    # STEP 2: EVIDENCE RETRIEVAL
    # ----------------------------------------------

    print("\n[2/5] Retrieving supporting evidence...")

    retrieval_result = retrieve_evidence({

        "claims": claims

    })

    if not isinstance(retrieval_result, dict):

        raise ValueError(
            "Evidence retrieval returned an invalid result."
        )

    retrieved_claims = retrieval_result.get(
        "claims",
        []
    )

    if not isinstance(retrieved_claims, list):

        raise ValueError(
            "Retrieved claims must be a list."
        )

    total_evidence = sum(

        len(item.get("evidence", []))

        for item in retrieved_claims

        if isinstance(item, dict)

    )

    print(f"Evidence collected: {total_evidence}")

    # ----------------------------------------------
    # STEP 3: EVIDENCE RANKING
    # ----------------------------------------------

    print("\n[3/5] Ranking evidence using MiniLM...")

    ranked_result = rank_evidence(
        retrieval_result
    )

    if not isinstance(ranked_result, dict):

        raise ValueError(
            "Evidence ranking returned an invalid result."
        )

    print("Evidence ranking completed.")

    # ----------------------------------------------
    # STEP 4: NLI VERIFICATION
    # ----------------------------------------------

    print("\n[4/5] Verifying claims using DeBERTa...")

    verification_result = verify_claims(
        ranked_result
    )

    if not isinstance(verification_result, dict):

        raise ValueError(
            "NLI verification returned an invalid result."
        )

    print("NLI verification completed.")

    # ----------------------------------------------
    # STEP 5: DECISION ENGINE
    # ----------------------------------------------

    print("\n[5/5] Running Decision Engine...")

    decision_result = decide_claims(
        verification_result
    )

    if not isinstance(decision_result, dict):

        raise ValueError(
            "Decision Engine returned an invalid result."
        )

    print("Claim-level decisions completed.")

    # ----------------------------------------------
    # FINAL RESULT
    # ----------------------------------------------

    final_result = {

        "system": "Veri-Claim-AI",

        "original_input": text,

        "extracted_claims": claims,

        "total_claims": len(claims),

        "total_evidence_retrieved": total_evidence,

        "verification_results": verification_result,

        "final_decisions": decision_result

    }

    return final_result


# ==================================================
# DISPLAY FINAL RESULTS
# ==================================================

def display_results(result):

    print("\n" + "=" * 65)
    print("                 FINAL RESULTS")
    print("=" * 65)

    final_decisions = result.get(
        "final_decisions",
        {}
    )

    summary = final_decisions.get(
        "summary",
        {}
    )

    print("\nOVERALL SUMMARY")
    print("-" * 45)

    print(
        f"Total Claims: {summary.get('total_claims', 0)}"
    )

    print(
        f"Supported: {summary.get('supported', 0)}"
    )

    print(
        f"Contradicted: {summary.get('contradicted', 0)}"
    )

    print(
        f"Unverifiable: {summary.get('unverifiable', 0)}"
    )

    decisions = final_decisions.get(
        "decisions",
        []
    )

    print("\nINDIVIDUAL CLAIM RESULTS")
    print("-" * 45)

    if not decisions:

        print("No claim decisions available.")

        return

    for item in decisions:

        print("\n" + "-" * 45)

        print(
            f"Claim ID: {item.get('claim_id')}"
        )

        print(
            f"Claim: {item.get('claim')}"
        )

        print(
            f"Decision: {item.get('decision', 'UNVERIFIABLE')}"
        )

        # Updated field name with backward compatibility
        confidence = item.get(
            "evidence_confidence",
            item.get("decision_confidence", 0.0)
        )

        print(
            f"Evidence Confidence: {confidence}"
        )

        print(
            f"Reason: {item.get('decision_reason', 'No explanation available.')}"
        )

        evidence = item.get(
            "evidence_summary",
            {}
        )

        print("\nEvidence Summary:")

        print(
            "Supporting evidence:",
            evidence.get("supporting_evidence_count", 0)
        )

        print(
            "Contradicting evidence:",
            evidence.get("contradicting_evidence_count", 0)
        )

        print(
            "Neutral evidence:",
            evidence.get("neutral_evidence_count", 0)
        )

        print(
            "Independent supporting sources:",
            evidence.get("independent_supporting_sources", 0)
        )

        print(
            "Independent contradicting sources:",
            evidence.get("independent_contradicting_sources", 0)
        )

    print("\n" + "=" * 65)


# ==================================================
# SAVE VERIFICATION REPORT
# ==================================================

def save_report(result):

    try:

        with open(
            REPORT_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(

                result,

                file,

                indent=4,

                ensure_ascii=False

            )

        print(
            f"\nComplete verification report saved to {REPORT_FILE}"
        )

        return True

    except (OSError, TypeError, ValueError) as error:

        print(
            f"\nReport generation failed: {error}"
        )

        return False


# ==================================================
# APPLICATION ENTRY POINT
# ==================================================

def main():

    print("\n" + "=" * 65)

    print("                 VERI-CLAIM-AI")

    print("          EVIDENCE-BASED VERIFICATION")

    print("=" * 65)

    print("\nEnter your text for verification.")

    print("Type END on a new line when finished.\n")

    lines = []

    while True:

        try:

            line = input()

        except EOFError:

            break

        if line.strip().upper() == "END":

            break

        lines.append(line)

    user_input = "\n".join(lines)

    try:

        # Execute complete pipeline
        result = run_pipeline(user_input)

        # Save report before displaying
        save_report(result)

        # Display final results
        display_results(result)

        print("\nVERIFICATION COMPLETED SUCCESSFULLY.")

    except Exception as error:

        print("\n" + "=" * 65)

        print("          PIPELINE EXECUTION FAILED")

        print("=" * 65)

        print(f"\nError: {error}")

        print("\nDetailed traceback:")

        traceback.print_exc()


# ==================================================
# RUN APPLICATION
# ==================================================

if __name__ == "__main__":

    main()