
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

Designed for:
- Direct Python execution
- Backend integration
- Future FastAPI / React frontend integration
"""

import json
import os
import tempfile
import traceback
from datetime import datetime, timezone

from claim_extractor import extract_claims
from evidence_retrieval import retrieve_evidence
from evidence_ranker import rank_evidence
from nli_verifier import verify_claims
from decision_engine import decide_claims


# ============================================================
# CONFIGURATION
# ============================================================

SYSTEM_NAME = "Veri-Claim-AI"

REPORT_FILE = "verification_report.json"

MIN_INPUT_LENGTH = 10


# ============================================================
# CUSTOM EXCEPTIONS
# ============================================================

class PipelineError(Exception):
    """Raised when a verification pipeline stage fails."""

    def __init__(self, stage, message):
        self.stage = stage
        self.message = message

        super().__init__(
            f"{stage}: {message}"
        )


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_input(text):
    """
    Validate user-provided text.
    """

    if not isinstance(text, str):
        raise ValueError(
            "Input must be a string."
        )

    text = text.strip()

    if not text:
        raise ValueError(
            "Input cannot be empty."
        )

    if len(text) < MIN_INPUT_LENGTH:
        raise ValueError(
            "Please enter a longer factual statement."
        )

    return text


# ============================================================
# CLAIM NORMALIZATION
# ============================================================

def normalize_claims(result):
    """
    Normalize claim extraction output.

    Supports:
    - List of claim dictionaries
    - Dictionary containing a claims list
    - List of claim strings
    """

    if isinstance(result, dict):

        claims = result.get("claims", [])

    elif isinstance(result, list):

        claims = result

    else:

        raise PipelineError(
            "Claim Extraction",
            "Unexpected claim extraction output."
        )

    if not isinstance(claims, list) or not claims:

        raise PipelineError(
            "Claim Extraction",
            "No factual claims could be extracted."
        )

    normalized = []

    for index, item in enumerate(claims, start=1):

        if isinstance(item, str):

            claim_text = item.strip()

            claim = {
                "claim_id": index,
                "claim": claim_text
            }

        elif isinstance(item, dict):

            claim_text = item.get("claim", "")

            if not isinstance(claim_text, str):
                continue

            claim_text = claim_text.strip()

            claim = item.copy()

            claim["claim"] = claim_text

            if not claim.get("claim_id"):
                claim["claim_id"] = index

        else:
            continue

        if claim_text:
            normalized.append(claim)

    if not normalized:

        raise PipelineError(
            "Claim Extraction",
            "No valid factual claims were extracted."
        )

    return normalized


# ============================================================
# PIPELINE OUTPUT VALIDATION
# ============================================================

def validate_pipeline_output(result, stage):
    """
    Ensure each pipeline stage returns a dictionary.
    """

    if not isinstance(result, dict):

        raise PipelineError(
            stage,
            "Module returned an invalid output. "
            "Expected a dictionary."
        )

    return result


# ============================================================
# EVIDENCE STATISTICS
# ============================================================

def calculate_evidence_statistics(retrieval_result):
    """
    Calculate evidence retrieval statistics.
    """

    claims = retrieval_result.get(
        "claims",
        []
    )

    total_evidence = 0

    claim_statistics = []

    for item in claims:

        if not isinstance(item, dict):
            continue

        evidence = item.get(
            "evidence",
            []
        )

        if not isinstance(evidence, list):
            evidence = []

        count = len(evidence)

        total_evidence += count

        claim_statistics.append({

            "claim_id": item.get("claim_id"),

            "claim": item.get("claim"),

            "evidence_count": count

        })

    return {

        "total_evidence": total_evidence,

        "claims_with_evidence": sum(
            item["evidence_count"] > 0
            for item in claim_statistics
        ),

        "claims_without_evidence": sum(
            item["evidence_count"] == 0
            for item in claim_statistics
        ),

        "claim_statistics": claim_statistics

    }


# ============================================================
# MAIN VERIFICATION PIPELINE
# ============================================================

def run_pipeline(text):
    """
    Execute the complete hallucination detection pipeline.

    This function is independent of terminal input/output.

    It can be called directly from a future FastAPI endpoint.
    """

    text = validate_input(text)

    pipeline_started = datetime.now(timezone.utc)

    # --------------------------------------------------------
    # STEP 1: CLAIM EXTRACTION
    # --------------------------------------------------------

    try:

        extracted_result = extract_claims(text)

        claims = normalize_claims(
            extracted_result
        )

    except PipelineError:
        raise

    except Exception as error:

        raise PipelineError(
            "Claim Extraction",
            str(error)
        ) from error

    # --------------------------------------------------------
    # STEP 2: EVIDENCE RETRIEVAL
    # --------------------------------------------------------

    try:

        retrieval_result = retrieve_evidence({

            "claims": claims

        })

        retrieval_result = validate_pipeline_output(
            retrieval_result,
            "Evidence Retrieval"
        )

        retrieved_claims = retrieval_result.get(
            "claims",
            []
        )

        if not isinstance(retrieved_claims, list):

            raise PipelineError(
                "Evidence Retrieval",
                "Retrieved claims must be a list."
            )

        evidence_statistics = calculate_evidence_statistics(
            retrieval_result
        )

    except PipelineError:
        raise

    except Exception as error:

        raise PipelineError(
            "Evidence Retrieval",
            str(error)
        ) from error

    # --------------------------------------------------------
    # STEP 3: EVIDENCE RANKING
    # --------------------------------------------------------

    try:

        ranked_result = rank_evidence(
            retrieval_result
        )

        ranked_result = validate_pipeline_output(
            ranked_result,
            "Evidence Ranking"
        )

    except PipelineError:
        raise

    except Exception as error:

        raise PipelineError(
            "Evidence Ranking",
            str(error)
        ) from error

    # --------------------------------------------------------
    # STEP 4: NLI VERIFICATION
    # --------------------------------------------------------

    try:

        verification_result = verify_claims(
            ranked_result
        )

        verification_result = validate_pipeline_output(
            verification_result,
            "NLI Verification"
        )

    except PipelineError:
        raise

    except Exception as error:

        raise PipelineError(
            "NLI Verification",
            str(error)
        ) from error

    # --------------------------------------------------------
    # STEP 5: DECISION ENGINE
    # --------------------------------------------------------

    try:

        decision_result = decide_claims(
            verification_result
        )

        decision_result = validate_pipeline_output(
            decision_result,
            "Decision Engine"
        )

    except PipelineError:
        raise

    except Exception as error:

        raise PipelineError(
            "Decision Engine",
            str(error)
        ) from error

    # --------------------------------------------------------
    # EXECUTION METADATA
    # --------------------------------------------------------

    pipeline_completed = datetime.now(timezone.utc)

    duration = (
        pipeline_completed - pipeline_started
    ).total_seconds()

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    retrieval_states = [
        item.get("retrieval_status", {}).get("state", "unknown")
        for item in retrieval_result.get("claims", [])
        if isinstance(item, dict)
        and isinstance(item.get("retrieval_status", {}), dict)
    ]

    retrieval_state = (
        "partial" if "partial" in retrieval_states
        else "complete" if retrieval_states and all(
            state == "complete" for state in retrieval_states
        )
        else "unknown"
    )

    final_result = {

        "system": SYSTEM_NAME,

        "status": "success",

        "original_input": text,

        "metadata": {

            "started_at": pipeline_started.isoformat(),

            "completed_at": pipeline_completed.isoformat(),

            "execution_time_seconds": round(
                duration,
                3
            ),

            "total_claims": len(claims),

            "evidence_retrieval_status": retrieval_state,

            "pipeline_stages_completed": 5

        },

        "extracted_claims": claims,

        "total_claims": len(claims),

        "evidence_statistics": evidence_statistics,

        "total_evidence_retrieved": (
            evidence_statistics["total_evidence"]
        ),

        "retrieval_results": retrieval_result,

        "ranking_results": ranked_result,

        "verification_results": verification_result,

        "final_decisions": decision_result

    }

    return final_result


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(result):
    """
    Display verification results in the terminal.
    """

    print("\n" + "=" * 65)

    print("                 VERI-CLAIM-AI")

    print("                 FINAL RESULTS")

    print("=" * 65)

    decisions = result.get(
        "final_decisions",
        {}
    )

    summary = decisions.get(
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

    print("\nEVIDENCE STATISTICS")

    print("-" * 45)

    evidence_stats = result.get(
        "evidence_statistics",
        {}
    )

    print(
        f"Evidence Retrieved: "
        f"{evidence_stats.get('total_evidence', 0)}"
    )

    print(
        f"Claims with Evidence: "
        f"{evidence_stats.get('claims_with_evidence', 0)}"
    )

    print(
        f"Claims without Evidence: "
        f"{evidence_stats.get('claims_without_evidence', 0)}"
    )

    print("\nINDIVIDUAL CLAIM RESULTS")

    print("-" * 45)

    claim_decisions = decisions.get(
        "decisions",
        []
    )

    if not claim_decisions:

        print("No claim decisions available.")

        return

    for item in claim_decisions:

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

        confidence = item.get(
            "evidence_confidence",
            0.0
        )

        print(
            f"Evidence Confidence: {confidence}"
        )

        print(
            f"Reason: {item.get('decision_reason', '')}"
        )

        evidence = item.get(
            "evidence_summary",
            {}
        )

        print("\nEvidence Summary:")

        print(
            "Supporting:",
            evidence.get("supporting_evidence_count", 0)
        )

        print(
            "Contradicting:",
            evidence.get("contradicting_evidence_count", 0)
        )

        print(
            "Neutral:",
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


# ============================================================
# SAVE VERIFICATION REPORT
# ============================================================

def save_report(result, report_file=REPORT_FILE):
    """
    Save the complete verification result as JSON.

    Uses a temporary file followed by an atomic replace
    to reduce the risk of incomplete reports.
    """

    if not isinstance(result, dict):

        raise ValueError(
            "Report data must be a dictionary."
        )

    directory = os.path.dirname(
        os.path.abspath(report_file)
    )

    temporary_path = None

    try:

        with tempfile.NamedTemporaryFile(

            mode="w",

            encoding="utf-8",

            suffix=".tmp",

            prefix="vericlaim_",

            dir=directory,

            delete=False

        ) as file:

            temporary_path = file.name

            json.dump(

                result,

                file,

                indent=4,

                ensure_ascii=False,

                allow_nan=False

            )

        os.replace(
            temporary_path,
            report_file
        )

        print(
            f"\nVerification report saved: {report_file}"
        )

        return True

    except (OSError, TypeError, ValueError) as error:

        print(
            f"\nReport generation failed: {error}"
        )

        if temporary_path and os.path.exists(
            temporary_path
        ):

            try:
                os.remove(temporary_path)

            except OSError:
                pass

        return False


# ============================================================
# TERMINAL APPLICATION
# ============================================================

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

        print("\nStarting verification pipeline...\n")

        result = run_pipeline(
            user_input
        )

        print("Verification pipeline completed.")

        save_report(result)

        display_results(result)

        print(
            "\nVERIFICATION COMPLETED SUCCESSFULLY."
        )

    except (ValueError, PipelineError) as error:

        print("\nPIPELINE EXECUTION FAILED")

        print(f"\nError: {error}")

    except Exception as error:

        print("\nUNEXPECTED APPLICATION ERROR")

        print(f"\nError: {error}")

        traceback.print_exc()


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
