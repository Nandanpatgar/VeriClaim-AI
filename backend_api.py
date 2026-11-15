
"""
VERI-CLAIM-AI
FastAPI Backend

Endpoints:
    GET  /
    GET  /api/health
    POST /api/verify
"""

from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from main import run_pipeline, PipelineError


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

app = FastAPI(
    title="Veri-Claim-AI API",
    description=(
        "Evidence-based, claim-level LLM hallucination "
        "detection and verification system."
    ),
    version="1.0.0"
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# REQUEST MODEL
# ============================================================

class VerificationRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="Text to verify for factual accuracy."
    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "application": "Veri-Claim-AI",
        "status": "running",
        "documentation": "/docs"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health_check():

    return {
        "status": "healthy",
        "application": "Veri-Claim-AI",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat()
    }


# ============================================================
# VERIFICATION ENDPOINT
# ============================================================

@app.post("/api/verify")
def verify_text(request: VerificationRequest):

    try:

        result = run_pipeline(request.text)

        return {
            "status": "success",
            "message": "Verification completed successfully.",
            "data": result
        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except PipelineError as error:

        raise HTTPException(
            status_code=500,
            detail={
                "stage": error.stage,
                "message": error.message
            }
        )

    except Exception:

        raise HTTPException(
            status_code=500,
            detail="An unexpected verification error occurred."
        )
