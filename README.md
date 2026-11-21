# Veri-Claim-AI

### Claim-Level Hallucination Detection in Large Language Models Using Evidence-Based Verification

**Every Claim Verified by Evidence.**

Veri-Claim-AI is an evidence-based hallucination detection system designed to identify, analyze and evaluate factual inaccuracies in Large Language Model (LLM) generated responses.

Unlike traditional approaches that evaluate an entire response as a single unit, Veri-Claim-AI performs claim-level verification by decomposing generated text into individual factual claims, retrieving relevant evidence, evaluating semantic relevance, applying Natural Language Inference (NLI), and generating explainable verification results.

The system aims to improve the reliability, transparency and factual consistency of AI-generated information through evidence-grounded verification.

---

## 1. Problem Statement

Large Language Models can generate responses that appear convincing and contextually appropriate but contain inaccurate, misleading or fabricated information. These inaccuracies are commonly referred to as hallucinations.

Traditional fact-checking approaches often evaluate complete responses, making it difficult to identify exactly which statements are incorrect and why.

Veri-Claim-AI addresses this limitation through a claim-level verification approach that independently evaluates factual statements using retrieved evidence and Natural Language Inference.

The system aims to answer three fundamental questions:

- Is the generated claim supported by available evidence?
- Does the retrieved evidence contradict the claim?
- Is there insufficient evidence to determine the claim's factual validity?

---

## 2. Project Objectives

- Automatically extract individual factual claims from LLM-generated responses.
- Retrieve relevant evidence from external knowledge sources.
- Rank retrieved passages according to their relevance to individual claims.
- Evaluate claim-evidence relationships using Natural Language Inference.
- Assess the reliability of retrieved information sources.
- Classify claims into Supported, Contradicted and Unverifiable categories.
- Calculate claim-level confidence and verification metrics.
- Generate explainable verification reports containing evidence and decision information.
- Provide an interactive web interface for submitting text and examining verification results.

---

## 3. System Architecture

Veri-Claim-AI follows a modular, multi-stage verification architecture.

```text
                 USER INPUT
                      |
                      v
              CLAIM EXTRACTION
                      |
                      v
              EVIDENCE RETRIEVAL
                      |
                      v
               EVIDENCE RANKING
                      |
                      v
               NLI VERIFICATION
                      |
                      v
             SOURCE RELIABILITY
                      |
                      v
               DECISION ENGINE
                      |
                      v
             CLAIM CLASSIFICATION
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
      SUPPORTED  CONTRADICTED  UNVERIFIABLE
          |           |           |
          +-----------+-----------+
                      |
                      v
              REPORT GENERATION
                      |
                      v
              VERIFICATION RESULTS
                      |
                      v
                REACT FRONTEND
```

### Pipeline Components

#### 1. Claim Extraction

The input response is processed to identify individual factual claims.

The system uses a locally hosted Large Language Model through Ollama to extract structured claims from the submitted text.

**Technology:** Llama 3.2, Ollama.

#### 2. Evidence Retrieval

Relevant evidence is retrieved for each extracted claim using external knowledge sources.

The retrieval component supports knowledge retrieval through Wikipedia and Tavily.

The retrieved passages are associated with their source information and claim context.

**Technologies:** Wikipedia API, Tavily API.

#### 3. Evidence Ranking

Retrieved passages are evaluated to identify evidence that is semantically relevant to the corresponding claim.

Sentence embeddings are generated using a transformer-based embedding model. The ranker blends cosine similarity (80%) with token-overlap F1 (20%) so paraphrases remain visible while exact names and numbers influence passage ordering. Both component scores are included in the result; the combined relevance score is not a probability of truth.
Sentence embeddings are generated using a transformer-based embedding model, and cosine similarity is used to estimate semantic relevance.

**Technologies:** Sentence Transformers, all-MiniLM-L6-v2, Cosine Similarity.

#### 4. Natural Language Inference

The NLI component evaluates the relationship between an individual claim and its retrieved evidence.

Each claim-evidence pair is classified into one of three categories:

- **Entailment:** The evidence supports the claim.
- **Contradiction:** The evidence contradicts the claim.
- **Neutral:** The evidence does not provide sufficient information to establish support or contradiction.

**Technology:** cross-encoder/nli-deberta-v3-base.

#### 5. Source Reliability Assessment

The system evaluates retrieved sources using reliability-related attributes, including:

- Source domain.
- Source classification.
- HTTPS availability.
- Source reliability score.
- Source diversity.

The reliability assessment helps the decision engine consider the quality of retrieved evidence.

Source reliability scores are heuristic indicators and should not be interpreted as independent proof of factual accuracy.

#### 6. Decision Engine

The decision engine combines verification information to determine the final classification of each claim.

The decision process considers relevant evidence, NLI predictions, similarity scores, confidence values and source reliability information.

Each claim receives one of the following classifications:

| Classification | Description                                                           |
| -------------- | --------------------------------------------------------------------- |
| SUPPORTED      | Available evidence provides support for the claim.                    |
| CONTRADICTED   | Available evidence provides contradictory information.                |
| UNVERIFIABLE   | Available evidence is insufficient to establish the claim's validity. |

The decision policy uses a configurable minimum number of qualifying
publisher domains before assigning `SUPPORTED` or `CONTRADICTED`.
Repeated pages or subdomains from one publisher still count as one source.
This policy is configurable in `decision_engine.py`; it is not a calibrated
accuracy guarantee.
Evidence that fails the relevance, NLI-confidence, or source-reliability
filters is kept separately from evidence used for the verdict.

If all configured retrieval providers fail, the API returns a service error
instead of reporting the claim as `UNVERIFIABLE`. When retrieval partially
works, the response carries provider warnings so users can interpret results
with the available coverage in mind. If Tavily is not configured, the report
marks the search as partial and notes that only Wikipedia was queried.

The displayed claim score is a heuristic combination of NLI confidence and
source reliability. It is not a calibrated probability that the claim is
correct and is not the system's measured accuracy.

#### 7. Report Generation

The reporting component organizes the verification results into a structured output.

The results include claim-level decisions, confidence information, evidence details and explanations.

---

## 4. Technology Stack

### Backend

| Technology            | Purpose                       |
| --------------------- | ----------------------------- |
| Python 3.11           | Core implementation           |
| FastAPI               | Backend API development       |
| Ollama                | Local LLM execution           |
| Llama 3.2             | Claim extraction              |
| Sentence Transformers | Semantic embedding generation |
| all-MiniLM-L6-v2      | Evidence ranking              |
| DeBERTa NLI           | Claim-evidence verification   |
| Scikit-learn          | Similarity calculations       |
| Requests              | External API communication    |
| Wikipedia API         | Evidence retrieval            |
| Tavily API            | Web-based evidence retrieval  |

### Frontend

| Technology | Purpose                          |
| ---------- | -------------------------------- |
| React      | User interface                   |
| Vite       | Frontend development environment |
| JavaScript | Application logic                |
| CSS        | Interface styling                |
| Fetch API  | Backend communication            |

---

## 5. Project Structure

The project follows a modular architecture that separates verification components from the frontend application.

```text
VERI-CLAIM-AI/
│
├── main.py
├── backend_api.py
│
├── claim_extractor.py
├── evidence_retrieval.py
├── evidence_ranker.py
├── nli_verifier.py
├── source_reliability.py
├── decision_engine.py
├── evaluation.py
│
├── tests/
│   ├── test_input_validation.py
│   ├── test_data_contracts.py
│   └── ...
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── main.jsx
│   │   └── components/
│   │       ├── ClaimCard.jsx
│   │       ├── EvidenceGroup.jsx
│   │       └── ErrorBoundary.jsx
│   │
│   ├── index.html
│   └── package.json
│
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

The structure above represents the main project components. Additional modules and test files may be present as development progresses.

---

## 6. Installation and Setup

### Prerequisites

Ensure the following software is installed:

- Python 3.11
- Node.js and npm
- Git
- Ollama
- Llama 3.2 model

### Step 1: Clone the Repository

```bash
git clone https://github.com/Nandanpatgar/Veri-Claim-AI.git
```

```bash
cd VeriClaim-AI
```

### Step 2: Create a Python Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root.

Add the API credentials required by your configured evidence retrieval services.

Example:

```env
TAVILY_API_KEY=your_tavily_api_key
```

Keep API credentials private. Never commit the `.env` file to GitHub.

### Step 5: Configure Ollama

Install Ollama and download the required model:

```bash
ollama pull llama3.2
```

Ensure the Ollama service is running before executing the claim extraction pipeline.

### Step 6: Start the Backend

From the project root, execute:

```bash
python -m uvicorn backend_api:app --reload
uvicorn backend_api:app --reload
```

The backend will be available at:

```text
http://127.0.0.1:8000
```

### Step 7: Access API Documentation

Open:

```text
http://127.0.0.1:8000/docs
```

FastAPI provides interactive API documentation through Swagger UI.

### Step 8: Install Frontend Dependencies

Open a second terminal:

```bash
cd frontend
```

Install the dependencies:

```bash
npm install
```

### Step 9: Start the Frontend

```bash
npm run dev
```

The frontend will be available at:

```text
http://localhost:5173/
```

For a frontend served from another origin, set `VERICLAIM_CORS_ORIGINS` on
the backend to a comma-separated list of exact origins, and set
`VITE_API_BASE_URL` before building/starting Vite if the API is not at
`http://127.0.0.1:8000`. Example:

```text
VERICLAIM_CORS_ORIGINS=http://localhost:5173,https://vericlaim.example
VITE_API_BASE_URL=https://api.vericlaim.example
```

---

## 7. API Endpoints

The application exposes REST API endpoints through FastAPI.

| Method | Endpoint      | Description                                                 |
| ------ | ------------- | ----------------------------------------------------------- |
| GET    | `/api/health` | Checks backend availability and application health.         |
| POST   | `/api/verify` | Accepts text and initiates the claim verification pipeline. |

### Health Check

**Request:**

```http
GET /api/health
```

**Example Response:**

```json
{
  "status": "healthy",
  "application": "Veri-Claim-AI"
}
```

### Verify Text

**Request:**

```http
POST /api/verify
```

**Example Request Body:**

```json
{
  "text": "Virat Kohli is an Indian international cricketer."
}
```

The verification endpoint processes the submitted text and returns structured verification information.

For the complete request and response schemas, refer to the interactive Swagger documentation at `/docs`.

---

## 8. Verification Categories

Veri-Claim-AI evaluates individual claims using three primary classifications.

### Supported

A claim is classified as supported when the retrieved evidence provides sufficient support for the statement.

### Contradicted

A claim is classified as contradicted when the retrieved evidence provides information that conflicts with the statement.

### Unverifiable

A claim is classified as unverifiable when the available evidence is insufficient to establish whether the statement is correct or incorrect.

An absence of supporting evidence alone does not establish that a claim is false.

---

## 9. Testing

The project includes automated tests for validating important components of the verification system.

To execute the test suite:

```bash
python -m pytest tests -v
```

Individual test modules can also be executed:

```bash
python -m pytest tests/test_input_validation.py -v
```

```bash
python -m pytest tests/test_data_contracts.py -v
```

Testing helps validate input handling, data contracts and individual verification components.

### Labeled Evaluation

`evaluation.py` computes accuracy on covered predictions, macro F1, per-class
precision/recall/F1, false-positive and false-negative rates, a confusion
matrix, and classification coverage from caller-supplied labeled records.
`None` predictions are treated as abstentions and excluded from accuracy while
reducing coverage. The repository does not include a curated ground-truth
benchmark, so this tool does not imply or report a measured system accuracy
until evaluated against an independently labeled dataset.

---

## 10. Key Features

- Claim-level hallucination detection.
- Automated factual claim extraction.
- Evidence retrieval from external knowledge sources.
- Hybrid semantic and lexical evidence ranking.
- Semantic evidence ranking.
- Natural Language Inference-based verification.
- Source reliability assessment.
- Evidence-based claim classification.
- Explainable verification decisions.
- FastAPI-based backend integration.
- React-based web interface.
- Responsive multi-view interface with claim-level evidence explorer.
- Browser-local verification history and analytics based on completed reports.
- Text-file input and JSON/CSV report exports.
- Structured verification results.

---

## 11. Limitations

Although Veri-Claim-AI aims to improve the factual reliability of LLM-generated responses, certain limitations remain:

- Verification quality depends on the availability and relevance of retrieved evidence.
- External knowledge sources may contain outdated or conflicting information.
- NLI models can produce incorrect predictions.
- Semantic similarity does not independently establish factual correctness.
- Source reliability scoring is heuristic and may not capture every aspect of source credibility.
- Complex claims may require multiple pieces of evidence for accurate verification.
- Model predictions and confidence scores should not be interpreted as guarantees of factual truth.

The system is intended as an evidence-assisted verification tool rather than a replacement for expert fact-checking.

---

## 12. Future Enhancements

Potential improvements include:

- Integration with additional authoritative knowledge sources.
- Improved source credibility evaluation.
- Advanced claim decomposition for complex statements.
- Better handling of conflicting evidence.
- Improved confidence calibration.
- Claim-level evidence visualization.
- Downloadable PDF verification reports.
- Verification history and analytics.
- Support for multiple LLM providers.
- Benchmark-based evaluation of hallucination detection performance.

---

## 13. Research and Development Focus

The project explores the intersection of:

- Large Language Models.
- Natural Language Processing.
- Information Retrieval.
- Semantic Search.
- Natural Language Inference.
- Explainable Artificial Intelligence.
- Automated Fact Verification.

The central research focus is the use of evidence-grounded, claim-level verification to identify and explain factual inconsistencies in AI-generated responses.

---

## 14. Contributors

**Nandan Patgar**

Project: Veri-Claim-AI

GitHub: [Nandanpatgar](https://github.com/Nandanpatgar)

---

## 15. License

This project is developed for academic and research purposes.

A formal open-source license can be added when the project licensing terms are finalized.

---

**VERI-CLAIM-AI**

*Every Claim Verified by Evidence.*
