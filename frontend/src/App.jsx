import { useState } from "react";
import ClaimCard from "./components/ClaimCard.jsx";
import ErrorBoundary from "./components/ErrorBoundary.jsx";
import "./App.css";

const API_URL = "http://127.0.0.1:8000/api/verify";
const MAX_LENGTH = 10000;
const SAMPLE_TEXT = "VCET Puttur was established in 2001. Mrs. Akshaya D. Shetty is listed as an Assistant Professor at VCET. The official profile lists Digital Electronics and Communication as her specialization.";

function asObjectList(value) {
  return Array.isArray(value)
    ? value.filter((item) => item && typeof item === "object" && !Array.isArray(item))
    : [];
}

function VerificationApp() {
  const [input, setInput] = useState("");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function verifyText(event) {
    event.preventDefault();
    if (input.trim().length < 10) {
      setError("Enter at least 10 characters of text to verify.");
      return;
    }

    setLoading(true);
    setError("");
    setResults(null);
    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ text: input.trim() }),
      });
      const raw = await response.text();
      let payload;
      try {
        payload = raw ? JSON.parse(raw) : null;
      } catch {
        throw new Error("The backend returned an unreadable response. Please try again.");
      }

      if (!response.ok) {
        const detail = payload?.detail;
        const message = typeof detail === "string" ? detail : detail?.message || payload?.message;
        throw new Error(message || `Verification failed (HTTP ${response.status}).`);
      }

      const data = payload?.data ?? payload;
      if (!data || typeof data !== "object" || Array.isArray(data)) {
        throw new Error("The backend response did not contain verification results.");
      }
      const decisionsValue = data.final_decisions;
      const decisions = Array.isArray(decisionsValue)
        ? asObjectList(decisionsValue)
        : asObjectList(decisionsValue?.decisions);
      setResults({ data, decisions, summary: decisionsValue?.summary || {} });
      if (!decisions.length) {
        setError("The request completed, but the backend returned no claim decisions. Try clear factual statements.");
      }
    } catch (requestError) {
      setError(requestError instanceof TypeError
        ? "Cannot connect to the backend. Start FastAPI and try again."
        : requestError.message || "Verification failed. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  function downloadReport() {
    if (!results?.data) return;
    const blob = new Blob([JSON.stringify(results.data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "veri-claim-report.json";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  const counts = results?.summary || {};
  const summaryCards = [
    ["Total Claims", counts.total_claims ?? results?.decisions.length ?? 0, ""],
    ["Supported", counts.supported ?? 0, "supported-card"],
    ["Contradicted", counts.contradicted ?? 0, "contradicted-card"],
    ["Unverifiable", counts.unverifiable ?? 0, "unverifiable-card"],
  ];

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="logo">VERI-CLAIM-AI</div>
          <span className="header-tag">AI Hallucination Detection</span>
        </div>
      </header>
      <main className="container">
        <section className="hero">
          <h1>Verify Every Claim.</h1>
          <p>Detect AI hallucinations through evidence-based claim verification.</p>
        </section>
        <form className="input-section" onSubmit={verifyText}>
          <label htmlFor="claim-input">Enter text to verify</label>
          <div className="sample-row">
            <span>Need a quick example?</span>
            <button type="button" className="sample-btn" onClick={() => { setInput(SAMPLE_TEXT); setError(""); }} disabled={loading}>Use VCET sample</button>
          </div>
          <textarea id="claim-input" value={input} onChange={(event) => setInput(event.target.value.slice(0, MAX_LENGTH))} placeholder="Paste your AI-generated response here..." rows={7} maxLength={MAX_LENGTH} disabled={loading} />
          <div className="input-footer">
            <span>{input.length.toLocaleString()} / {MAX_LENGTH.toLocaleString()} characters</span>
            <button type="submit" disabled={loading || input.trim().length < 10}>
              {loading ? <><span className="button-spinner" /> Verifying…</> : "Verify Claims →"}
            </button>
          </div>
        </form>
        {loading && <div className="loading" role="status" aria-live="polite">
          <div className="spinner" />
          <p>Analyzing claims and retrieving evidence…</p>
          <small>This may take a minute depending on the models.</small>
        </div>}
        {error && <div className="error" role="alert">{error}</div>}
        {results && <section className="results-section" aria-live="polite">
          <div className="results-heading">
            <div><h2>Verification Results</h2><p>Claim-level analysis of your submitted text.</p></div>
            <button type="button" className="download-btn" onClick={downloadReport}>Download JSON</button>
          </div>
          <div className="summary">
            {summaryCards.map(([label, value, style]) => <div className={`summary-card ${style}`} key={label}>
              <span>{label}</span><strong>{Number.isFinite(Number(value)) ? value : 0}</strong>
            </div>)}
          </div>
          {!results.decisions.length
            ? <div className="empty-results">No claim decisions were included in this response.</div>
            : <div className="claims">{results.decisions.map((claim, index) => <ClaimCard key={claim.claim_id ?? index} claim={claim} index={index} />)}</div>}
        </section>}
      </main>
      <footer>Veri-Claim-AI · Evidence-Based Hallucination Detection</footer>
    </div>
  );
}

export default function App() {
  return <ErrorBoundary><VerificationApp /></ErrorBoundary>;
}
