import { Component, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000/api/verify";
const MAX_LENGTH = 10000;

class ErrorBoundary extends Component {
  state = { hasError: false };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  render() {
    if (this.state.hasError) {
      return <main className="fatal-error"><h1>Something went wrong</h1><p>The results could not be displayed. Reload the page and try again.</p></main>;
    }
    return this.props.children;
  }
}

function safeArray(value) {
  return Array.isArray(value) ? value.filter((item) => item && typeof item === "object" && !Array.isArray(item)) : [];
}

function formatScore(value, percent = true) {
  const score = Number(value);
  if (!Number.isFinite(score)) return "Not available";
  return `${(percent ? score * 100 : score).toFixed(1)}%`;
}

function EvidenceGroup({ title, items }) {
  return (
    <section className="evidence-group">
      <h4>{title} <span>{items.length}</span></h4>
      {items.length ? items.map((item, index) => (
        <article className="evidence-item" key={item.evidence_id ?? `${title}-${index}`}>
          <div className="source-heading">
            <strong>{item.source || item.title || item.domain || "Evidence source"}</strong>
            {item.url && <a href={item.url} target="_blank" rel="noreferrer">View source ↗</a>}
          </div>
          {item.title && item.source && <p className="source-title">{item.title}</p>}
          {item.evidence_text && <p>{item.evidence_text}</p>}
          <div className="source-meta">
            {item.reliability_score != null && <span>Reliability {formatScore(item.reliability_score)}</span>}
            {item.nli_confidence != null && <span>NLI {formatScore(item.nli_confidence)}</span>}
            {item.nli_probabilities && typeof item.nli_probabilities === "object" && item.nli_probabilities.entailment != null && <span>NLI: {Object.entries(item.nli_probabilities).map(([label, score]) => `${label} ${formatScore(score)}`).join(" · ")}</span>}
          </div>
        </article>
      )) : <p className="no-evidence">No {title.toLowerCase()} evidence returned.</p>}
    </section>
  );
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
      try { payload = raw ? JSON.parse(raw) : null; }
      catch { throw new Error("The backend returned an unreadable response. Please try again."); }
      if (!response.ok) {
        const detail = payload?.detail;
        const message = typeof detail === "string" ? detail : detail?.message || payload?.message;
        throw new Error(message || `Verification failed (HTTP ${response.status}).`);
      }
      const data = payload?.data ?? payload;
      if (!data || typeof data !== "object" || Array.isArray(data)) throw new Error("The backend response did not contain verification results.");
      const decisionsValue = data.final_decisions;
      const decisions = Array.isArray(decisionsValue) ? decisionsValue : safeArray(decisionsValue?.decisions);
      setResults({ data, decisions, summary: decisionsValue?.summary || {} });
      if (!decisions.length) setError("The request completed, but the backend returned no claim decisions. Try a longer text with clear factual statements.");
    } catch (err) {
      setError(err instanceof TypeError ? "Cannot connect to the backend. Start FastAPI and try again." : err.message || "Verification failed. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  const counts = results?.summary || {};
  const downloadReport = () => {
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
  };

  return (
    <div className="app">
      <header className="header"><div className="header-inner"><div className="logo">VERI-CLAIM-AI</div><span className="header-tag">AI Hallucination Detection</span></div></header>
      <main className="container">
        <section className="hero"><h1>Verify Every Claim.</h1><p>Detect AI hallucinations through evidence-based claim verification.</p></section>
        <form className="input-section" onSubmit={verifyText}>
          <label htmlFor="claim-input">Enter text to verify</label>
          <textarea id="claim-input" value={input} onChange={(event) => setInput(event.target.value.slice(0, MAX_LENGTH))} placeholder="Paste your AI-generated response here..." rows={7} maxLength={MAX_LENGTH} disabled={loading} />
          <div className="input-footer"><span>{input.length.toLocaleString()} / {MAX_LENGTH.toLocaleString()} characters</span><button type="submit" disabled={loading || input.trim().length < 10}>{loading ? <><span className="button-spinner" /> Verifying…</> : "Verify Claims →"}</button></div>
        </form>
        {loading && <div className="loading" role="status" aria-live="polite"><div className="spinner" /><p>Analyzing claims and retrieving evidence…</p><small>This may take a minute depending on the models.</small></div>}
        {error && <div className="error" role="alert">{error}</div>}
        {results && <section className="results-section" aria-live="polite">
          <div className="results-heading"><div><h2>Verification Results</h2><p>Claim-level analysis of your submitted text.</p></div><button type="button" className="download-btn" onClick={downloadReport}>Download JSON</button></div>
          <div className="summary">
            {[["Total Claims", counts.total_claims ?? results.decisions.length, ""], ["Supported", counts.supported ?? 0, "supported-card"], ["Contradicted", counts.contradicted ?? 0, "contradicted-card"], ["Unverifiable", counts.unverifiable ?? 0, "unverifiable-card"]].map(([label, value, style]) => <div className={`summary-card ${style}`} key={label}><span>{label}</span><strong>{Number.isFinite(Number(value)) ? value : 0}</strong></div>)}
          </div>
          {!results.decisions.length ? <div className="empty-results">No claim decisions were included in this response.</div> : <div className="claims">{results.decisions.map((claim, index) => {
            const rawStatus = String(claim.decision || "UNVERIFIABLE").toUpperCase();
            const status = ["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"].includes(rawStatus) ? rawStatus.toLowerCase() : "unverifiable";
            const evidence = claim.evidence_details && typeof claim.evidence_details === "object" ? claim.evidence_details : {};
            const confidence = Number(claim.evidence_confidence);
            return <article className="claim-card" key={claim.claim_id ?? index}>
              <div className="claim-top"><span className="claim-number">CLAIM {claim.claim_id ?? index + 1}</span><span className={`status ${status}`}>{status.toUpperCase()}</span></div>
              <h3>{typeof claim.claim === "string" ? claim.claim : "Claim text unavailable"}</h3>
              <div className="confidence"><span>Evidence confidence</span><strong>{formatScore(confidence)}</strong></div>
              <div className="progress" aria-label={`Evidence confidence ${formatScore(confidence)}`}><div className={`progress-fill ${status}`} style={{ width: `${Number.isFinite(confidence) ? Math.max(0, Math.min(confidence * 100, 100)) : 0}%` }} /></div>
              {claim.decision_reason && <p className="reason">{claim.decision_reason}</p>}
              <div className="evidence"><EvidenceGroup title="Supporting evidence" items={safeArray(evidence.supporting)} /><EvidenceGroup title="Contradicting evidence" items={safeArray(evidence.contradicting)} />{safeArray(evidence.neutral).length > 0 && <EvidenceGroup title="Neutral evidence" items={safeArray(evidence.neutral)} />}</div>
            </article>;
          })}</div>}
        </section>}
      </main>
      <footer>Veri-Claim-AI · Evidence-Based Hallucination Detection</footer>
    </div>
  );
}

export default function App() { return <ErrorBoundary><VerificationApp /></ErrorBoundary>; }
