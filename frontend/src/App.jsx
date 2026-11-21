import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity, ArrowDown, ArrowRight, BarChart3, Check, ChevronDown,
  CircleAlert, Clock3, FileDown, FileText, History, Home, Layers3,
  Menu, MessageSquareText, Network, ShieldCheck, Sparkles,
  Trash2, Upload, X
} from "lucide-react";
import ClaimCard from "./components/ClaimCard.jsx";
import ErrorBoundary from "./components/ErrorBoundary.jsx";
import "./App.css";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const API_URL = `${API_BASE_URL}/api/verify`;
const MAX_LENGTH = 10000;
const HISTORY_KEY = "vericlaim.sessions.v1";
const MAX_HISTORY = 8;
const NAV_ITEMS = [
  { id: "home", label: "Home", icon: Home },
  { id: "verify", label: "Verify", icon: ShieldCheck },
  { id: "dashboard", label: "Dashboard", icon: BarChart3 },
  { id: "history", label: "History", icon: History },
  { id: "about", label: "About", icon: Layers3 },
];

function asObjectList(value) {
  return Array.isArray(value)
    ? value.filter((item) => item && typeof item === "object" && !Array.isArray(item))
    : [];
}

function readHistory() {
  try {
    const stored = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    return Array.isArray(stored) ? stored.filter((item) => item?.result && Array.isArray(item?.decisions)).slice(0, MAX_HISTORY) : [];
  } catch {
    return [];
  }
}

function extractReport(payload) {
  const data = payload?.data ?? payload;
  if (!data || typeof data !== "object" || Array.isArray(data)) {
    throw new Error("The backend response did not contain verification results.");
  }
  const decisionBlock = data.final_decisions;
  const decisions = Array.isArray(decisionBlock)
    ? asObjectList(decisionBlock)
    : asObjectList(decisionBlock?.decisions);
  return { data, decisions, summary: decisionBlock?.summary || {} };
}

function makeTitle(text) {
  const firstLine = String(text || "Verification session").split(/[.!?\n]/)[0].trim();
  return firstLine.length > 66 ? `${firstLine.slice(0, 63).trimEnd()}…` : firstLine || "Verification session";
}

function formatDate(value) {
  if (!value) return "Date unavailable";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Date unavailable" : new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(date);
}

function safeCount(value) {
  const number = Number(value);
  return Number.isFinite(number) && number >= 0 ? number : 0;
}

function saveBlob(content, type, filename) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function escapeCsv(value) {
  return `"${String(value ?? "").replaceAll('"', '""')}"`;
}

function IconButton({ label, onClick, children, className = "" }) {
  return <button className={`icon-button ${className}`} type="button" aria-label={label} title={label} onClick={onClick}>{children}</button>;
}

function SiteNav({ active, onNavigate }) {
  const [menuOpen, setMenuOpen] = useState(false);
  return (
    <header className="site-header">
      <div className="nav-shell">
        <button className="brand-lockup" type="button" onClick={() => onNavigate("home")} aria-label="Veri-Claim-AI home">
          <span className="brand-mark"><ShieldCheck size={20} strokeWidth={2.3} /></span>
          <span className="brand-type"><strong>VERI<span>·</span>CLAIM</strong><small>INTELLIGENCE, VERIFIED</small></span>
        </button>
        <button className="mobile-menu-toggle" type="button" aria-label={menuOpen ? "Close navigation" : "Open navigation"} aria-expanded={menuOpen} onClick={() => setMenuOpen((open) => !open)}>
          {menuOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
        <nav className={`primary-nav ${menuOpen ? "is-open" : ""}`} aria-label="Main navigation">
          {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
            <button className={`nav-link ${active === id ? "active" : ""}`} type="button" key={id} onClick={() => { onNavigate(id); setMenuOpen(false); }}>
              <Icon size={15} strokeWidth={1.8} /><span>{label}</span>
            </button>
          ))}
          <button className="nav-cta" type="button" onClick={() => { onNavigate("verify"); setMenuOpen(false); }}>Start a check <ArrowRight size={15} /></button>
        </nav>
      </div>
    </header>
  );
}

const FEATURES = [
  { number: "01", icon: MessageSquareText, title: "Claim by claim", copy: "Separate a long response into factual statements that can be checked on their own.", className: "feature-red" },
  { number: "02", icon: Network, title: "Evidence intelligence", copy: "Find relevant passages, compare meaning with exact terms, and keep source context attached.", className: "feature-violet" },
  { number: "03", icon: ShieldCheck, title: "Explainable decisions", copy: "See what supports, challenges, or leaves each claim unresolved—with reasons and provenance.", className: "feature-gold" },
];

function FeatureCard({ feature, index }) {
  const Icon = feature.icon;
  return (
    <article className={`feature-card ${feature.className}`} style={{ "--feature-index": index }}>
      <div className="feature-card-top"><span>{feature.number} / CAPABILITY</span><Icon size={21} strokeWidth={1.7} /></div>
      <div className="feature-art" aria-hidden="true"><span className="art-orbit orbit-one" /><span className="art-orbit orbit-two" /><span className="art-core"><Icon size={29} /></span></div>
      <div><h3>{feature.title}</h3><p>{feature.copy}</p></div>
      <ArrowDown className="feature-arrow" size={17} aria-hidden="true" />
    </article>
  );
}

function HeroSection({ onNavigate }) {
  return (
    <>
      <section className="hero-cinema">
        <div className="hero-grid" aria-hidden="true" />
        <div className="hero-glow hero-glow-red" aria-hidden="true" />
        <div className="hero-glow hero-glow-blue" aria-hidden="true" />
        <div className="hero-content">
          <div className="eyebrow"><span className="live-dot" /> EVIDENCE-GROUNDED AI VERIFICATION</div>
          <h1>Truth deserves<br /><span>to be verified.</span></h1>
          <p className="hero-copy">An intelligent evidence-grounded platform that investigates factual claims, evaluates sources, and surfaces uncertainty in AI-generated content.</p>
          <div className="hero-actions">
            <button className="button-primary" type="button" onClick={() => onNavigate("verify")}><Sparkles size={17} /> Start verification <ArrowRight size={16} /></button>
            <button className="button-quiet" type="button" onClick={() => onNavigate("about")}>Explore how it works <ArrowDown size={15} /></button>
          </div>
          <div className="hero-capabilities">
            <span><Check size={14} /> Claim-level analysis</span><span><Check size={14} /> Evidence-linked results</span><span><Check size={14} /> Transparent uncertainty</span>
          </div>
        </div>
        <div className="hero-index" aria-hidden="true"><span>VERIFICATION SYSTEM</span><strong>01 — 05</strong></div>
        <div className="hero-bottom-rule" aria-hidden="true"><span /></div>
      </section>
      <section className="feature-section content-shell">
        <div className="section-heading"><div><span className="eyebrow">A CLEARER VIEW OF WHAT’S TRUE</span><h2>Evidence is the story.</h2></div><p>Every result connects a claim to the information behind its classification.</p></div>
        <div className="feature-grid">{FEATURES.map((feature, index) => <FeatureCard key={feature.number} feature={feature} index={index} />)}</div>
      </section>
      <section className="manifesto-band"><div className="manifesto-inner"><span className="eyebrow">THE VERI-CLAIM METHOD</span><p>From generated text to grounded evidence.<br /><em>One claim at a time.</em></p><button className="text-link" type="button" onClick={() => onNavigate("verify")}>Enter the verification workspace <ArrowRight size={16} /></button></div><div className="manifesto-visual" aria-hidden="true"><div className="manifesto-ring ring-a" /><div className="manifesto-ring ring-b" /><div className="manifesto-center"><ShieldCheck size={31} /></div></div></section>
    </>
  );
}

function ProcessingState() {
  return (
    <div className="processing-panel" role="status" aria-live="polite">
      <div className="processing-spinner"><span /></div>
      <div className="processing-copy"><span className="eyebrow">ANALYSIS IN PROGRESS</span><h3>Following the evidence.</h3><p>The backend is extracting claims, retrieving sources, and evaluating the evidence. It does not stream intermediate stage updates.</p></div>
      <div className="processing-meta"><span className="live-dot" /> Request active</div>
    </div>
  );
}

function VerificationWorkspace({ input, setInput, onSubmit, loading, error, onFile, results, onExport, onExportCsv, onClear }) {
  const fileRef = useRef(null);
  const [filter, setFilter] = useState("ALL");
  const decisions = results?.decisions || [];
  const summary = results?.summary || {};
  const total = safeCount(summary.total_claims ?? decisions.length);
  const supported = safeCount(summary.supported);
  const contradicted = safeCount(summary.contradicted);
  const unverifiable = safeCount(summary.unverifiable);
  const verifiableRate = total ? Math.round(((supported + contradicted) / total) * 100) : 0;
  const filteredDecisions = filter === "ALL" ? decisions : decisions.filter((item) => String(item.decision).toUpperCase() === filter);
  const duration = Number(results?.data?.metadata?.execution_time_seconds);

  return (
    <section className="workspace-page content-shell" id="verify-workspace">
      <div className="page-heading">
        <div><div className="eyebrow"><span className="live-dot" /> VERIFICATION WORKSPACE</div><h1>Put a claim<br /><span>under the lens.</span></h1><p>Submit AI-generated text to trace its factual claims back to evidence.</p></div>
        <div className="workspace-stamp"><ShieldCheck size={21} /><span>LOCAL WORKSPACE</span><small>Text is sent to your configured backend.</small></div>
      </div>

      <form className="input-console" onSubmit={onSubmit}>
        <div className="console-head"><div><span className="console-kicker">01 / SOURCE TEXT</span><h2>What would you like to verify?</h2></div><span className="console-format"><span className="format-dot" /> PLAIN TEXT</span></div>
        <label className="sr-only" htmlFor="claim-input">Text to verify</label>
        <textarea id="claim-input" value={input} onChange={(event) => setInput(event.target.value.slice(0, MAX_LENGTH))} placeholder="Paste a response, article, or factual statement here…" rows={9} maxLength={MAX_LENGTH} disabled={loading} />
        <div className="console-tools"><div className="console-guidance"><span className="char-count">{input.length.toLocaleString()} <i>/ {MAX_LENGTH.toLocaleString()}</i></span><span className="guidance-divider" /><span>Specific factual statements produce clearer results.</span></div><div className="console-actions"><input ref={fileRef} type="file" accept=".txt,.md,text/plain,text/markdown" className="sr-only" onChange={(event) => { onFile(event); event.target.value = ""; }} /><button className="button-secondary" type="button" disabled={loading} onClick={() => fileRef.current?.click()}><Upload size={15} /> Upload .txt</button><button className="button-primary verify-button" type="submit" disabled={loading || input.trim().length < 10}>{loading ? <><span className="mini-spinner" /> Verifying…</> : <>Verify claims <ArrowRight size={16} /></>}</button></div></div>
      </form>
      <div className="workspace-footnote"><CircleAlert size={14} /><span>Results are evidence-assisted classifications, not guarantees. Scores are heuristic unless explicitly identified otherwise.</span></div>
      {error && <div className="error-banner" role="alert"><CircleAlert size={18} /><div><strong>Couldn’t complete verification</strong><p>{error}</p></div><button type="button" aria-label="Dismiss error" onClick={onClear}><X size={17} /></button></div>}
      {loading && <ProcessingState />}

      {results && !loading && <section className="report-section" aria-live="polite">
        <div className="report-title-row"><div><span className="eyebrow">02 / VERIFICATION REPORT</span><h2>Your evidence brief.</h2><p>{results.data?.metadata?.completed_at ? `Completed ${formatDate(results.data.metadata.completed_at)}` : "Claim-level analysis of the submitted text."}{Number.isFinite(duration) ? ` · ${duration.toFixed(1)} sec` : ""}</p></div><div className="report-actions"><button className="button-secondary" type="button" onClick={onExportCsv}><FileText size={15} /> Export CSV</button><button className="button-secondary" type="button" onClick={onExport}><FileDown size={15} /> Export JSON</button></div></div>
        <div className="report-overview">
          <div className="overview-total"><span className="eyebrow">CLAIMS REVIEWED</span><strong>{total}</strong><div className="coverage-line"><span>Classifiable</span><b>{verifiableRate}%</b></div><div className="coverage-track"><span style={{ width: `${verifiableRate}%` }} /></div><small>Supported + contradicted claims</small></div>
          <div className="overview-breakdown"><div className="distribution-heading"><span>Classification distribution</span><span>{total} total</span></div><div className="distribution-track" aria-label={`${supported} supported, ${contradicted} contradicted, ${unverifiable} unverifiable`}>{total > 0 && <><span className="dist-supported" style={{ width: `${(supported / total) * 100}%` }} /><span className="dist-contradicted" style={{ width: `${(contradicted / total) * 100}%` }} /><span className="dist-unverifiable" style={{ width: `${(unverifiable / total) * 100}%` }} /></>}</div><div className="distribution-legend"><span><i className="legend-dot supported" />Supported <b>{supported}</b></span><span><i className="legend-dot contradicted" />Contradicted <b>{contradicted}</b></span><span><i className="legend-dot unverifiable" />Unverifiable <b>{unverifiable}</b></span></div><div className="report-integrity"><ShieldCheck size={16} /><span>Evidence-backed classifications</span><span className="integrity-status">REPORT READY</span></div></div>
        </div>
        {results.data?.metadata?.evidence_retrieval_status === "partial" && <div className="report-warning"><CircleAlert size={16} /><span>At least one retrieval provider was unavailable. Review each claim’s retrieval notes before relying on the report.</span></div>}
        <div className="claims-toolbar"><div><span className="eyebrow">CLAIM EXPLORER</span><h3>Findings <span>{decisions.length}</span></h3></div><div className="filter-pills" role="group" aria-label="Filter claims">{["ALL", "SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"].map((value) => <button key={value} type="button" className={filter === value ? "selected" : ""} onClick={() => setFilter(value)}>{value === "ALL" ? "All" : value[0] + value.slice(1).toLowerCase()}</button>)}</div></div>
        {filteredDecisions.length ? <div className="claim-list">{filteredDecisions.map((claim, index) => <ClaimCard key={claim.claim_id ?? index} claim={claim} index={index} />)}</div> : <div className="empty-state"><ShieldCheck size={26} /><h3>No claims in this view</h3><p>Try another classification filter.</p></div>}
        <p className="report-disclaimer">The evidence score shown per claim is a composite heuristic. It is not a calibrated probability that a claim is true.</p>
      </section>}
    </section>
  );
}

function MetricCard({ label, value, detail, icon: Icon, tone = "" }) {
  return <article className={`metric-card ${tone}`}><div className="metric-icon"><Icon size={18} /></div><span>{label}</span><strong>{value}</strong><small>{detail}</small></article>;
}

function DashboardView({ sessions, onNavigate, onOpen }) {
  const totals = useMemo(() => sessions.reduce((acc, session) => {
    const summary = session.summary || {};
    acc.sessions += 1;
    acc.claims += safeCount(summary.total_claims ?? session.decisions.length);
    acc.supported += safeCount(summary.supported);
    acc.contradicted += safeCount(summary.contradicted);
    acc.unverifiable += safeCount(summary.unverifiable);
    acc.duration += safeCount(session.result?.metadata?.execution_time_seconds);
    return acc;
  }, { sessions: 0, claims: 0, supported: 0, contradicted: 0, unverifiable: 0, duration: 0 }), [sessions]);
  const averageDuration = totals.sessions ? (totals.duration / totals.sessions).toFixed(1) : "—";
  const maxClaims = Math.max(1, ...sessions.slice(0, 7).map((session) => safeCount(session.summary?.total_claims ?? session.decisions.length)));
  const trend = [...sessions].slice(0, 7).reverse();
  return (
    <section className="content-shell interior-page">
      <div className="page-heading compact-heading"><div><div className="eyebrow">YOUR VERIFICATION ACTIVITY</div><h1>Signal, over time.</h1><p>Private analytics calculated from reports saved in this browser.</p></div><button className="button-primary" type="button" onClick={() => onNavigate("verify")}>New verification <ArrowRight size={16} /></button></div>
      {!sessions.length ? <div className="empty-state dashboard-empty"><BarChart3 size={30} /><h3>Your dashboard starts with a verification.</h3><p>Once you verify text, actual claim counts and durations will appear here.</p><button className="button-primary" type="button" onClick={() => onNavigate("verify")}>Start a verification <ArrowRight size={15} /></button></div> : <>
        <div className="metric-grid"><MetricCard label="Sessions" value={totals.sessions} detail="Saved in this browser" icon={Activity} /><MetricCard label="Claims reviewed" value={totals.claims} detail="Across saved reports" icon={MessageSquareText} /><MetricCard label="Average run time" value={`${averageDuration}${averageDuration === "—" ? "" : "s"}`} detail="Backend-reported duration" icon={Clock3} /><MetricCard label="Unverifiable" value={totals.unverifiable} detail="Claims without a clear verdict" icon={CircleAlert} tone="metric-amber" /></div>
        <div className="dashboard-grid"><section className="panel trend-panel"><div className="panel-heading"><div><span className="eyebrow">RECENT ACTIVITY</span><h2>Claims per session</h2></div><span className="panel-range">LAST {trend.length}</span></div><div className="trend-chart" role="img" aria-label={`Claim counts in the last ${trend.length} saved sessions`}>{trend.map((session) => { const count = safeCount(session.summary?.total_claims ?? session.decisions.length); const height = Math.max(8, (count / maxClaims) * 100); return <button className="trend-column" type="button" key={session.id} onClick={() => onOpen(session)} aria-label={`${session.title}: ${count} claims`}><span className="trend-value">{count}</span><span className="trend-bar" style={{ height: `${height}%` }} /><small>{new Date(session.createdAt).toLocaleDateString(undefined, { month: "short", day: "numeric" })}</small></button>; })}</div><div className="chart-footnote">Claim totals from completed reports only.</div></section>
          <section className="panel verdict-panel"><div className="panel-heading"><div><span className="eyebrow">CLASSIFICATION MIX</span><h2>Across saved claims</h2></div><ShieldCheck size={18} /></div><div className="mix-list">{[["Supported", totals.supported, "supported"], ["Contradicted", totals.contradicted, "contradicted"], ["Unverifiable", totals.unverifiable, "unverifiable"]].map(([label, count, tone]) => <div className="mix-row" key={tone}><div><span><i className={`legend-dot ${tone}`} />{label}</span><b>{count}</b></div><div className="mix-track"><span className={`mix-fill ${tone}`} style={{ width: `${totals.claims ? (count / totals.claims) * 100 : 0}%` }} /></div></div>)}</div><p className="chart-footnote">Percentages are derived from the classifications returned by your backend.</p></section></div>
        <section className="panel recent-panel"><div className="panel-heading"><div><span className="eyebrow">RECENT REPORTS</span><h2>Continue where you left off.</h2></div><button className="text-link" type="button" onClick={() => onNavigate("history")}>View history <ArrowRight size={15} /></button></div><div className="recent-list">{sessions.slice(0, 3).map((session) => <HistoryRow key={session.id} session={session} onOpen={onOpen} />)}</div></section>
      </>}
    </section>
  );
}

function HistoryRow({ session, onOpen, onDelete }) {
  const summary = session.summary || {};
  const date = formatDate(session.result?.metadata?.completed_at || session.createdAt);
  return <article className="history-row"><div className="history-row-icon"><FileText size={18} /></div><button className="history-row-main" type="button" onClick={() => onOpen(session)}><strong>{session.title || "Verification report"}</strong><span>{date} <i>·</i> {safeCount(summary.total_claims ?? session.decisions.length)} claims</span></button><div className="history-counts"><span className="count-supported">{safeCount(summary.supported)} supported</span><span className="count-contradicted">{safeCount(summary.contradicted)} contradicted</span><span className="count-unverifiable">{safeCount(summary.unverifiable)} unverifiable</span></div><button className="history-open" type="button" onClick={() => onOpen(session)}>Open <ArrowRight size={14} /></button>{onDelete && <IconButton label={`Delete ${session.title}`} className="history-delete" onClick={() => onDelete(session.id)}><Trash2 size={15} /></IconButton>}</article>;
}

function HistoryView({ sessions, onOpen, onDelete, onNavigate }) {
  return <section className="content-shell interior-page">
    <div className="page-heading compact-heading"><div><div className="eyebrow">LOCAL REPORT ARCHIVE</div><h1>Every check,<br /><span>within reach.</span></h1><p>Reports are saved in this browser only. They are not synced to a server.</p></div><span className="history-count-badge"><History size={15} /> {sessions.length} / {MAX_HISTORY} saved</span></div>
    {sessions.length ? <div className="history-list">{sessions.map((session) => <HistoryRow key={session.id} session={session} onOpen={onOpen} onDelete={onDelete} />)}</div> : <div className="empty-state"><History size={29} /><h3>No saved reports yet.</h3><p>Complete a verification and it will appear here for this browser.</p><button className="button-primary" type="button" onClick={() => onNavigate("verify")}>Verify some text <ArrowRight size={15} /></button></div>}
    <div className="privacy-note"><ShieldCheck size={16} /><span>Browser-local history can include submitted text and evidence returned by your backend. Clear a report here to remove its saved copy.</span></div>
  </section>;
}

function AboutView({ onNavigate }) {
  return <section className="content-shell interior-page about-page">
    <div className="page-heading compact-heading"><div><div className="eyebrow">HOW VERI-CLAIM WORKS</div><h1>Trace the claim.<br /><span>See the evidence.</span></h1><p>A layered workflow designed to make factual checks easier to inspect.</p></div><div className="about-seal"><ShieldCheck size={24} /><span>CLAIM-LEVEL<br />VERIFICATION</span></div></div>
    <div className="method-list">{[
      ["01", "Extract", "A local language model identifies factual claims in the submitted text."],
      ["02", "Retrieve", "The backend searches Wikipedia and, when configured, Tavily for relevant passages."],
      ["03", "Rank", "MiniLM semantic relevance is combined with exact-term overlap to order evidence."],
      ["04", "Evaluate", "DeBERTa NLI estimates whether each passage supports, contradicts, or is neutral to a claim."],
      ["05", "Decide", "A heuristic decision engine weighs evidence relevance, NLI, source reliability, and source diversity."],
    ].map(([number, title, copy], index) => <article className="method-row" key={number}><span className="method-number">{number}</span><div className="method-connector"><span /></div><div className="method-content"><span className="eyebrow">STAGE {number}</span><h2>{title}</h2><p>{copy}</p></div><span className="method-watermark">0{index + 1}</span></article>)}</div>
    <div className="about-limit"><CircleAlert size={18} /><div><strong>Evidence-assisted, not infallible.</strong><p>Retrieval may miss relevant sources, source scoring is heuristic, and model scores are not calibrated probabilities. Inspect the linked passages and uncertainty notes before relying on a classification.</p></div></div>
    <button className="button-primary about-cta" type="button" onClick={() => onNavigate("verify")}>Enter the workspace <ArrowRight size={16} /></button>
  </section>;
}

function AppContent() {
  const [active, setActive] = useState("home");
  const [input, setInput] = useState("");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [toast, setToast] = useState("");
  const [sessions, setSessions] = useState(readHistory);

  useEffect(() => {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(sessions.slice(0, MAX_HISTORY)));
    } catch {
      window.dispatchEvent(new Event("vericlaim-storage-error"));
    }
  }, [sessions]);

  useEffect(() => {
    const handleStorageError = () => setToast("Browser storage is full. New reports may not remain in local history.");
    window.addEventListener("vericlaim-storage-error", handleStorageError);
    return () => window.removeEventListener("vericlaim-storage-error", handleStorageError);
  }, []);

  useEffect(() => {
    if (!toast) return undefined;
    const timeout = window.setTimeout(() => setToast(""), 4800);
    return () => window.clearTimeout(timeout);
  }, [toast]);

  function navigate(page) {
    setActive(page);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  async function verifyText(event) {
    event.preventDefault();
    const text = input.trim();
    if (text.length < 10) {
      setError("Enter at least 10 characters of text to verify.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ text }),
      });
      const raw = await response.text();
      let payload;
      try { payload = raw ? JSON.parse(raw) : null; } catch { throw new Error("The backend returned an unreadable response. Please try again."); }
      if (!response.ok) {
        const detail = payload?.detail;
        const message = typeof detail === "string" ? detail : detail?.message || payload?.message;
        throw new Error(message || `Verification failed (HTTP ${response.status}).`);
      }
      const report = extractReport(payload);
      if (!report.decisions.length) throw new Error("The request completed, but the backend returned no claim decisions. Try clear factual statements.");
      setResults(report);
      const session = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        createdAt: new Date().toISOString(),
        title: makeTitle(text),
        summary: report.summary,
        decisions: report.decisions,
        result: report.data,
      };
      setSessions((current) => [session, ...current.filter((item) => item.title !== session.title || item.createdAt !== session.createdAt)].slice(0, MAX_HISTORY));
      setToast("Verification report is ready and saved in this browser.");
      window.setTimeout(() => document.getElementById("verify-workspace")?.scrollIntoView({ behavior: "smooth", block: "start" }), 80);
    } catch (requestError) {
      setError(requestError instanceof TypeError ? "Cannot connect to the backend. Start FastAPI and try again." : requestError.message || "Verification failed. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  function handleFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (file.size > 1024 * 1024) {
      setError("Text files must be smaller than 1 MB.");
      return;
    }
    if (!/\.(txt|md)$/i.test(file.name)) {
      setError("Choose a .txt or .md text file.");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result || "");
      setInput(text.slice(0, MAX_LENGTH));
      setError("");
      setToast(text.length > MAX_LENGTH
        ? `Text file loaded and trimmed to ${MAX_LENGTH.toLocaleString()} characters.`
        : "Text file loaded into the verification workspace.");
    };
    reader.onerror = () => setError("Could not read that file. Try another text file.");
    reader.readAsText(file);
  }

  function exportJson(report = results?.data) {
    if (!report) return;
    saveBlob(JSON.stringify(report, null, 2), "application/json", "veri-claim-report.json");
  }

  function exportCsv(report = results) {
    if (!report?.decisions?.length) return;
    const columns = ["claim_id", "claim", "decision", "evidence_confidence", "decision_reason", "supporting_evidence", "contradicting_evidence"];
    const rows = report.decisions.map((claim) => [
      claim.claim_id, claim.claim, claim.decision, claim.evidence_confidence, claim.decision_reason,
      claim.evidence_summary?.supporting_evidence_count, claim.evidence_summary?.contradicting_evidence_count,
    ]);
    saveBlob([columns, ...rows].map((row) => row.map(escapeCsv).join(",")).join("\r\n"), "text/csv;charset=utf-8", "veri-claim-claims.csv");
  }

  function openSession(session) {
    setResults(extractReport(session.result));
    setInput(session.result?.original_input || "");
    setError("");
    setActive("verify");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function deleteSession(id) {
    setSessions((current) => current.filter((session) => session.id !== id));
    setToast("Saved report removed from this browser.");
  }

  return (
    <div className="app-shell">
      <SiteNav active={active} onNavigate={navigate} />
      <main>
        {active === "home" && <HeroSection onNavigate={navigate} />}
        {active === "verify" && <VerificationWorkspace input={input} setInput={(value) => { setInput(value); if (error) setError(""); }} onSubmit={verifyText} loading={loading} error={error} onFile={handleFile} results={results} onExport={() => exportJson()} onExportCsv={() => exportCsv()} onClear={() => setError("")} />}
        {active === "dashboard" && <DashboardView sessions={sessions} onNavigate={navigate} onOpen={openSession} />}
        {active === "history" && <HistoryView sessions={sessions} onOpen={openSession} onDelete={deleteSession} onNavigate={navigate} />}
        {active === "about" && <AboutView onNavigate={navigate} />}
      </main>
      <footer className="site-footer"><button className="footer-brand" type="button" onClick={() => navigate("home")}><ShieldCheck size={17} /> VERI·CLAIM</button><span>Evidence first. Certainty, earned.</span><span className="footer-version">VERI-CLAIM-AI 2.0</span></footer>
      {toast && <div className="toast-message" role="status"><Check size={16} /><span>{toast}</span><IconButton label="Dismiss notification" onClick={() => setToast("")}><X size={14} /></IconButton></div>}
      <button className="scroll-top" type="button" aria-label="Back to top" onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}><ChevronDown size={16} /></button>
    </div>
  );
}

export default function App() {
  return <ErrorBoundary><AppContent /></ErrorBoundary>;
}
