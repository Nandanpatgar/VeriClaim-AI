import { useState } from "react";
import { ArrowDownRight, CircleAlert, Minus, ShieldCheck, ShieldX } from "lucide-react";
import EvidenceGroup from "./EvidenceGroup.jsx";

function safeArray(value) {
  return Array.isArray(value) ? value.filter((item) => item && typeof item === "object" && !Array.isArray(item)) : [];
}

function formatScore(value) {
  const score = Number(value);
  return Number.isFinite(score) ? `${(score * 100).toFixed(1)}%` : "Not available";
}

const STATUS_META = {
  supported: { label: "Supported", icon: ShieldCheck, eyebrow: "EVIDENCE SUPPORTS THIS CLAIM" },
  contradicted: { label: "Contradicted", icon: ShieldX, eyebrow: "EVIDENCE CHALLENGES THIS CLAIM" },
  unverifiable: { label: "Unverifiable", icon: CircleAlert, eyebrow: "EVIDENCE IS INCONCLUSIVE" },
};

export default function ClaimCard({ claim, index }) {
  const [expanded, setExpanded] = useState(false);
  const rawStatus = String(claim.decision || "UNVERIFIABLE").toUpperCase();
  const status = ["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"].includes(rawStatus) ? rawStatus.toLowerCase() : "unverifiable";
  const meta = STATUS_META[status];
  const StatusIcon = meta.icon;
  const evidence = claim.evidence_details && typeof claim.evidence_details === "object" ? claim.evidence_details : {};
  const confidence = Number(claim.evidence_confidence);
  const validConfidence = Number.isFinite(confidence) ? Math.max(0, Math.min(confidence, 1)) : null;
  const retrievalWarnings = Array.isArray(claim.retrieval_status?.warnings) ? claim.retrieval_status.warnings.filter((warning) => typeof warning === "string") : [];
  const evidenceSummary = claim.evidence_summary || {};
  const supporting = safeArray(evidence.supporting);
  const contradicting = safeArray(evidence.contradicting);
  const neutral = safeArray(evidence.neutral);
  const sourceCount = status === "contradicted" ? evidenceSummary.independent_contradicting_sources : evidenceSummary.independent_supporting_sources;

  return (
    <article className={`claim-card claim-${status}`}>
      <div className="claim-card-rail" aria-hidden="true" />
      <div className="claim-card-main">
        <div className="claim-card-top"><span className="claim-id">CLAIM / {String(claim.claim_id ?? index + 1).padStart(2, "0")}</span><span className={`classification-badge ${status}`}><StatusIcon size={14} />{meta.label}</span></div>
        <h3>{typeof claim.claim === "string" ? claim.claim : "Claim text unavailable"}</h3>
        {typeof claim.source_text === "string" && claim.source_text && claim.source_text !== claim.claim && <p className="claim-source-text"><span>FROM SUBMITTED TEXT</span>{claim.source_text}</p>}
        <p className="claim-reason">{claim.decision_reason || "The available evidence did not support a definitive classification."}</p>
        <div className="claim-signals">
          <div className="signal-evidence"><span>EVIDENCE</span><div><b className="supported-text">{supporting.length} supporting</b><b className="contradicted-text">{contradicting.length} challenging</b><b className="neutral-text">{neutral.length} neutral</b></div></div>
          <div className="signal-sources"><span>QUALIFYING DOMAINS</span><strong>{Number.isFinite(Number(sourceCount)) ? sourceCount : "—"}</strong></div>
          {validConfidence !== null && <div className="signal-score"><div><span>Evidence score</span><strong>{formatScore(validConfidence)}</strong></div><small>Heuristic · not a probability</small></div>}
        </div>
        {claim.retrieval_status?.state === "partial" && <div className="claim-retrieval-warning"><CircleAlert size={15} /><span><strong>Partial retrieval.</strong> {retrievalWarnings.length ? retrievalWarnings.join("; ") : "Some configured evidence providers were unavailable."}</span></div>}
        <details className="claim-details" open={expanded} onToggle={(event) => setExpanded(event.currentTarget.open)}>
          <summary><span>{expanded ? "Hide evidence and details" : "Review evidence and technical details"}</span><ArrowDownRight size={15} /></summary>
          <div className="claim-detail-body">
            <div className="evidence-explorer">
              <div className="evidence-explorer-heading"><div><span className="eyebrow">SOURCE MATERIAL</span><h4>Evidence</h4></div><span>{supporting.length + contradicting.length + neutral.length} passages</span></div>
              <div className="evidence-groups"><EvidenceGroup title="Supporting" items={evidence.supporting} tone="supporting" /><EvidenceGroup title="Contradicting" items={evidence.contradicting} tone="contradicting" />{neutral.length > 0 && <EvidenceGroup title="Neutral" items={evidence.neutral} tone="neutral" />}</div>
            </div>
            {Number(evidenceSummary.filtered_evidence_count) > 0 && <details className="excluded-evidence"><summary><Minus size={13} /> {evidenceSummary.filtered_evidence_count} passage(s) excluded from the decision</summary><div className="evidence-groups"><EvidenceGroup title="Excluded supporting" items={claim.excluded_evidence_details?.supporting} tone="supporting" /><EvidenceGroup title="Excluded contradicting" items={claim.excluded_evidence_details?.contradicting} tone="contradicting" /><EvidenceGroup title="Excluded neutral" items={claim.excluded_evidence_details?.neutral} tone="neutral" /></div></details>}
            <details className="technical-details">
              <summary>Technical verification details <ArrowDownRight size={14} /></summary>
              <div className="technical-detail-content">
                <div className="method-strip"><span className="detail-label">METHOD</span><span>MiniLM + lexical ranking</span><span>DeBERTa natural language inference</span><span>Source reliability heuristics</span></div>
                <div className="provenance-strip"><span className="detail-label">RETRIEVAL STATUS</span><span className={claim.retrieval_status?.wikipedia === "failed" ? "provider-failed" : "provider-ok"}>Wikipedia {claim.retrieval_status?.wikipedia === "failed" ? "unavailable" : claim.retrieval_status?.wikipedia === "available" ? "queried" : "status unavailable"}</span><span className={claim.retrieval_status?.tavily === "failed" ? "provider-failed" : "provider-ok"}>Tavily {claim.retrieval_status?.tavily === "available" ? "queried" : claim.retrieval_status?.tavily === "failed" ? "unavailable" : "not configured"}</span></div>
                {Number.isFinite(Number(sourceCount)) && <p className="technical-domain-note">Qualifying publisher domains: {sourceCount}. Publisher subdomains are grouped together.</p>}
                <p className="claim-caveat">The evidence score combines NLI confidence with source reliability. It is a heuristic, not a calibrated probability that the claim is correct.</p>
              </div>
            </details>
          </div>
        </details>
      </div>
    </article>
  );
}
