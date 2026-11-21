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
  const [expanded, setExpanded] = useState(index === 0);
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
      <div className="claim-card-rail" aria-hidden="true"><span /></div>
      <div className="claim-card-main">
        <div className="claim-card-top"><span className="claim-id">CLAIM / {String(claim.claim_id ?? index + 1).padStart(2, "0")}</span><span className={`classification-badge ${status}`}><StatusIcon size={14} />{meta.label}</span></div>
        <p className="claim-eyebrow">{meta.eyebrow}</p>
        <h3>{typeof claim.claim === "string" ? claim.claim : "Claim text unavailable"}</h3>
        {typeof claim.source_text === "string" && claim.source_text && claim.source_text !== claim.claim && <p className="claim-source-text"><span>FROM SUBMITTED TEXT</span>{claim.source_text}</p>}
        <p className="claim-reason">{claim.decision_reason || "The available evidence did not support a definitive classification."}</p>
        <div className="claim-signals">
          {validConfidence !== null && <div className="signal-score"><div><span>Evidence score</span><strong>{formatScore(validConfidence)}</strong></div><div className="signal-track"><span className={status} style={{ width: `${validConfidence * 100}%` }} /></div><small>Composite heuristic · not calibrated</small></div>}
          <div className="signal-sources"><span>INDEPENDENT DOMAINS</span><strong>{Number.isFinite(Number(sourceCount)) ? sourceCount : "—"}</strong></div>
          <div className="signal-evidence"><span>EVIDENCE PASSES</span><div><b className="supported-text">{supporting.length} <i>+</i></b><b className="contradicted-text">{contradicting.length} <i>−</i></b><b className="neutral-text">{neutral.length} <i>~</i></b></div></div>
        </div>
        {claim.retrieval_status?.state === "partial" && <div className="claim-retrieval-warning"><CircleAlert size={15} /><span><strong>Partial retrieval.</strong> {retrievalWarnings.length ? retrievalWarnings.join("; ") : "Some configured evidence providers were unavailable."}</span></div>}
        <details className="claim-details" open={expanded} onToggle={(event) => setExpanded(event.currentTarget.open)}>
          <summary><span>{expanded ? "Hide verification detail" : "Explore evidence and method"}</span><ArrowDownRight size={15} /></summary>
          <div className="claim-detail-body">
            <div className="method-strip"><span className="detail-label">METHODS APPLIED</span><span>Semantic + term ranking</span><span>Natural language inference</span><span>Source reliability heuristics</span></div>
            <div className="provenance-strip"><span className="detail-label">RETRIEVAL</span><span className={claim.retrieval_status?.wikipedia === "failed" ? "provider-failed" : "provider-ok"}>Wikipedia {claim.retrieval_status?.wikipedia === "failed" ? "unavailable" : "checked"}</span><span className={claim.retrieval_status?.tavily === "failed" ? "provider-failed" : "provider-ok"}>Tavily {claim.retrieval_status?.tavily === "available" ? "checked" : claim.retrieval_status?.tavily === "failed" ? "unavailable" : "not configured"}</span></div>
            <div className="evidence-explorer">
              <div className="evidence-explorer-heading"><div><span className="eyebrow">THE RECEIPTS</span><h4>Evidence explorer</h4></div><span>{safeArray(evidence.supporting).length + safeArray(evidence.contradicting).length + safeArray(evidence.neutral).length} passages</span></div>
              <div className="evidence-groups"><EvidenceGroup title="Supporting" items={evidence.supporting} tone="supporting" /><EvidenceGroup title="Contradicting" items={evidence.contradicting} tone="contradicting" />{neutral.length > 0 && <EvidenceGroup title="Neutral" items={evidence.neutral} tone="neutral" />}</div>
            </div>
            {Number(evidenceSummary.filtered_evidence_count) > 0 && <details className="excluded-evidence"><summary><Minus size={13} /> {evidenceSummary.filtered_evidence_count} passage(s) excluded from the decision</summary><div className="evidence-groups"><EvidenceGroup title="Excluded supporting" items={claim.excluded_evidence_details?.supporting} tone="supporting" /><EvidenceGroup title="Excluded contradicting" items={claim.excluded_evidence_details?.contradicting} tone="contradicting" /><EvidenceGroup title="Excluded neutral" items={claim.excluded_evidence_details?.neutral} tone="neutral" /></div></details>}
            <p className="claim-caveat">NLI and source scores are model or heuristic outputs. They do not establish certainty; inspect the linked original sources.</p>
          </div>
        </details>
      </div>
    </article>
  );
}
