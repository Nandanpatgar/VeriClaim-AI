import EvidenceGroup from "./EvidenceGroup.jsx";

function safeArray(value) {
  return Array.isArray(value)
    ? value.filter((item) => item && typeof item === "object" && !Array.isArray(item))
    : [];
}

function formatScore(value) {
  const score = Number(value);
  return Number.isFinite(score) ? `${(score * 100).toFixed(1)}%` : "Not available";
}

export default function ClaimCard({ claim, index }) {
  const rawStatus = String(claim.decision || "UNVERIFIABLE").toUpperCase();
  const status = ["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"].includes(rawStatus)
    ? rawStatus.toLowerCase()
    : "unverifiable";
  const evidence = claim.evidence_details && typeof claim.evidence_details === "object"
    ? claim.evidence_details
    : {};
  const confidence = Number(claim.evidence_confidence);
  const retrievalWarnings = Array.isArray(claim.retrieval_status?.warnings)
    ? claim.retrieval_status.warnings.filter((warning) => typeof warning === "string")
    : [];
  const filteredCount = Number(claim.evidence_summary?.filtered_evidence_count) || 0;

  return (
    <article className="claim-card">
      <div className="claim-top">
        <span className="claim-number">CLAIM {claim.claim_id ?? index + 1}</span>
        <span className={`status ${status}`}>{status.toUpperCase()}</span>
      </div>
      <h3>{typeof claim.claim === "string" ? claim.claim : "Claim text unavailable"}</h3>
      {typeof claim.source_text === "string" && claim.source_text && <p className="source-original"><strong>From input:</strong> {claim.source_text}</p>}
      <div className="confidence"><span>Heuristic evidence score</span><strong>{formatScore(confidence)}</strong></div>
      <div className="progress" aria-label={`Heuristic evidence score ${formatScore(confidence)}`}>
        <div className={`progress-fill ${status}`} style={{ width: `${Number.isFinite(confidence) ? Math.max(0, Math.min(confidence * 100, 100)) : 0}%` }} />
      </div>
      <small className="score-note">Composite NLI and source score; not a calibrated probability of correctness.</small>
      {claim.decision_reason && <p className="reason">{claim.decision_reason}</p>}
      {claim.retrieval_status?.state === "partial" && <p className="retrieval-warning"><strong>Partial evidence retrieval.</strong> {retrievalWarnings.length ? retrievalWarnings.join("; ") : "Some providers were unavailable."}</p>}
      <div className="evidence">
        <EvidenceGroup title="Supporting evidence" items={evidence.supporting} />
        <EvidenceGroup title="Contradicting evidence" items={evidence.contradicting} />
        {safeArray(evidence.neutral).length > 0 && <EvidenceGroup title="Neutral evidence" items={evidence.neutral} />}
      </div>
      {filteredCount > 0 && <details className="excluded-evidence">
        <summary>{filteredCount} passage(s) excluded from the decision</summary>
        <div className="evidence">
          <EvidenceGroup title="Excluded supporting passages" items={claim.excluded_evidence_details?.supporting} />
          <EvidenceGroup title="Excluded contradicting passages" items={claim.excluded_evidence_details?.contradicting} />
          {safeArray(claim.excluded_evidence_details?.neutral).length > 0 && <EvidenceGroup title="Excluded neutral passages" items={claim.excluded_evidence_details.neutral} />}
        </div>
      </details>}
    </article>
  );
}
