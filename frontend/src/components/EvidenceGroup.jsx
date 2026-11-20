function safeArray(value) {
  return Array.isArray(value)
    ? value.filter((item) => item && typeof item === "object" && !Array.isArray(item))
    : [];
}

function formatScore(value) {
  const score = Number(value);
  return Number.isFinite(score) ? `${(score * 100).toFixed(1)}%` : "Not available";
}

export default function EvidenceGroup({ title, items }) {
  const evidence = safeArray(items);

  return (
    <section className="evidence-group">
      <h4>{title} <span>{evidence.length}</span></h4>
      {evidence.length ? evidence.map((item, index) => (
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
