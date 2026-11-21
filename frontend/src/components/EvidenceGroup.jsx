import { ArrowUpRight, CalendarDays, ExternalLink, ShieldCheck } from "lucide-react";

function safeArray(value) {
  return Array.isArray(value) ? value.filter((item) => item && typeof item === "object" && !Array.isArray(item)) : [];
}

function formatScore(value) {
  const score = Number(value);
  return Number.isFinite(score) ? `${(score * 100).toFixed(0)}%` : "—";
}

function safeHttpUrl(value) {
  if (typeof value !== "string") return "";
  try {
    const url = new URL(value);
    return ["http:", "https:"].includes(url.protocol) ? url.href : "";
  } catch {
    return "";
  }
}

function hostname(url) {
  try { return new URL(url).hostname.replace(/^www\./, ""); } catch { return ""; }
}

export default function EvidenceGroup({ title, items, tone = "neutral" }) {
  const evidence = safeArray(items);
  const toneLabel = tone === "supporting" ? "SUPPORTS" : tone === "contradicting" ? "CHALLENGES" : "CONTEXT";
  return (
    <section className={`evidence-group evidence-${tone}`}>
      <div className="evidence-group-title"><span className="evidence-tone-mark" /><h5>{title}</h5><span className="evidence-count">{evidence.length}</span></div>
      {evidence.length ? evidence.map((item, index) => {
        const url = safeHttpUrl(item.url);
        const domain = item.domain || hostname(url);
        const date = item.published_date || item.published_at || item.date;
        return <article className="evidence-item" key={item.evidence_id ?? `${tone}-${index}`}>
          <div className="evidence-source-line"><span className="source-icon"><ShieldCheck size={14} /></span><div className="evidence-source-name"><strong>{item.source || item.title || domain || "Evidence source"}</strong>{domain && <small>{domain}</small>}</div><span className={`evidence-relation ${tone}`}>{toneLabel}</span></div>
          {item.title && item.source && <p className="source-title">{item.title}</p>}
          {typeof item.evidence_text === "string" && <blockquote>{item.evidence_text}</blockquote>}
          <div className="evidence-meta-row">
            {item.source_type && <span>{item.source_type}</span>}
            {date && <span><CalendarDays size={12} />{String(date)}</span>}
            {item.reliability_score != null && <span><ShieldCheck size={12} />Source {formatScore(item.reliability_score)}</span>}
            {item.similarity_score != null && <span>Relevance {formatScore(item.similarity_score)}</span>}
            {item.semantic_similarity_score != null && <span>Meaning {formatScore(item.semantic_similarity_score)}</span>}
            {item.lexical_overlap_score != null && <span>Term match {formatScore(item.lexical_overlap_score)}</span>}
            {item.nli_confidence != null && <span>NLI {formatScore(item.nli_confidence)}</span>}
          </div>
          {item.nli_label && <div className="nli-result"><span>Passage relationship</span><strong>{String(item.nli_label).toUpperCase()}</strong>{item.nli_probabilities && <small>{Object.entries(item.nli_probabilities).map(([label, score]) => `${label} ${formatScore(score)}`).join(" · ")}</small>}</div>}
          {url ? <a className="source-link" href={url} target="_blank" rel="noopener noreferrer"><span>Open original source</span><ExternalLink size={13} /><ArrowUpRight size={12} /></a> : <span className="source-unavailable">No source link supplied</span>}
        </article>;
      }) : <p className="no-evidence">No {title.toLowerCase()} passages returned.</p>}
    </section>
  );
}
