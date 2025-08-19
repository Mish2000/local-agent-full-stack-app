import { useState } from "react";

export default function TracesBar({ traceId }: { traceId: string | null }) {
    if (!traceId) return null;

    const short = traceId.slice(0, 8) + "…" + traceId.slice(-4);
    const [copied, setCopied] = useState(false);

    const copy = async () => {
        try {
            await navigator.clipboard.writeText(traceId);
            setCopied(true);
            setTimeout(() => setCopied(false), 1200);
        } catch { /* ignore */ }
    };

    // If you set VITE_LANGFUSE_HOST, we show a generic link to Langfuse; without project slug it’s mainly a convenience.
    const host = import.meta.env.VITE_LANGFUSE_HOST || "https://cloud.langfuse.com";

    return (
        <div className="container" style={{ padding: "6px 16px 0 16px" }}>
            <div style={{
                background: "var(--panel)",
                border: "1px solid var(--border)",
                borderRadius: 12,
                padding: "8px 12px",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                gap: 12
            }}>
                <div style={{ fontSize: 13 }}>
                    <b>Trace:</b> <code dir="ltr">{short}</code>
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                    <button onClick={copy} className="button" style={{ height: 36, padding: "6px 10px" }}>
                        {copied ? "✓ Copied" : "Copy ID"}
                    </button>
                    <a href={host} target="_blank" rel="noopener noreferrer" className="button" style={{ height: 36, padding: "6px 10px", textDecoration: "none" }}>
                        Open Langfuse
                    </a>
                </div>
            </div>
        </div>
    );
}
