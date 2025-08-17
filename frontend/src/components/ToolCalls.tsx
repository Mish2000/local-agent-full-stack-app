import type { ToolEvent } from "../lib/sse";

export default function ToolCalls({ items }: { items: ToolEvent[] }) {
    if (!items || items.length === 0) return null;
    return (
        <div className="container" style={{ padding: "0 16px 12px 16px" }}>
            <details style={{ background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 12, padding: "10px 12px" }} open>
                <summary style={{ cursor: "pointer", fontWeight: 600 }}>
                    Tool calls ({items.length})
                </summary>
                <div style={{ display: "grid", gap: 8, marginTop: 8, direction: "ltr" }}>
                    {items.map((t, i) => (
                        <div key={i} style={{ border: "1px solid var(--border)", borderRadius: 10, padding: "8px 10px", background: "var(--bg)" }}>
                            <div style={{ fontSize: 13, opacity: 0.9 }}>
                                <b>name:</b> {t.name}
                            </div>
                            {t.args && (
                                <pre style={{ marginTop: 6, whiteSpace: "pre-wrap" }}>
{JSON.stringify(t.args, null, 2)}
                                </pre>
                            )}
                            {t.error && (
                                <div style={{ color: "#ef4444", marginTop: 6 }}>
                                    Error: {t.error}
                                </div>
                            )}
                            {t.result && (
                                <pre style={{ marginTop: 6, whiteSpace: "pre-wrap" }}>
{JSON.stringify(t.result, null, 2)}
                                </pre>
                            )}
                        </div>
                    ))}
                </div>
            </details>
        </div>
    );
}
