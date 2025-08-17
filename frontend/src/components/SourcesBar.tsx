import type { Source } from "../lib/sse";

function baseName(p: string) {
    if (!p) return "unknown";
    const parts = p.replaceAll("\\", "/").split("/");
    return parts[parts.length - 1] || p;
}

export default function SourcesBar({ items }: { items: Source[] }) {
    if (!items || items.length === 0) return null;
    return (
        <div className="container" style={{ padding: "6px 16px 16px 16px" }}>
            <div style={{ fontSize: 13, opacity: 0.85, marginBottom: 6, textAlign: "end" }}>מקורות</div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "flex-end" }}>
                {items.map((s) => {
                    const range =
                        s.start_line && s.end_line ? ` · lines ${s.start_line}-${s.end_line}` : "";
                    return (
                        <div
                            key={s.id}
                            title={s.preview}
                            style={{
                                border: "1px solid var(--border)",
                                background: "var(--panel)",
                                color: "var(--text-strong)",
                                borderRadius: 999,
                                padding: "6px 10px",
                                fontSize: 13,
                                maxWidth: 360,
                                textOverflow: "ellipsis",
                                overflow: "hidden",
                                whiteSpace: "nowrap",
                            }}
                        >
                            [{s.id}] {baseName(s.source)}
                            {range}
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
