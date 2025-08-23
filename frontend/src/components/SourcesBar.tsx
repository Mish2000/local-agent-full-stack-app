import type { Source } from "@/lib/sse";

function baseName(p: string) {
    if (!p) return "unknown";
    const parts = p.replaceAll("\\", "/").split("/");
    return parts[parts.length - 1] || p;
}
function hostFromUrl(u?: string) {
    if (!u) return "";
    try { return new URL(u).host || ""; } catch { return ""; }
}

export default function SourcesBar({ items }: { items: Source[] }) {
    if (!items || items.length === 0) return null;
    return (
        <div className="container pt-1 pb-4">
            <div className="text-[13px] opacity-85 mb-2 text-right">מקורות</div>
            <div className="flex flex-wrap gap-2 justify-end">
                {items.map((s) => {
                    const range = s.start_line && s.end_line ? ` · lines ${s.start_line}-${s.end_line}` : "";
                    const label = hostFromUrl(s.url) || baseName(s.source);
                    const cls =
                        "max-w-[360px] truncate rounded-full border border-neutral-200 dark:border-neutral-800 bg-panel dark:bg-neutral-900 px-3 py-1.5 text-[13px] text-neutral-900 dark:text-neutral-50";
                    return s.url ? (
                        <a key={s.id} href={s.url} target="_blank" rel="noopener noreferrer" title={s.preview} className={cls}>
                            [{s.id}] {label}{range}
                        </a>
                    ) : (
                        <div key={s.id} title={s.preview} className={cls}>
                            [{s.id}] {label}{range}
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
