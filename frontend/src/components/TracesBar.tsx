import { useState } from "react";
import { Button } from "@/components/ui/button";

export default function TracesBar({ traceId }: { traceId: string | null }) {
    const [copied, setCopied] = useState(false);
    if (!traceId) return null;

    const short = traceId.slice(0, 8) + "…" + traceId.slice(-4);
    const copy = async () => {
        try {
            await navigator.clipboard.writeText(traceId);
            setCopied(true);
            setTimeout(() => setCopied(false), 1200);
        } catch { /* empty */ }
    };
    const host = import.meta.env.VITE_LANGFUSE_HOST || "https://cloud.langfuse.com";

    return (
        <div className="container pt-1">
            <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-panel dark:bg-neutral-900 px-3 py-2 flex items-center justify-between gap-3">
                <div className="text-[13px]"><b>Trace:</b> <code dir="ltr">{short}</code></div>
                <div className="flex gap-2">
                    <Button onClick={copy} variant="outline" className="h-9">{copied ? "✓ הועתק" : "העתק ID"}</Button>
                    <a href={host} target="_blank" rel="noopener noreferrer" className="no-underline">
                        <Button variant="outline" className="h-9">Open Langfuse</Button>
                    </a>
                </div>
            </div>
        </div>
    );
}
