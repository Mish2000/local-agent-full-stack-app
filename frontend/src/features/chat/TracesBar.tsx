// src/components/TracesBar.tsx
import * as React from "react";
import { Button } from "@/components/ui/button.tsx";
import type {JSX} from "react";

const STORAGE_KEY = "show-reasoning-steps";

type Props = { traceId: string | null };

export default function TracesBar({ traceId }: Props): JSX.Element | null {
    // Respect the global “steps” visibility flag (same as ToolCalls)
    const [visible, setVisible] = React.useState<boolean>(() => {
        try {
            return (localStorage.getItem(STORAGE_KEY) ?? "1") === "1";
        } catch {
            return true;
        }
    });

    React.useEffect(() => {
        const onStorage = (e: StorageEvent) => {
            if (e.key === STORAGE_KEY) {
                setVisible((e.newValue ?? "1") === "1");
            }
        };
        window.addEventListener("storage", onStorage);
        return () => window.removeEventListener("storage", onStorage);
    }, []);

    const [copied, setCopied] = React.useState(false);
    if (!traceId || !visible) return null;

    const short = `${traceId.slice(0, 8)}…${traceId.slice(-4)}`;
    const copy = async () => {
        try {
            await navigator.clipboard.writeText(traceId);
            setCopied(true);
            window.setTimeout(() => setCopied(false), 1200);
        } catch {
            /* ignore */
        }
    };

    // Allow overriding host from .env; sensible default to Langfuse Cloud
    const host = (import.meta as any).env?.VITE_LANGFUSE_HOST || "https://cloud.langfuse.com";

    return (
        <div className="container pt-1">
            <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-panel dark:bg-neutral-900 px-3 py-2 flex items-center justify-between gap-3">
                <div className="text-[13px]">
                    <b>Trace:</b> <code dir="ltr">{short}</code>
                </div>
                <div className="flex gap-2">
                    <Button onClick={copy} variant="outline" className="h-9">
                        {copied ? "✓ הועתק" : "העתק ID"}
                    </Button>
                    <a href={host} target="_blank" rel="noopener noreferrer" className="no-underline">
                        <Button variant="outline" className="h-9">Open Langfuse</Button>
                    </a>
                </div>
            </div>
        </div>
    );
}
