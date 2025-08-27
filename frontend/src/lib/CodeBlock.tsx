// src/components/CodeBlock.tsx
import { useState } from "react";
import { Button } from "@/components/ui/button";

export default function CodeBlock({ code, lang }: { code: string; lang?: string }) {
    const [copied, setCopied] = useState(false);

    const doCopy = async () => {
        try {
            await navigator.clipboard.writeText(code);
            setCopied(true);
            setTimeout(() => setCopied(false), 1200);
        } catch {
            /* ignore */
        }
    };

    return (
        <div className="relative my-3 overflow-hidden rounded-xl border">
            <div className="flex items-center justify-between px-3 py-1.5 text-xs border-b bg-[var(--panel)]">
                <span className="opacity-70">{lang ? lang : "code"}</span>
                <Button
                    variant="outline"
                    size="sm"
                    className="h-7 px-2 py-1"
                    onClick={doCopy}
                    aria-label="Copy code block"
                >
                    {copied ? "Copied!" : "Copy"}
                </Button>
            </div>

            {/* Inline styles here make the block immune to any transient CSS-order issues during HMR */}
            <pre
                className="overflow-x-auto p-3 text-sm leading-relaxed"
                data-lang={lang || ""}
                style={{ whiteSpace: "pre" }}
            >
        <code style={{ whiteSpace: "inherit", display: "block" }}>{code}</code>
      </pre>
        </div>
    );
}
