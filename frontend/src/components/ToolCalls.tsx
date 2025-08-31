// src/components/ToolCalls.tsx
import * as React from "react";
import type { ToolEvent } from "@/lib/sse";
import { Card, CardContent, CardHeader } from "@/components/ui/card";

function safeStringify(v: unknown): string {
    try {
        return typeof v === "string" ? v : JSON.stringify(v ?? null, null, 2);
    } catch {
        return String(v);
    }
}

export default function ToolCalls({ items }: { items: ToolEvent[] }) {
    const [enabled, setEnabled] = React.useState<boolean>(() => {
        try {
            return (localStorage.getItem("show-reasoning-steps") ?? "1") === "1";
        } catch {
            return true;
        }
    });

    React.useEffect(() => {
        const onStorage = (e: StorageEvent) => {
            if (!e.key) return;
            if (e.key === "show-reasoning-steps") {
                setEnabled((e.newValue ?? "1") === "1");
            }
        };
        window.addEventListener("storage", onStorage);
        return () => window.removeEventListener("storage", onStorage);
    }, []);

    if (!enabled || !Array.isArray(items) || items.length === 0) return null;

    // IMPORTANT: do NOT use the "assistant" bubble class here, because it forces margin-left:auto (right anchoring).
    // We center this panel within the chat column so it never collides with the sidebar and is fully visible.
    return (
        <div className="msg w-full max-w-[80ch] self-center mx-auto px-2 sm:px-0">
            <div className="flex items-center justify-between mb-2">
                <div className="text-[13px] opacity-85">צעדי חשיבה</div>
                <button
                    type="button"
                    className="text-[12px] opacity-75 hover:opacity-100 underline"
                    onClick={() => {
                        try {
                            localStorage.setItem("show-reasoning-steps", "0");
                            window.dispatchEvent(new StorageEvent("storage", { key: "show-reasoning-steps", newValue: "0" }));
                        } catch {
                            /* ignore */
                        }
                    }}
                >
                    הסתר
                </button>
            </div>

            <div className="grid gap-2">
                {items.map((ev, idx) => {
                    const key = `${String(ev.name ?? ev["tool"] ?? "tool")}-${idx}`;
                    const titleLeft = String(ev["tool"] ?? "tool");
                    const titleRight = ev.name ? `· ${String(ev.name)}` : "";
                    const argsText = safeStringify(ev.args);
                    const resultText = safeStringify(ev.result);

                    return (
                        <Card key={key}>
                            <CardHeader className="text-sm font-semibold">
                                {titleLeft} {titleRight}
                            </CardHeader>
                            <CardContent>
                                <pre className="whitespace-pre-wrap text-xs leading-relaxed">{argsText}</pre>
                                {ev.result !== undefined && (
                                    <div className="mt-2">
                                        <div className="text-xs opacity-70 mb-1">תוצאה</div>
                                        <pre className="whitespace-pre-wrap text-xs leading-relaxed">{resultText}</pre>
                                    </div>
                                )}
                            </CardContent>
                        </Card>
                    );
                })}
            </div>
        </div>
    );
}
