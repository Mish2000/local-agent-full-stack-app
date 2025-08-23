import type { ToolEvent } from "@/lib/sse";
import { Card, CardContent, CardHeader } from "@/components/ui/card";

export default function ToolCalls({ items }: { items: ToolEvent[] }) {
    if (!items || items.length === 0) return null;
    return (
        <div className="container pb-3">
            <details open className="rounded-2xl border border-neutral-200 dark:border-neutral-800">
                <summary className="cursor-pointer font-semibold px-3 py-2">Tool calls ({items.length})</summary>
                <div className="grid gap-2 mt-2">
                    {items.map((t, i) => (
                        <Card key={i} className="bg-white dark:bg-neutral-900">
                            <CardHeader className="text-[13px]">
                                <b>name:</b> {t.name}
                            </CardHeader>
                            <CardContent className="text-[13px]">
                                {t.args && <pre className="whitespace-pre-wrap">{JSON.stringify(t.args, null, 2)}</pre>}
                                {t.error && <div className="text-red-500 mt-2">Error: {t.error}</div>}
                                {t.result && <pre className="whitespace-pre-wrap mt-2">{JSON.stringify(t.result, null, 2)}</pre>}
                            </CardContent>
                        </Card>
                    ))}
                </div>
            </details>
        </div>
    );
}
