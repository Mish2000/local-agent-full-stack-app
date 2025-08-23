// src/lib/sse.ts
export type Source = {
    id: number;
    source: string;
    preview: string;
    score: number;
    start_line?: number;
    end_line?: number;
    url?: string;
};

export type ToolEvent = {
    name: string;
    args?: Record<string, never>;
    result?: never;
    error?: string;
};

export type SSEHandlers = {
    onToken: (t: string) => void;
    onDone: () => void;
    onError: (msg: string) => void;
    onSources?: (arr: Source[]) => void;
    onTool?: (ev: ToolEvent) => void;
    onTrace?: (traceId: string) => void;
};

export type RagMode = "auto" | "none" | "dense" | "rerank" | "web";

export function openChatSSE(
    prompt: string,
    handlers: SSEHandlers,
    opts?: { mode?: RagMode; cid?: string } // <<< NEW
): EventSource {
    const base = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";
    const mode: RagMode = opts?.mode ?? "auto";
    const cid = opts?.cid ?? "default"; // <<< NEW
    const url = `${base}/chat/stream?q=${encodeURIComponent(prompt)}&mode=${mode}&cid=${encodeURIComponent(cid)}`;
    const es = new EventSource(url);

    es.addEventListener("token", (e: MessageEvent) => handlers.onToken(e.data));
    es.addEventListener("trace", (e: MessageEvent) => {
        try {
            const obj = JSON.parse(e.data);
            if (obj?.id) handlers.onTrace?.(obj.id);
        } catch { /* empty */ }
    });
    es.addEventListener("sources", (e: MessageEvent) => {
        try { handlers.onSources?.(JSON.parse(e.data)); } catch { /* empty */ }
    });
    es.addEventListener("tool", (e: MessageEvent) => {
        try { handlers.onTool?.(JSON.parse(e.data)); } catch { /* empty */ }
    });
    es.addEventListener("done", () => {
        handlers.onDone();
        es.close();
    });
    es.onerror = () => {
        handlers.onError("SSE connection error");
        es.close();
    };
    return es;
}
