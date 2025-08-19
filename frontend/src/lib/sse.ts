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
    onTrace?: (traceId: string) => void; // NEW
};

export type RagMode = "auto" | "none" | "dense" | "rerank" | "web";

export function openChatSSE(
    prompt: string,
    handlers: SSEHandlers,
    opts?: { mode?: RagMode }
): EventSource {
    const base = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";
    const mode: RagMode = opts?.mode ?? "auto";
    const url = `${base}/chat/stream?q=${encodeURIComponent(prompt)}&mode=${mode}`;
    const es = new EventSource(url);

    es.addEventListener("token", (e: MessageEvent) => handlers.onToken(e.data));

    es.addEventListener("trace", (e: MessageEvent) => {
        try {
            const obj = JSON.parse(e.data);
            if (obj?.id) handlers.onTrace?.(obj.id);
        } catch { /* ignore */ }
    });

    es.addEventListener("sources", (e: MessageEvent) => {
        try {
            const arr = JSON.parse(e.data);
            handlers.onSources?.(arr);
        } catch { /* ignore */ }
    });

    es.addEventListener("tool", (e: MessageEvent) => {
        try {
            const obj = JSON.parse(e.data);
            handlers.onTool?.(obj);
        } catch { /* ignore */ }
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
