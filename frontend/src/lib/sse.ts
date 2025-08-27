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
    tool: string;
    name: string;
    args?: Record<string, unknown>;
    result?: unknown;
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

type OpenOpts = {
    mode?: RagMode;
    cid?: string;                // guest/ephemeral id
    chatId?: number;             // persisted chat id
    scope?: "user" | "chat";     // retrieval namespacing (defaults in backend: user)
};

export function openChatSSE(
    prompt: string,
    handlers: SSEHandlers,
    opts: OpenOpts = {}
): EventSource {
    const base = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";
    const mode: RagMode = opts.mode ?? "auto";
    const params = new URLSearchParams({
        q: prompt,
        mode,
    });

    if (opts.cid) params.set("cid", opts.cid);
    if (typeof opts.chatId === "number") params.set("chat_id", String(opts.chatId));
    if (opts.scope) params.set("scope", opts.scope);

    const url = `${base}/chat/stream?${params.toString()}`;

    // IMPORTANT: include cookies with SSE across origins
    const es = new EventSource(url, { withCredentials: true });

    es.addEventListener("token", (e: MessageEvent) => handlers.onToken(e.data));
    es.addEventListener("trace", (e: MessageEvent) => {
        try {
            const obj = JSON.parse(e.data);
            if (obj?.id) handlers.onTrace?.(obj.id);
        } catch { /* ignore */ }
    });
    es.addEventListener("sources", (e: MessageEvent) => {
        try { handlers.onSources?.(JSON.parse(e.data)); } catch { /* ignore */ }
    });
    es.addEventListener("tool", (e: MessageEvent) => {
        try { handlers.onTool?.(JSON.parse(e.data)); } catch { /* ignore */ }
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
