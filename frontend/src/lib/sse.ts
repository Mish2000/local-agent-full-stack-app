// src/lib/sse.ts
export type RagMode = "offline" | "web" | "auto";

export type Source = {
    id: number;
    source: string;
    preview?: string;
    start_line?: number;
    end_line?: number;
    score?: number;
    url?: string;
};

export type ToolEvent = {
    name: string;
    args?: unknown;
    result?: unknown;
    debug?: unknown;
};

type Handlers = {
    onToken: (t: string) => void;
    onSources: (s: Source[]) => void;
    onTool: (e: ToolEvent) => void;
    onTrace: (id: string) => void;
    onDone: () => void;
    onError: (msg: string) => void;
};

type OpenOpts = {
    mode: RagMode;
    cid?: string;
    chatId?: number;
    scope: "user" | "chat";
};

const API_BASE = (import.meta as any).env?.VITE_API_BASE ?? "http://localhost:8000";

export function openChatSSE(q: string, h: Handlers, opts: OpenOpts) {
    const url = new URL(`${API_BASE}/chat/stream`);
    url.searchParams.set("q", q);
    url.searchParams.set("mode", opts.mode);
    url.searchParams.set("scope", opts.scope);
    if (opts.chatId) url.searchParams.set("chat_id", String(opts.chatId));
    if (opts.cid) url.searchParams.set("cid", opts.cid);

    const es = new EventSource(url.toString(), { withCredentials: true });

    es.addEventListener("token", (e: MessageEvent) => h.onToken((e as MessageEvent).data ?? ""));
    es.addEventListener("sources", (e: MessageEvent) => {
        try { h.onSources(JSON.parse((e as MessageEvent).data ?? "[]")); } catch { h.onSources([]); }
    });
    es.addEventListener("tool", (e: MessageEvent) => {
        try { h.onTool(JSON.parse((e as MessageEvent).data ?? "{}")); } catch { /* ignore */ }
    });
    es.addEventListener("trace", (e: MessageEvent) => h.onTrace((e as MessageEvent).data ?? ""));
    es.addEventListener("error", (e: MessageEvent) => h.onError((e as MessageEvent).data ?? "stream error"));
    es.addEventListener("done", () => {
        es.close();
        h.onDone();
    });

    return es;
}
