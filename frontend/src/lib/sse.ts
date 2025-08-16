export type SSEHandlers = {
    onToken: (t: string) => void;
    onDone: () => void;
    onError: (msg: string) => void;
};

export function openChatSSE(query: string, handlers: SSEHandlers): EventSource {
    const url = `http://localhost:8000/chat/stream?q=${encodeURIComponent(query)}`;
    const es = new EventSource(url);

    es.addEventListener("token", (e) => {
        handlers.onToken((e as MessageEvent).data ?? "");
    });

    es.addEventListener("done", () => {
        handlers.onDone();
        es.close();
    });

    es.addEventListener("error", (e) => {
        try {
            const data = (e as MessageEvent).data ?? "SSE error";
            handlers.onError(String(data));
        } catch {
            handlers.onError("SSE connection error");
        }
    });

    return es;
}

