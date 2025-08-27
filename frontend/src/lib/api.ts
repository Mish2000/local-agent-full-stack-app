// src/lib/api.ts
const API = "http://localhost:8000";

export type ApiOptions<B = unknown> = {
    method?: "GET" | "POST" | "PUT" | "DELETE" | "PATCH";
    body?: B;
    headers?: Record<string, string>;
};

export async function api<T, B = unknown>(
    path: string,
    opts: ApiOptions<B> = {} as ApiOptions<B>
): Promise<T> {
    const res = await fetch(`${API}${path}`, {
        method: opts.method ?? "GET",
        credentials: "include",
        headers: {
            "Content-Type": "application/json",
            ...(opts.headers || {}),
        },
        body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
    });

    if (!res.ok) {
        let msg = `HTTP ${res.status}`;
        try {
            const j = await res.json();
            if (j?.detail) msg = j.detail;
            if (j?.error) msg = j.error;
            if (typeof j === "string") msg = j;
        } catch {
            /* ignore */
        }
        throw new Error(msg);
    }

    const text = await res.text();
    return (text ? JSON.parse(text) : null) as T;
}

// Upload docs for RAG: accepts multiple files; scope can be "user" or "chat"
export async function uploadDocs(
    files: File[],
    scope: "user" | "chat" = "user",
    chatId?: string | number
): Promise<{ ok: boolean; chunks: number; files_indexed: number; files_skipped: string[] }> {
    const fd = new FormData();
    for (const f of files) fd.append("files", f);
    // send scope/chat_id in the body so Postman & browsers are equally happy
    fd.append("scope", scope);
    if (scope === "chat" && chatId != null) fd.append("chat_id", String(chatId));

    const res = await fetch(`${API}/rag/upload`, {
        method: "POST",
        credentials: "include",
        body: fd,
    });
    if (!res.ok) {
        let msg = `Upload failed (${res.status})`;
        try {
            const j = await res.json();
            if (j?.detail) msg = j.detail;
        } catch { /* ignore */ }
        throw new Error(msg);
    }
    return res.json();
}



