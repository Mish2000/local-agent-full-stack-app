// src/lib/api.ts
const API = "http://localhost:8000";

export type ApiOptions<B = unknown> = {
    method?: "GET" | "POST" | "PUT" | "DELETE";
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
        } catch { /* ignore */ }
        throw new Error(msg);
    }

    const text = await res.text();
    return (text ? JSON.parse(text) : null) as T;
}

