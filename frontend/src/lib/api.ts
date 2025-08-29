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
    // Detect FormData to avoid forcing JSON headers/encoding.
    const isFormData =
        typeof FormData !== "undefined" && opts.body instanceof FormData;

    const headers: Record<string, string> = {
        ...(opts.headers || {}),
    };
    if (!isFormData) {
        headers["Content-Type"] = headers["Content-Type"] ?? "application/json";
    }

    const res = await fetch(`${API}${path}`, {
        method: opts.method ?? "GET",
        credentials: "include",
        headers,
        body: isFormData
            ? (opts.body as unknown as FormData)
            : opts.body !== undefined
                ? JSON.stringify(opts.body)
                : undefined,
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

export type UploadDocsResponse = {
    ok: boolean;
    files_received?: number;
    files_indexed?: number;
    files_skipped?: string[];
    chunks?: number;
    scope?: "user" | "chat";
    chat_id?: string | null;
};

export async function uploadDocs(
    files: File[],
    scope: "user" | "chat",
    chatId?: number
): Promise<UploadDocsResponse> {
    const fd = new FormData();
    for (const f of files) fd.append("files", f);

    const qs = new URLSearchParams({ scope });
    if (scope === "chat" && typeof chatId === "number") {
        qs.set("chat_id", String(chatId));
    }

    return api<UploadDocsResponse>(`/rag/upload?${qs.toString()}`, {
        method: "POST",
        body: fd as unknown as FormData, // handled above by api()
    });
}

export type FileItem = {
    id: number;
    filename: string;
    mime?: string | null;
    size_bytes?: number | null;
    created_at: string;
};

export async function listFiles(
    scope: "user" | "chat",
    chatId?: string | number
): Promise<FileItem[]> {
    const qs = new URLSearchParams({ scope });
    if (scope === "chat" && chatId != null) qs.set("chat_id", String(chatId));
    return api<FileItem[]>(`/files?${qs.toString()}`);
}

export async function deleteFile(fileId: number): Promise<{ ok: boolean }> {
    return api<{ ok: boolean }>(`/files/${fileId}`, { method: "DELETE" });
}

export type StagedItem = {
    sha256_hex: string;
    filename: string;
    size_bytes: number;
    mime?: string | null;
};

export async function listStaged(draftId: string): Promise<StagedItem[]> {
    const qs = new URLSearchParams({ draft_id: draftId });
    return api<StagedItem[]>(`/files/stage?${qs.toString()}`);
}

export async function unstage(
    draftId: string,
    sha256_hex: string
): Promise<{ ok: boolean }> {
    const qs = new URLSearchParams({ draft_id: draftId });
    return api<{ ok: boolean }>(
        `/files/stage/${encodeURIComponent(sha256_hex)}?${qs.toString()}`,
        {
            method: "DELETE",
        }
    );
}

/**
 * Stage files to the backend with upload progress. Returns draft_id and list.
 * Note: we use XHR because fetch() has no upload progress events.
 */
export function stageUpload(
    files: File[],
    draftId?: string,
    onProgress?: (p: number) => void
): Promise<{ ok: boolean; draft_id: string; items: StagedItem[] }> {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        const API = "http://localhost:8000";
        const url = `${API}/files/stage`;

        xhr.open("POST", url, true);
        xhr.withCredentials = true;

        xhr.upload.onprogress = (e) => {
            if (!onProgress) return;
            if (e.lengthComputable) {
                const p = e.total > 0 ? e.loaded / e.total : 0;
                onProgress(Math.max(0, Math.min(1, p)));
            }
        };

        xhr.onerror = () => reject(new Error("Network error during upload"));
        xhr.onload = () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                try {
                    resolve(JSON.parse(xhr.responseText));
                } catch {
                    reject(new Error("Bad JSON from server"));
                }
            } else {
                try {
                    const j = JSON.parse(xhr.responseText);
                    reject(new Error(j?.detail || `HTTP ${xhr.status}`));
                } catch {
                    reject(new Error(`HTTP ${xhr.status}`));
                }
            }
        };

        const fd = new FormData();
        for (const f of files) fd.append("files", f);
        if (draftId) fd.append("draft_id", draftId);
        xhr.send(fd);
    });
}

export async function commitStaged(
    draftId: string,
    chatId: number
): Promise<{ ok: boolean; count: number }> {
    return api<{ ok: boolean; count: number }, { draft_id: string; chat_id: number }>(
        `/files/commit`,
        {
            method: "POST",
            body: { draft_id: draftId, chat_id: chatId },
        }
    );
}
