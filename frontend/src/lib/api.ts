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
        body: fd as unknown as FormData,
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
 * Stage files with progress using XHR (fetch lacks upload progress).
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

export type AgentMode = "offline" | "web" | "auto";

export interface ModesItem {
    id: AgentMode;
    label: string;
    desc: string;
}

export type ProfileSettings = {
    instruction_enabled: boolean;
    instruction_text: string;
    avatar_kind?: "" | "system" | "upload";
    avatar_value?: string;
    display_name?: string;
};

export type AccountUpdate = {
    display_name?: string;
    current_password?: string;
    new_password?: string;
};

export async function apiUpdateAccount(p: AccountUpdate): Promise<{ ok: boolean }> {
    const r = await fetch(`${BASE}/profile/account`, {
        method: "PUT",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(p),
    });
    if (!r.ok) throw new Error(`PUT /profile/account ${r.status}`);
    return r.json();
}

const BASE = import.meta.env.VITE_BACKEND_ORIGIN ?? "http://localhost:8000";

export async function apiGetModes(): Promise<ModesItem[]> {
    const r = await fetch(`${BASE}/modes`, { credentials: "include" });
    if (!r.ok) throw new Error(`GET /modes ${r.status}`);
    return r.json();
}

export async function apiGetProfileSettings(): Promise<ProfileSettings> {
    const r = await fetch(`${BASE}/profile/settings`, { credentials: "include" });
    if (!r.ok) throw new Error(`GET /profile/settings ${r.status}`);
    return r.json();
}

/** Backend returns { ok: true } for PUT. */
export async function apiPutProfileSettings(
    p: ProfileSettings
): Promise<{ ok: boolean }> {
    const r = await fetch(`${BASE}/profile/settings`, {
        method: "PUT",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(p),
    });
    if (!r.ok) throw new Error(`PUT /profile/settings ${r.status}`);
    return r.json();
}

export async function apiUploadAvatar(file: File): Promise<{ ok: boolean }> {
    const fd = new FormData();
    fd.append("file", file);
    return api<{ ok: boolean }>("/profile/avatar/upload", {
        method: "POST",
        body: fd,
    });
}

/** Helper for consistent avatar URL + cache-busting when needed. */
export function avatarUrl(cacheBust = true): string {
    const u = `${BASE}/profile/avatar`;
    return cacheBust ? `${u}?t=${Date.now()}` : u;
}
