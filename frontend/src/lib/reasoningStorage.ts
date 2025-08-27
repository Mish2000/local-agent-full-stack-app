// src/lib/reasoningStorage.ts
import type { Source, ToolEvent } from "@/lib/sse";

const KEY_PREFIX = "chat-last:";

type Stored = {
    tools: ToolEvent[];
    sources: Source[];
    traceId: string | null;
    ts: number;
};

export function saveLastForChat(
    chatId: number,
    data: { tools?: ToolEvent[]; sources?: Source[]; traceId?: string | null }
) {
    try {
        const payload: Stored = {
            tools: Array.isArray(data.tools) ? data.tools : [],
            sources: Array.isArray(data.sources) ? data.sources : [],
            traceId: data.traceId ?? null,
            ts: Date.now(),
        };
        sessionStorage.setItem(KEY_PREFIX + String(chatId), JSON.stringify(payload));
    } catch {
        /* ignore */
    }
}

export function loadLastForChat(
    chatId: number
): { tools: ToolEvent[]; sources: Source[]; traceId: string | null } | null {
    try {
        const raw = sessionStorage.getItem(KEY_PREFIX + String(chatId));
        if (!raw) return null;
        const obj = JSON.parse(raw) as Partial<Stored> | null;
        if (!obj || !Array.isArray(obj.tools) || !Array.isArray(obj.sources)) return null;
        return {
            tools: obj.tools as ToolEvent[],
            sources: obj.sources as Source[],
            traceId: (obj.traceId ?? null) as string | null,
        };
    } catch {
        return null;
    }
}

export function clearLastForChat(chatId: number) {
    try {
        sessionStorage.removeItem(KEY_PREFIX + String(chatId));
    } catch {
        /* ignore */
    }
}
