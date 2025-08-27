// src/lib/modeStorage.ts
import type { RagMode } from "@/lib/sse";

const KEY_PREFIX = "chat-mode:";

function isRagMode(v: string | null): v is RagMode {
    return v === "auto" || v === "none" || v === "dense" || v === "rerank" || v === "web";
}

export function getChatMode(chatId?: string | null): RagMode {
    if (!chatId) return "none";
    try {
        const v = localStorage.getItem(KEY_PREFIX + chatId);
        return isRagMode(v) ? v : "none";
    } catch {
        return "none";
    }
}

export function setChatMode(chatId: string | null | undefined, mode: RagMode): void {
    if (!chatId) return;
    try {
        localStorage.setItem(KEY_PREFIX + chatId, mode);
        // tell listeners (e.g., Header / Chat) about the change
        window.dispatchEvent(new StorageEvent("storage", { key: KEY_PREFIX + chatId, newValue: mode }));
    } catch {
        /* ignore */
    }
}

export function clearChatMode(chatId?: number): void {
    if (!chatId) return;
    try {
        localStorage.removeItem(KEY_PREFIX + chatId);
        window.dispatchEvent(new StorageEvent("storage", { key: KEY_PREFIX + chatId, newValue: null as unknown as string }));
    } catch {
        /* ignore */
    }
}
