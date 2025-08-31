// src/lib/modes.ts
import * as React from "react";
import type { RagMode } from "@/lib/sse";

export const DEFAULT_MODE: RagMode = "offline";

/**
 * Returns the localStorage key for a given chatId.
 * For new chats (no chatId yet), we do NOT use any global/default key —
 * this guarantees the UI always defaults to "offline" on refresh/new chat.
 */
function key(chatId?: number | string): string | null {
    if (chatId === undefined || chatId === null) return null;
    return `agent-mode:${chatId}`;
}

function isValid(m?: string | null): m is RagMode {
    return m === "offline" || m === "web" || m === "auto";
}

/**
 * Per-chat mode hook.
 * - When chatId is undefined (new conversation page), we always expose DEFAULT_MODE.
 * - When chatId is defined, we load from per-chat storage and persist updates there.
 */
export function useChatMode(chatId?: number) {
    const [mode, setModeState] = React.useState<RagMode>(DEFAULT_MODE);

    React.useEffect(() => {
        const k = key(chatId);
        if (!k) {
            // No chat yet -> always default to "offline"
            setModeState(DEFAULT_MODE);
            return;
        }
        let m: string | null = null;
        try {
            m = localStorage.getItem(k);
        } catch {
            /* ignore */
        }
        setModeState(isValid(m) ? (m as RagMode) : DEFAULT_MODE);
    }, [chatId]);

    const setMode = React.useCallback(
        (m: RagMode) => {
            setModeState(m);
            const k = key(chatId);
            if (!k) return; // no-op until we actually have a chat id
            try {
                localStorage.setItem(k, m);
            } catch {
                /* ignore */
            }
        },
        [chatId]
    );

    return { mode, setMode };
}
