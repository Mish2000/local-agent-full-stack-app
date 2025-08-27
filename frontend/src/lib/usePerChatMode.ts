// src/lib/usePerChatMode.ts
import * as React from "react";
import type { RagMode } from "@/lib/sse";
import { getChatMode, setChatMode } from "@/lib/modeStorage";

export function usePerChatMode(chatId: string | null): [RagMode, (m: RagMode) => void] {
    const [mode, setMode] = React.useState<RagMode>(() => getChatMode(chatId));

    // When the route/chat changes, re-hydrate from storage
    React.useEffect(() => {
        setMode(getChatMode(chatId));
    }, [chatId]);

    const update = React.useCallback(
        (m: RagMode) => {
            setMode(m);
            setChatMode(chatId, m);
        },
        [chatId]
    );

    return [mode, update];
}
