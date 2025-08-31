// src/lib/chats.ts
import { api } from "@/lib/api";

export type ChatSummary = {
    id: number;
    title: string;
    updated_at: string;
    last_preview: string | null;
};

export type ChatMessageRow = {
    id: number;
    role: "user" | "assistant";
    content: string;
    created_at: string;
};

export async function listChats(): Promise<ChatSummary[]> {
    return api<ChatSummary[]>("/chats");
}

export async function createChat(title?: string): Promise<ChatSummary> {
    return api<ChatSummary, { title?: string }>("/chats", {
        method: "POST",
        body: { title },
    });
}

export async function renameChat(chatId: number, title: string): Promise<ChatSummary> {
    return api<ChatSummary, { title: string }>(`/chats/${chatId}`, {
        method: "PATCH",
        body: { title },
    });
}

export async function deleteChat(chatId: number): Promise<{ ok: true }> {
    return api<{ ok: true }>(`/chats/${chatId}`, { method: "DELETE" });
}

export async function listMessages(chatId: number): Promise<ChatMessageRow[]> {
    return api<ChatMessageRow[]>(`/chats/${chatId}/messages`);
}

export async function autoTitle(
    chatId: number,
    firstUserText?: string
): Promise<{ id: number; title: string }> {
    // Always send a small JSON body to keep proxies/CORS happy and avoid empty-body edge cases
    const body = JSON.stringify({
        first_user_text: (firstUserText ?? "").trim(),
    });

    return api<{ id: number; title: string }>(`/chats/${chatId}/auto-title`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
    });
}

