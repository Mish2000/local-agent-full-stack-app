// src/components/AttachmentsShelf.tsx
import * as React from "react";
import {listFiles, type FileItem, unstage, type StagedItem} from "@/lib/api";
import {Trash2, Paperclip, Clock} from "lucide-react";
import {toast} from "sonner";

type Props = {
    chatId: number | null;
    draftId: string | null;
    staged: StagedItem[];
    onRefreshStaged: () => void;
    /** When this number changes, re-fetch saved files list (for existing chats). */
    refreshKey?: number;
};

export default function AttachmentsShelf({
                                             chatId,
                                             draftId,
                                             staged,
                                             onRefreshStaged,
                                             refreshKey = 0,
                                         }: Props) {
    const [saved, setSaved] = React.useState<FileItem[]>([]);

    React.useEffect(() => {
        let cancelled = false;
        (async () => {
            if (!chatId) {
                setSaved([]);
                return;
            }
            try {
                const rows = await listFiles("chat", chatId);
                if (!cancelled) setSaved(rows);
            } catch {
                if (!cancelled) setSaved([]);
            }
        })();
        return () => {
            cancelled = true;
        };
    }, [chatId, refreshKey]);

    const removeStaged = async (sha: string) => {
        if (!draftId) return;
        try {
            await unstage(draftId, sha);
            onRefreshStaged();
        } catch (e: unknown) {
            toast.error(e instanceof Error ? e.message : "Failed to remove");
        }
    };

    const showPending = staged.length > 0;
    const showSaved = (saved.length > 0) && !!chatId;

    return (
        <div
            className="hidden xl:block"
            aria-hidden
            style={{
                alignSelf: "flex-start",
                marginRight: "auto",
                position: "sticky",
                top: 12,
                width: 280,
            }}
        >
            <div className="rounded-2xl border bg-[var(--panel)] shadow-sm overflow-hidden"
                 style={{minWidth: 280, width: "max-content", display: "inline-block"}}>
                <div
                    className="px-3 py-2 border-b border-[var(--border)] text-sm font-semibold flex items-center gap-2">
                    <Paperclip className="size-4 opacity-80"/>
                    העלאות
                </div>

                {/* Pending (staged) files — shown for both new & existing chats */}
                {showPending && (
                    <div className="px-3 py-2 border-b border-[var(--border)]">
                        <div className="text-xs uppercase tracking-wide opacity-70 mb-2 flex items-center gap-1">
                            <Clock className="size-3.5"/>
                            Pending (will attach on Send)
                        </div>
                        <ul className="grid gap-1.5">
                            {staged.map((s) => (
                                <li
                                    key={s.sha256_hex}
                                    className="flex items-center justify-between gap-2 rounded-lg px-2 py-1 text-sm border border-[var(--border)] bg-[var(--bg)]"
                                >
                                    <div className="whitespace-nowrap text-sm leading-tight" title={s.filename}>
                                        {s.filename}
                                        <span className="opacity-60 text-xs"> · {s.size_bytes} bytes</span>
                                    </div>
                                    <button
                                        className="inline-flex items-center gap-1 text-xs rounded-md border px-2 py-1"
                                        onClick={() => removeStaged(s.sha256_hex)}
                                        aria-label={`Remove ${s.filename}`}
                                        title="Remove"
                                    >
                                        <Trash2 className="size-3.5"/>
                                        Remove
                                    </button>
                                </li>
                            ))}
                        </ul>
                    </div>
                )}

                {/* Saved files for this chat */}
                <div className="px-3 py-2">
                    <div className="text-xs uppercase tracking-wide opacity-70 mb-2">
                        {chatId ? "קבצים שמורים יופיעו כאן" : "לא התחילה שיחה"}
                    </div>
                    {!chatId ? (
                        <div className="text-xs opacity-60">התחל שיחה או העלה קובץ</div>
                    ) : showSaved ? (
                        <ul className="grid gap-1.5">
                            {saved.map((f) => (
                                <li
                                    key={f.id}
                                    className="flex items-center justify-between gap-2 rounded-lg px-2 py-1 text-sm border border-[var(--border)] bg-[var(--bg)]"
                                >
                                    <div className="whitespace-nowrap text-sm leading-tight" title={f.filename}>
                                        {f.filename}
                                        {typeof f.size_bytes === "number" && (
                                            <span className="opacity-60 text-xs"> · {f.size_bytes} bytes</span>
                                        )}
                                    </div>
                                </li>
                            ))}
                        </ul>
                    ) : (
                        <div className="text-xs opacity-60">כרגע אין קבצים שמורים</div>
                    )}
                </div>
            </div>
        </div>
    );
}
