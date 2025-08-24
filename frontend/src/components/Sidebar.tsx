// src/components/Sidebar.tsx
// src/components/Sidebar.tsx
import type {ChatSummary} from "@/lib/chats";
import { Pencil, Trash2, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";

function fmtTime(iso: string) {
    try {
        const d = new Date(iso);
        return d.toLocaleString();
    } catch {
        return iso;
    }
}

type Props = {
    items: ChatSummary[];
    activeId: number | null;
    onSelect: (id: number) => void;
    onNew: () => void;
    onRename: (id: number) => void;
    onDelete: (id: number) => void;
};

export default function Sidebar({ items, activeId, onSelect, onNew, onRename, onDelete }: Props) {
    return (
        <aside className="sidebar h-full shrink-0 border-r border-[var(--border)] bg-[var(--panel)] flex flex-col">
            <div className="p-3 border-b border-[var(--border)] flex items-center justify-between">
                <div className="font-bold">השיחות שלי</div>
                <Button className="h-9 rounded-2xl" onClick={onNew} title="שיחה חדשה">
                    <Plus className="size-4 ml-1" /> חדשה
                </Button>
            </div>

            <div className="flex-1 overflow-y-auto">
                {items.length === 0 && (
                    <div className="p-4 text-sm opacity-70">אין עדיין שיחות. התחילו אחת חדשה.</div>
                )}

                <ul className="p-2 grid gap-2 overflow-x-hidden">
                    {items.map((c) => {
                        const isActive = c.id === activeId;
                        return (
                            <li key={c.id} className="min-w-0">
                                <button
                                    onClick={() => onSelect(c.id)}
                                    className={[
                                        "w-full text-right rounded-xl border p-3 transition min-w-0 overflow-hidden",
                                        isActive
                                            ? "bg-white dark:bg-neutral-900 border-neutral-300 dark:border-neutral-700"
                                            : "bg-[var(--bg)] border-[var(--border)] hover:bg-white/70 dark:hover:bg-neutral-900/70",
                                    ].join(" ")}
                                >
                                    <div className="font-semibold truncate">{c.title || "Untitled chat"}</div>
                                    {c.last_preview && (
                                        <div className="text-[12px] opacity-80 truncate mt-0.5">{c.last_preview}</div>
                                    )}
                                    <div className="text-[11px] opacity-60 mt-1">{fmtTime(c.updated_at)}</div>
                                </button>
                                <div className="flex items-center justify-end gap-2 mt-1 pr-1">
                                    <button
                                        className="text-[12px] opacity-80 hover:opacity-100 flex items-center gap-1"
                                        onClick={() => onRename(c.id)}
                                        title="שנה שם"
                                    >
                                        <Pencil className="size-3" /> שינוי שם
                                    </button>
                                    <button
                                        className="text-[12px] opacity-80 hover:opacity-100 text-red-600 flex items-center gap-1"
                                        onClick={() => onDelete(c.id)}
                                        title="מחיקה"
                                    >
                                        <Trash2 className="size-3" /> מחק
                                    </button>
                                </div>
                            </li>
                        );
                    })}
                </ul>
            </div>
        </aside>
    );
}
