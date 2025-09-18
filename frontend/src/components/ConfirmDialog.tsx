// src/components/ConfirmDialog.tsx
import * as React from "react";
import { createPortal } from "react-dom";
import { Button } from "@/components/ui/button";
import { Trash2, X } from "lucide-react";

type Props = {
    open: boolean;
    title?: string;
    message: string;
    confirmText?: string;
    cancelText?: string;
    destructive?: boolean;
    dir?: "rtl" | "ltr";
    onConfirm: () => void;
    onCancel: () => void;
};

export default function ConfirmDialog({
                                          open,
                                          title = "לאשר פעולה?",
                                          message,
                                          confirmText = "מחק",
                                          cancelText = "ביטול",
                                          destructive = true,
                                          dir = "rtl",
                                          onConfirm,
                                          onCancel,
                                      }: Props) {
    React.useEffect(() => {
        if (!open) return;
        const onKey = (e: KeyboardEvent) => {
            if (e.key === "Escape") onCancel();
            if (e.key === "Enter") onConfirm();
        };
        document.addEventListener("keydown", onKey);
        return () => document.removeEventListener("keydown", onKey);
    }, [open, onCancel, onConfirm]);

    if (!open) return null;

    return createPortal(
        <div
            dir={dir}
            className="fixed inset-0 z-[100] flex items-center justify-center p-4"
            role="dialog"
            aria-modal="true"
            aria-labelledby="confirm-dialog-title"
        >
            {/* Overlay */}
            <div
                className="absolute inset-0 bg-black/40 backdrop-blur-[2px]"
                onClick={onCancel}
            />

            {/* Panel */}
            <div className="relative w-[min(92vw,480px)] rounded-2xl border border-[var(--border)] bg-[var(--panel)] shadow-[var(--shadow)]">
                <div className="p-5 border-b border-[var(--border)] flex items-start gap-3">
                    <div className="mt-0.5">
                        <Trash2 className="size-5 opacity-80" />
                    </div>
                    <div className="flex-1">
                        <h2 id="confirm-dialog-title" className="text-[17px] font-semibold text-[var(--text-strong)]">
                            {title}
                        </h2>
                        <p className="mt-1 text-sm opacity-80">{message}</p>
                    </div>
                    <button
                        type="button"
                        onClick={onCancel}
                        className="rounded-full p-1.5 hover:bg-black/5 dark:hover:bg-white/10"
                        aria-label="Close"
                        title="Close"
                    >
                        <X className="size-4" />
                    </button>
                </div>

                <div className="p-4 flex items-center justify-end gap-2">
                    <Button
                        variant="outline"
                        className="rounded-2xl h-9"
                        onClick={onCancel}
                    >
                        {cancelText}
                    </Button>

                    <Button
                        className={[
                            "rounded-2xl h-9",
                            destructive
                                ? "bg-red-600 hover:bg-red-700 text-white"
                                : ""
                        ].join(" ")}
                        onClick={onConfirm}
                    >
                        {confirmText}
                    </Button>
                </div>
            </div>
        </div>,
        document.body
    );
}
