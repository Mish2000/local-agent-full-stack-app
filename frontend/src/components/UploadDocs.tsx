// src/components/UploadDocs.tsx
import * as React from "react";
import { uploadDocs } from "@/lib/api";
import { toast } from "sonner";
import { Upload, Paperclip, X } from "lucide-react";

type Props = {
    /** If a chat is active we default to "chat" scope, otherwise "user" */
    chatId: number | null;
};

export default function UploadDocs({ chatId }: Props) {
    const [busy, setBusy] = React.useState(false);
    const [showOverlay, setShowOverlay] = React.useState(false);
    const inputRef = React.useRef<HTMLInputElement | null>(null);

    const scope: "user" | "chat" = chatId ? "chat" : "user";

    const pickFiles = React.useCallback(() => {
        inputRef.current?.click();
    }, []);

    const doUpload = React.useCallback(
        async (files: FileList | File[] | null) => {
            if (!files || files.length === 0) return;
            setBusy(true);
            try {
                const resp = await uploadDocs(Array.from(files), scope, chatId ?? undefined);
                if (resp.ok) {
                    const okCnt = resp.files_indexed ?? 0;
                    const skip = (resp.files_skipped ?? []).join(", ");
                    toast.success(`Indexed ${okCnt} file${okCnt === 1 ? "" : "s"}${skip ? ` (skipped: ${skip})` : ""}`);
                } else {
                    toast.error("Upload failed");
                }
            } catch (e: unknown) {
                toast.error(e instanceof Error ? e.message : "Upload failed");
            } finally {
                setBusy(false);
                setShowOverlay(false);
            }
        },
        [scope, chatId]
    );

    // Global drag listeners to show an overlay anywhere in the app
    React.useEffect(() => {
        const onDragOver = (e: DragEvent) => {
            if (!e.dataTransfer) return;
            const hasFiles = Array.from(e.dataTransfer.items || []).some((i) => i.kind === "file");
            if (hasFiles) {
                e.preventDefault();
                setShowOverlay(true);
            }
        };
        const onDrop = (e: DragEvent) => {
            if (!e.dataTransfer) return;
            const files = e.dataTransfer.files;
            if (showOverlay) e.preventDefault();
            if (files && files.length) void doUpload(files);
            setShowOverlay(false);
        };
        const onDragLeave = (e: DragEvent) => {
            // hide overlay when leaving window
            if ((e.relatedTarget as Node | null) === null) setShowOverlay(false);
        };

        window.addEventListener("dragover", onDragOver);
        window.addEventListener("drop", onDrop);
        window.addEventListener("dragleave", onDragLeave);
        return () => {
            window.removeEventListener("dragover", onDragOver);
            window.removeEventListener("drop", onDrop);
            window.removeEventListener("dragleave", onDragLeave);
        };
    }, [doUpload, showOverlay]);

    return (
        <>
            {/* Tiny, neat action chip */}
            <div className="inline-flex items-center gap-2 rounded-xl border px-3 py-1.5 text-sm bg-[var(--panel)]">
                <button
                    type="button"
                    className="inline-flex items-center gap-1"
                    onClick={pickFiles}
                    disabled={busy}
                    title={scope === "chat" ? "Upload to this chat" : "Upload to your user docs"}
                    aria-label="Upload documents"
                >
                    <Paperclip className="size-4" />
                    <span>Upload docs</span>
                    <span className="opacity-60 text-xs">({scope})</span>
                </button>
                <input
                    ref={inputRef}
                    type="file"
                    multiple
                    className="hidden"
                    onChange={(e) => void doUpload(e.currentTarget.files)}
                />
            </div>

            {/* Drop overlay */}
            {showOverlay && (
                <div
                    className="fixed inset-0 z-[60] grid place-items-center bg-black/30 backdrop-blur-[1px]"
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={(e) => e.preventDefault()}
                >
                    <div
                        role="button"
                        tabIndex={0}
                        className="w-[min(560px,92vw)] rounded-2xl border-2 border-dashed border-white/70 bg-[var(--panel)] text-center p-10 shadow-xl outline-none"
                        onClick={() => inputRef.current?.click()}
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={(e) => {
                            e.preventDefault();
                            void doUpload(e.dataTransfer?.files ?? null);
                        }}
                    >
                        <Upload className="size-8 mx-auto mb-3" />
                        <div className="text-lg font-semibold mb-1">Drop files to upload</div>
                        <div className="text-sm opacity-80 mb-3">
                            Scope: <b>{scope}</b>
                            {chatId ? ` (chat #${chatId})` : ""}
                        </div>
                        <div className="text-xs opacity-70">TXT, PDF, DOCX, MD…</div>
                        <div className="mt-5 text-xs opacity-70">or click to choose</div>
                    </div>
                    <button
                        className="absolute top-4 right-4 rounded-full border p-1.5 bg-[var(--panel)]"
                        onClick={() => setShowOverlay(false)}
                        aria-label="Close"
                        title="Close"
                    >
                        <X className="size-4" />
                    </button>
                </div>
            )}
        </>
    );
}
