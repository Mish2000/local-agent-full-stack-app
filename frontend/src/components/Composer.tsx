// src/components/Composer.tsx
import React, {useCallback, useEffect, useMemo, useRef, useState} from "react";
import {type Dir, detectDir} from "@/lib/text";
import {Textarea} from "@/components/ui/textarea";
import {Button} from "@/components/ui/button";
import {Paperclip} from "lucide-react";

type Props = {
    disabled: boolean;
    onSend: (text: string, dir: Dir) => void;
    onHeightChange?: (h: number) => void;
    canSendEmpty: boolean;
    onPickFiles: (files: File[]) => void;
    uploadingProgress?: number | null;
    rightActions?: React.ReactNode;
    showAttach?: boolean;
};

export default function Composer({
                                     disabled,
                                     onSend,
                                     onHeightChange,
                                     canSendEmpty,
                                     onPickFiles,
                                     uploadingProgress = null,
                                     rightActions,
                                     showAttach = true,
                                 }: Props) {
    const [text, setText] = useState("");
    const [lastDir, setLastDir] = useState<Dir>("rtl");

    const footerRef = useRef<HTMLElement | null>(null);
    const taRef = useRef<HTMLTextAreaElement | null>(null);
    const fileRef = useRef<HTMLInputElement | null>(null);

    const inputDir: Dir = text.trim() ? detectDir(text) : lastDir;

    const placeholder = useMemo(
        () => (lastDir === "rtl" ? "כתוב כל דבר…" : "Ask anything…"),
        [lastDir]
    );

    useEffect(() => {
        if (text.trim()) setLastDir(detectDir(text));
    }, [text]);

    const resizeTextarea = useCallback(() => {
        const el = taRef.current;
        if (!el) return;
        // reset height then cap at ~30% viewport height
        el.style.height = "0px";
        const maxPx = Math.floor(window.innerHeight * 0.3);
        el.style.height = Math.min(el.scrollHeight, maxPx) + "px";

        if (onHeightChange && footerRef.current) {
            const h = Math.ceil(footerRef.current.getBoundingClientRect().height);
            onHeightChange(h);
        }
    }, [onHeightChange]);

    useEffect(() => {
        resizeTextarea();
    }, [resizeTextarea, text]);

    useEffect(() => {
        const onResize = () => resizeTextarea();
        window.addEventListener("resize", onResize);
        return () => window.removeEventListener("resize", onResize);
    }, [resizeTextarea]);

    const submit = () => {
        const value = text.trim();
        if (disabled) return;
        if (!value && !canSendEmpty) return; // allow empty iff we have staged files
        onSend(value, inputDir);
        setLastDir(inputDir);
        setText("");
        requestAnimationFrame(() => taRef.current?.focus());
    };

    // Hotkeys: focus ("/"), blur (Esc), Send (Enter or Ctrl/Cmd+Enter)
    useEffect(() => {
        const onKey = (e: KeyboardEvent) => {
            if (e.key === "/" && !e.altKey && !e.ctrlKey && !e.metaKey) {
                const target = e.target as HTMLElement | null;
                const tag = target?.tagName?.toLowerCase();
                const isTyping = tag === "input" || tag === "textarea" || target?.isContentEditable;
                if (!isTyping && document.activeElement !== taRef.current) {
                    e.preventDefault();
                    taRef.current?.focus();
                    return;
                }
            }
            if (e.key === "Escape" && document.activeElement === taRef.current) {
                (document.activeElement as HTMLElement).blur();
            }
        };
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, []);

    const onFilesSelected = (fl: FileList | null) => {
        if (!fl || fl.length === 0) return;
        onPickFiles(Array.from(fl));
        // allow re-selecting same filename
        if (fileRef.current) fileRef.current.value = "";
    };

    const sendDisabled = disabled || (!text.trim() && !canSendEmpty);

    return (
        <footer ref={footerRef} className="footer-fixed">
            <div
                className="container"
                style={{
                    padding: 12,
                    display: "grid",
                    gridTemplateColumns: "1fr auto",
                    gap: 8,
                    direction: "rtl",
                    alignItems: "end",
                }}
            >
                {/* Input row: Paperclip + Export (rightActions) + Textarea */}
                <div className="flex items-end gap-2">
                    {showAttach && (
                        <>
                            <button
                                type="button"
                                className="inline-flex items-center justify-center rounded-2xl border border-[var(--border)] bg-[var(--bg)] h-10 w-10"
                                title={lastDir === "rtl" ? "צרף קבצים" : "Attach files"}
                                aria-label="Attach files"
                                onClick={() => fileRef.current?.click()}
                                disabled={disabled}
                            >
                                <Paperclip className="size-4" />
                            </button>
                            <input
                                ref={fileRef}
                                type="file"
                                multiple
                                className="hidden"
                                onChange={(e) => onFilesSelected(e.currentTarget.files)}
                            />
                        </>
                    )}

                    {/* Export PDF sits immediately to the RIGHT of the paperclip */}
                    {rightActions}

                    <Textarea
                        ref={taRef}
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        onKeyDown={(e) => {
                            if ((e.key === "Enter" && !e.shiftKey) || ((e.ctrlKey || e.metaKey) && e.key === "Enter")) {
                                e.preventDefault();
                                submit();
                            }
                        }}
                        placeholder={placeholder}
                        disabled={disabled}
                        dir={inputDir}
                        rows={1}
                        className="textarea input flex-1"
                        aria-label={lastDir === "rtl" ? "תיבת כתיבה" : "Message composer"}
                    />
                </div>

                {/* Actions row: upload progress + Send at the edge */}
                <div className="flex items-center gap-2">
                    {typeof uploadingProgress === "number" && (
                        <div
                            title="Uploading…"
                            aria-label="Uploading"
                            style={{
                                width: 36,
                                height: 36,
                                borderRadius: 9999,
                                background: `conic-gradient(var(--text) ${Math.round(uploadingProgress * 360)}deg, transparent 0deg)`,
                                mask: "radial-gradient(circle at center, transparent 54%, black 55%)",
                            }}
                        />
                    )}

                    <Button
                        onClick={submit}
                        disabled={sendDisabled}
                        className="button"
                        title={lastDir === "rtl" ? "שלח" : "Send"}
                        aria-label={lastDir === "rtl" ? "שלח" : "Send"}
                    >
                        {lastDir === "rtl" ? (disabled ? "מייצר…" : "שלח") : disabled ? "Generating…" : "Send"}
                    </Button>
                </div>
            </div>
        </footer>
    );
}
