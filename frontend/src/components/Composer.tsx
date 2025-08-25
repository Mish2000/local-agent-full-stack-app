// src/components/Composer.tsx

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { type Dir, detectDir } from "@/lib/text";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";

type Props = {
    disabled: boolean;
    onSend: (text: string, dir: Dir) => void;
    onHeightChange?: (h: number) => void;
};

export default function Composer({ disabled, onSend, onHeightChange }: Props) {
    const [text, setText] = useState("");
    const [lastDir, setLastDir] = useState<Dir>("rtl");

    const footerRef = useRef<HTMLElement | null>(null);
    const taRef = useRef<HTMLTextAreaElement | null>(null);

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
        // reset, then cap to ~30% of viewport height
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
        if (!value || disabled) return;
        onSend(value, inputDir);
        setLastDir(inputDir);
        setText("");
        // keep focus for quick follow-ups
        requestAnimationFrame(() => taRef.current?.focus());
    };

    // Global hotkeys that do NOT change layout/markup
    useEffect(() => {
        const onKey = (e: KeyboardEvent) => {
            // "/" → focus the composer unless user is typing in another field
            if (e.key === "/" && !e.altKey && !e.ctrlKey && !e.metaKey) {
                const target = e.target as HTMLElement | null;
                const tag = target?.tagName?.toLowerCase();
                const isTypingField =
                    tag === "input" || tag === "textarea" || target?.isContentEditable;
                if (!isTypingField && document.activeElement !== taRef.current) {
                    e.preventDefault();
                    taRef.current?.focus();
                    return;
                }
            }
            // Esc → blur the composer
            if (e.key === "Escape" && document.activeElement === taRef.current) {
                (document.activeElement as HTMLElement).blur();
            }
        };
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, []);

    return (
        <footer ref={footerRef} className="footer-fixed">
            <div
                className="container"
                style={{
                    padding: 16,
                    display: "flex",
                    gap: 8,
                    direction: "rtl",
                    alignItems: "flex-end",
                }}
            >
                <Textarea
                    ref={taRef}
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    onKeyDown={(e) => {
                        // Enter (no Shift) sends; Ctrl/Cmd+Enter also sends
                        if (
                            (e.key === "Enter" && !e.shiftKey) ||
                            ((e.ctrlKey || e.metaKey) && e.key === "Enter")
                        ) {
                            e.preventDefault();
                            submit();
                        }
                    }}
                    placeholder={placeholder}
                    disabled={disabled}
                    dir={inputDir}
                    rows={1}
                    className="textarea input"
                    aria-label={lastDir === "rtl" ? "תיבת כתיבה" : "Message composer"}
                />
                <Button
                    onClick={submit}
                    disabled={disabled || !text.trim()}
                    className="button"
                    title={lastDir === "rtl" ? "שלח" : "Send"}
                    aria-label={lastDir === "rtl" ? "שלח" : "Send"}
                >
                    {lastDir === "rtl" ? (disabled ? "מייצר…" : "שלח") : disabled ? "Thinking…" : "Send"}
                </Button>
            </div>
        </footer>
    );
}
