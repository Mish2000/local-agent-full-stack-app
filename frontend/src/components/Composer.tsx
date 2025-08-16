import { useEffect, useMemo, useRef, useState } from "react";
import {type Dir, detectDir } from "../lib/text";

type Props = {
    disabled: boolean;
    onSend: (text: string, dir: Dir) => void;
    onHeightChange?: (h: number) => void; // notify parent when composer height changes
};

export default function Composer({ disabled, onSend, onHeightChange }: Props) {
    const [text, setText] = useState("");
    const [lastDir, setLastDir] = useState<Dir>("rtl"); // remembered from what you typed last

    const footerRef = useRef<HTMLElement | null>(null);
    const taRef = useRef<HTMLTextAreaElement | null>(null);

    // Direction logic:
    // - When typing => follow actual content.
    // - When empty => show placeholder in the last typed language (default Hebrew).
    const inputDir: Dir = text.trim() ? detectDir(text) : lastDir;
    const placeholder = useMemo(
        () => (lastDir === "rtl" ? "שאל כל דבר…" : "Ask anything…"),
        [lastDir]
    );

    // Update lastDir while typing
    useEffect(() => {
        if (text.trim()) setLastDir(detectDir(text));
    }, [text]);

    // Auto-resize textarea (max 30vh) and inform parent of footer height
    const resizeTextarea = () => {
        const el = taRef.current;
        if (!el) return;
        el.style.height = "auto";
        const max = Math.floor(window.innerHeight * 0.3);
        const next = Math.min(el.scrollHeight, max);
        el.style.height = `${next}px`;
        if (onHeightChange && footerRef.current) {
            onHeightChange(footerRef.current.getBoundingClientRect().height);
        }
    };

    useEffect(() => { resizeTextarea(); }, []);
    useEffect(() => { resizeTextarea(); }, [text]);
    useEffect(() => {
        const onResize = () => { resizeTextarea(); };
        window.addEventListener("resize", onResize);
        return () => window.removeEventListener("resize", onResize);
    }, []);

    const submit = () => {
        const value = text.trim();
        if (!value || disabled) return;
        onSend(value, inputDir);
        // Remember dir used, then clear so placeholder reflects that dir next time
        setLastDir(inputDir);
        setText("");
    };

    return (
        <footer ref={footerRef} className="footer-fixed">
            <div
                className="container"
                style={{ padding: 16, display: "flex", gap: 8, direction: "rtl", alignItems: "flex-end" }}
            >
        <textarea
            ref={taRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    submit();
                }
                // Shift+Enter => newline (default behavior)
            }}
            placeholder={placeholder}
            className="input textarea"
            disabled={disabled}
            dir={inputDir}
            rows={1}
        />
                <button
                    onClick={submit}
                    disabled={disabled || !text.trim()}
                    className="button"
                    style={{ alignSelf: "flex-end" }}
                >
                    {disabled ? "מייצר…" : "שלח"}
                </button>
            </div>
        </footer>
    );
}
