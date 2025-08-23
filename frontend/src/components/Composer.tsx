import {useCallback, useEffect, useMemo, useRef, useState} from "react";
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
    const placeholder = useMemo(() => (lastDir === "rtl" ? "כתוב כל דבר…" : "Ask anything…"), [lastDir]);

    useEffect(() => {
        if (text.trim()) setLastDir(detectDir(text));
    }, [text]);

    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-expect-error
    // eslint-disable-next-line react-hooks/exhaustive-deps
    const resizeTextarea = useCallback(() => {
        const el = taRef.current;
        if (!el) return;
        el.style.height = "auto";
        const max = Math.floor(window.innerHeight * 0.3);
        el.style.height = `${Math.min(el.scrollHeight, max)}px`;
        if (onHeightChange && footerRef.current) {
            onHeightChange(footerRef.current.getBoundingClientRect().height);
        }
    });

    useEffect(() => { resizeTextarea(); }, [resizeTextarea]);
    useEffect(() => { resizeTextarea(); }, [resizeTextarea, text]);
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
    };

    return (
        <footer ref={footerRef} className="footer-fixed">
            <div className="container" style={{ padding: 16, display: "flex", gap: 8, direction: "rtl", alignItems: "flex-end" }}>
                <Textarea
                    ref={taRef}
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                            e.preventDefault();
                            submit();
                        }
                    }}
                    placeholder={placeholder}
                    disabled={disabled}
                    dir={inputDir}
                    rows={1}
                    className="textarea input"
                />
                <Button
                    onClick={submit}
                    disabled={disabled || !text.trim()}
                    className="button"
                    title="שלח"
                >
                    {disabled ? "מייצר…" : "שלח"}
                </Button>
            </div>
        </footer>
    );
}
