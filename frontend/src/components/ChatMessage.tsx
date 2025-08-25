// src/components/ChatMessage.tsx
import { useMemo, useState } from "react";
import Markdown from "../lib/markdown";
import { detectDir, type Dir } from "../lib/text";

/** ------------------------------------------------------------------
 *  Non-breaking math in plain text (outside code):
 *  - Keeps tiny expressions like: n-1, a + b, x/y, 2^k, (n-1), a*b, a×b, a÷b on one line.
 *  - Does NOT touch fenced or inline code; inline code is handled via CSS (nowrap).
 *  ------------------------------------------------------------------ */

const NBSP = "\u00A0";   // non-breaking space
const NBH  = "\u2011";   // non-breaking hyphen

function protectOperatorsChunk(chunk: string): string {
    return chunk
        .replace(/([A-Za-z0-9)])\s*-\s*([A-Za-z0-9(])/g, `$1${NBSP}${NBH}${NBSP}$2`)
        .replace(/([A-Za-z0-9)])\s*\+\s*([A-Za-z0-9(])/g, `$1${NBSP}+${NBSP}$2`)
        .replace(/([A-Za-z0-9)])\s*\/\s*([A-Za-z0-9(])/g, `$1${NBSP}/${NBSP}$2`)
        .replace(/([A-Za-z0-9)])\s*\*\s*([A-Za-z0-9(])/g, `$1${NBSP}*${NBSP}$2`)
        .replace(/([A-Za-z0-9)])\s*\^\s*([A-Za-z0-9(])/g, `$1${NBSP}^${NBSP}$2`)
        .replace(/([A-Za-z0-9)])\s*×\s*([A-Za-z0-9(])/g, `$1${NBSP}×${NBSP}$2`)
        .replace(/([A-Za-z0-9)])\s*÷\s*([A-Za-z0-9(])/g, `$1${NBSP}÷${NBSP}$2`);
}

function protectAtomicMathOutsideCode(text: string): string {
    const re = /```[\s\S]*?```|`[^`]*`/g;
    let out = "";
    let last = 0;
    let m: RegExpExecArray | null;

    while ((m = re.exec(text))) {
        const before = text.slice(last, m.index);
        out += protectOperatorsChunk(before); // transform plain text
        out += m[0];                          // keep code AS-IS
        last = m.index + m[0].length;
    }
    out += protectOperatorsChunk(text.slice(last));
    return out;
}

type Props = {
    id: string;
    role: "user" | "assistant";
    content: string;
};

export default function ChatMessage({ id, role, content }: Props) {
    const [copied, setCopied] = useState(false); // hooks at top-level (no lint warning)
    const dir: Dir = detectDir(content);
    const processed = useMemo(() => protectAtomicMathOutsideCode(content), [content]);

    const copyAll = async () => {
        try {
            await navigator.clipboard.writeText(content);
            setCopied(true);
            window.setTimeout(() => setCopied(false), 1200);
        } catch {
            /* ignore */
        }
    };

    if (role === "user") {
        return (
            <div
                id={id}
                className="msg bubble bubble-user"
                dir={dir}
                style={{ alignSelf: "flex-end", marginLeft: "auto", maxWidth: "80ch" }}
                role="article"
                aria-label="User message"
            >
                {content}
            </div>
        );
    }

    return (
        <div
            id={id}
            className="msg assistant group"
            dir={dir}
            style={{
                alignSelf: "flex-end",
                marginLeft: "auto",
                maxWidth: "80ch",
                position: "relative",
            }}
            role="article"
            aria-label="Assistant message"
        >
            {/* Inline code should never wrap, but code blocks must preserve newlines. */}
            <style>{`
        /* inline code: only when <code> is NOT inside a <pre> */
        .msg.assistant :not(pre) > code { white-space: nowrap !important; }
        /* fenced code blocks: preserve line breaks (and allow horizontal scroll for long lines) */
        .msg.assistant pre { white-space: pre !important; }
        .msg.assistant pre code { white-space: inherit !important; display: block; }
      `}</style>

            {/* Hover/focus toolbar */}
            <div
                className="absolute top-2 right-2 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity"
                role="toolbar"
                aria-label="Message actions"
            >
                <button
                    type="button"
                    onClick={copyAll}
                    className="h-7 rounded-md border px-2 text-xs hover:bg-[var(--hover)] focus:opacity-100 focus:outline-none"
                    title="Copy entire answer"
                    aria-label="Copy entire answer"
                >
                    {copied ? "Copied!" : "Copy"}
                </button>
            </div>

            {/* Render with non-breaking math applied */}
            <Markdown text={processed} />
        </div>
    );
}
