// src/components/ChatMessage.tsx
import { useMemo, useRef, useState } from "react";
import Markdown from "../../lib/markdown.tsx";
import { detectDir, type Dir } from "../../lib/text.ts";

/** ------------------------------------------------------------------
 *  Keep tiny math-like expressions readable in plain text (outside code).
 *  Does NOT touch fenced or inline code.
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

/**
 * Normalize Markdown fences live while streaming:
 *  - If model writes "```lang <code>" on one line, insert a newline after the lang.
 *  - Never touch content inside code fences.
 */
function normalizeAndProtect(text: string): string {
    // 1) normalize newlines and ensure "```lang\n" (so the parser recognizes the language immediately)
    const normalized = text
        .replace(/\r\n/g, "\n")
        .replace(/```([A-Za-z0-9_+-]+)[ \t]+(?=\S)/g, "```$1\n");

    // 2) protect only the non-code parts with NBSP/NBH tweaks
    let out = "";
    let i = 0;
    while (true) {
        const start = normalized.indexOf("```", i);
        if (start === -1) {
            out += protectOperatorsChunk(normalized.slice(i));
            break;
        }
        out += protectOperatorsChunk(normalized.slice(i, start));
        const fenceEnd = normalized.indexOf("```", start + 3);
        if (fenceEnd === -1) {
            // open fence: copy the rest untouched (it's code that is still streaming)
            out += normalized.slice(start);
            break;
        }
        out += normalized.slice(start, fenceEnd + 3);
        i = fenceEnd + 3;
    }
    return out;
}

type Props = {
    id: string;
    role: "user" | "assistant";
    content: string;
};

export default function ChatMessage({ id, role, content }: Props) {
    const [copied, setCopied] = useState(false);
    const dir: Dir = detectDir(content);
    const processed = useMemo(() => normalizeAndProtect(content), [content]);
    const containerRef = useRef<HTMLDivElement | null>(null);

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
            ref={containerRef}
            className="msg assistant group"
            dir={dir}
            style={{
                alignSelf: "flex-end",
                marginLeft: "auto",
                maxWidth: "80ch",
                position: "relative",
                // reserve a bit of room so the toolbar never overlaps text
                paddingTop: 36
            }}
            role="article"
            aria-label="Assistant message"
        >
            {/* Floating toolbar (copy whole answer) */}
            <div className="msg-toolbar" role="toolbar" aria-label="Message actions" data-no-copy>
                <button
                    type="button"
                    onClick={copyAll}
                    className="msg-toolbar-btn"
                    title="Copy entire answer"
                    aria-label="Copy entire answer"
                >
                    {copied ? "Copied!" : "Copy answer"}
                </button>
            </div>

            {/* Render prose + fenced code; CSS + the CodeBlock component will keep newlines live. */}
            <Markdown text={processed} />
        </div>
    );
}
