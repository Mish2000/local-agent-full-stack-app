// src/lib/markdown.tsx
import * as React from "react";

/** ------------------------------------------------------------------
 *  Tiny markdown renderer (no external deps, no path aliases)
 *  Supports:
 *   - headings (#..######)
 *   - unordered lists (- item)
 *   - paragraphs
 *   - fenced code ```lang ... ```
 *   - inline code `x`
 *   - links [text](https://...)
 *  ------------------------------------------------------------------ */

type Node =
    | { t: "p"; c: string }
    | { t: "h"; level: number; c: string }
    | { t: "ul"; items: string[] }
    | { t: "code"; lang?: string; c: string };

/** Local, self-contained CodeBlock to avoid any import/alias issues */
function CodeBlock({ code, lang }: { code: string; lang?: string }) {
    const [copied, setCopied] = React.useState(false);

    const doCopy = async () => {
        try {
            if (navigator.clipboard && navigator.clipboard.writeText) {
                await navigator.clipboard.writeText(code);
            } else {
                // Fallback for odd environments
                const ta = document.createElement("textarea");
                ta.value = code;
                ta.style.position = "fixed";
                ta.style.opacity = "0";
                document.body.appendChild(ta);
                ta.select();
                document.execCommand("copy");
                document.body.removeChild(ta);
            }
            setCopied(true);
            window.setTimeout(() => setCopied(false), 1200);
        } catch {
            // ignore
        }
    };

    return (
        <div className="relative my-3 overflow-hidden rounded-xl border">
            <div className="flex items-center justify-between px-3 py-1.5 text-xs border-b bg-[var(--panel)]">
                <span className="opacity-70">{lang ? lang : "code"}</span>
                <button
                    type="button"
                    onClick={doCopy}
                    className="h-7 rounded-md border px-2 py-1 text-xs hover:bg-[var(--hover)]"
                    aria-label="Copy code"
                    title="Copy code"
                >
                    {copied ? "Copied!" : "Copy"}
                </button>
            </div>
            <pre className="overflow-x-auto p-3 text-sm leading-relaxed" data-lang={lang || ""}>
        <code>{code}</code>
      </pre>
        </div>
    );
}

function splitFences(md: string): Array<{ kind: "text" | "code"; lang?: string; body: string }> {
    const out: Array<{ kind: "text" | "code"; lang?: string; body: string }> = [];
    const parts = md.split(/```/g);
    for (let i = 0; i < parts.length; i++) {
        const part = parts[i];
        if (i % 2 === 0) {
            if (part) out.push({ kind: "text", body: part });
        } else {
            const nl = part.indexOf("\n");
            let lang = "";
            let body = part;
            if (nl >= 0) {
                lang = part.slice(0, nl).trim();
                body = part.slice(nl + 1);
            }
            out.push({ kind: "code", lang: lang || undefined, body });
        }
    }
    return out;
}

function parseBlocks(md: string): Node[] {
    const nodes: Node[] = [];
    const segments = splitFences(md);
    for (const seg of segments) {
        if (seg.kind === "code") {
            const code = seg.body.replace(/\s+$/g, "");
            nodes.push({ t: "code", lang: seg.lang, c: code });
            continue;
        }
        const text = seg.body.replace(/\r\n/g, "\n");
        const lines = text.split("\n");
        let buffer: string[] = [];
        let list: string[] | null = null;

        const flushPara = () => {
            const content = buffer.join("\n").trim();
            if (content) nodes.push({ t: "p", c: content });
            buffer = [];
        };
        const flushList = () => {
            if (list && list.length) nodes.push({ t: "ul", items: list });
            list = null;
        };

        for (const raw of lines) {
            const line = raw.trimEnd();
            if (!line.trim()) {
                flushList();
                flushPara();
                continue;
            }
            const m = /^(#{1,6})\s+(.*)$/.exec(line);
            if (m) {
                flushList();
                flushPara();
                nodes.push({ t: "h", level: m[1].length, c: m[2] });
                continue;
            }
            const li = /^[-*]\s+(.*)$/.exec(line);
            if (li) {
                flushPara();
                if (!list) list = [];
                (list as string[]).push(li[1]);
                continue;
            }
            if (list) flushList();
            buffer.push(line);
        }
        flushList();
        flushPara();
    }
    return nodes;
}

function renderInline(text: string): React.ReactNode {
    // inline code (split on backticks)
    let parts: Array<string | React.ReactNode> = [];
    const segs = text.split(/`/g);
    for (let i = 0; i < segs.length; i++) {
        const s = segs[i];
        if (i % 2 === 1) {
            parts.push(
                <code key={"ic" + i} className="rounded px-1 py-0.5 border text-[0.9em]">
                    {s}
                </code>
            );
        } else if (s) {
            parts.push(s);
        }
    }

    // links [text](url)
    parts = parts.flatMap((chunk, idx) => {
        if (typeof chunk !== "string") return [chunk];
        const arr: React.ReactNode[] = [];
        const re = /\[([^\]]+)]\((https?:\/\/[^\s)]+)\)/g;
        let m: RegExpExecArray | null;
        let last = 0;
        const str = chunk;
        while ((m = re.exec(str))) {
            const before = str.slice(last, m.index);
            if (before) arr.push(before);
            arr.push(
                <a
                    key={"lnk" + idx + "-" + m.index}
                    href={m[2]}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="underline hover:opacity-80"
                >
                    {m[1]}
                </a>
            );
            last = m.index + m[0].length;
        }
        const tail = str.slice(last);
        if (tail) arr.push(tail);
        return arr;
    });

    return parts;
}

export default function Markdown({ text }: { text: string }) {
    const blocks = parseBlocks(text);
    return (
        <div className="space-y-2">
            {blocks.map((node, i) => {
                switch (node.t) {
                    case "h": {
                        const level = Math.min(6, Math.max(1, node.level));
                        return React.createElement(
                            `h${level}`,
                            { key: i, className: "mt-3 mb-2 font-semibold" },
                            renderInline(node.c)
                        );
                    }
                    case "ul":
                        return (
                            <ul key={i} className="list-disc ms-5 my-2 space-y-1">
                                {node.items.map((it, j) => (
                                    <li key={j}>{renderInline(it)}</li>
                                ))}
                            </ul>
                        );
                    case "code":
                        return <CodeBlock key={i} code={node.c} lang={node.lang} />;
                    case "p":
                    default:
                        return (
                            <p key={i} className="my-2 leading-relaxed">
                                {renderInline(node.c)}
                            </p>
                        );
                }
            })}
        </div>
    );
}
