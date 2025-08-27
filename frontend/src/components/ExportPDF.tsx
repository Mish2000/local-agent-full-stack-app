// src/components/ExportPDF.tsx
import * as React from "react";
import type { Source } from "@/lib/sse";
import { splitFences } from "@/lib/markdownUtils";
import { toast } from "sonner";

// Shape of messages we export
type Msg = { role: "user" | "assistant"; content: string };

type Props = {
    chatTitle: string;
    messages: Msg[];
    /** Map: assistant message index -> sources array */
    sourcesByIndex?: Record<number, Source[]>;
};

function escHtml(s: string): string {
    return s
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function renderTextHtml(text: string): string {
    const paras = text.replace(/\r\n/g, "\n").split(/\n{2,}/g);
    return paras.map((p) => `<p class="p">${escHtml(p).replaceAll("\n", "<br/>")}</p>`).join("");
}

function baseName(p?: string): string {
    if (!p) return "unknown";
    const parts = p.replaceAll("\\", "/").split("/");
    return parts[parts.length - 1] || p;
}
function hostFromUrl(u?: string): string {
    if (!u) return "";
    try {
        return new URL(u).host || "";
    } catch {
        return "";
    }
}

function renderMessageHtml(m: Msg, sources?: Source[]): string {
    const isUser = m.role === "user";
    const blocks = splitFences(m.content);

    const body = blocks
        .map((b) =>
            b.kind === "code"
                ? `<div class="code"><div class="code-h">${escHtml(b.lang || "code")}</div><pre><code>${escHtml(
                    b.body
                )}</code></pre></div>`
                : renderTextHtml(b.body)
        )
        .join("");

    const srcHtml =
        !isUser && sources && sources.length
            ? `
      <div class="sources">
        <div class="sources-h">מקורות</div>
        <ul>
          ${sources
                .map((s) => {
                    const range =
                        s.start_line && s.end_line ? ` · lines ${s.start_line}-${s.end_line}` : "";
                    const label = hostFromUrl(s.url) || baseName(s.source);
                    const text = `[${s.id}] ${label}${range}`;
                    return s.url
                        ? `<li><a href="${escHtml(s.url)}" target="_blank" rel="noopener noreferrer">${escHtml(
                            text
                        )}</a></li>`
                        : `<li>${escHtml(text)}</li>`;
                })
                .join("")}
        </ul>
      </div>`
            : "";

    return `
    <section class="${isUser ? "msg user" : "msg assistant"}">
      <div class="tag">${isUser ? "User" : "Assistant"}</div>
      ${body}
      ${srcHtml}
    </section>
  `;
}

/** Build a complete, self-contained HTML document (for iframe srcdoc) */
function buildExportHtml(title: string, when: string, items: Msg[], sourcesByIndex: Record<number, Source[]>): string {
    const head = `
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>${escHtml(title)}</title>
<style>
  /* A4-friendly layout: 794px ~ 210mm @ 96dpi */
  :root { --page-w: 794px; --text: #111; --muted:#555; --border:#ddd; --bg:#fff; --code:#f7f7f7; }
  html, body { background: var(--bg); color: var(--text); margin: 0; }
  body { font: 14px/1.6 system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
  .page { width: var(--page-w); margin: 16px auto; }
  h1 { font-size: 20px; margin: 0 0 8px; }
  .meta { font-size: 12px; color: var(--muted); margin-bottom: 16px; }
  .msg { border: 1px solid var(--border); border-radius: 10px; padding: 12px; margin: 12px 0; page-break-inside: avoid; }
  .msg.user { background: #fafafa; }
  .msg.assistant { background: #fff; }
  .tag { font-weight: 700; font-size: 12px; color: var(--muted); margin-bottom: 6px; }
  .p { margin: 6px 0; }
  pre { margin: 0; white-space: pre-wrap; word-break: break-word; overflow: hidden; }
  pre code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; font-size: 12px; display: block; }
  .code { border:1px solid var(--border); border-radius:10px; overflow:hidden; margin:10px 0; }
  .code-h { font-size: 11px; color: var(--muted); border-bottom: 1px solid var(--border); padding: 6px 10px; background: var(--code); }
  .code pre { padding: 10px; direction: ltr; }
  .sources { margin-top: 10px; }
  .sources-h { font-size: 12px; color: var(--muted); margin-bottom: 4px; font-weight: 600; }
  .sources ul { margin: 0; padding: 0 16px; }
  .sources li { margin: 3px 0; }
  a { color: #0645ad; text-decoration: none; }
  a:hover { text-decoration: underline; }
  @page { size: A4; margin: 16mm; }
</style>`;

    const messagesHtml = items
        .map((m, i) => renderMessageHtml(m, sourcesByIndex[i]))
        .join("");

    const body = `
<div class="page" dir="rtl" lang="he">
  <header>
    <h1>${escHtml(title)}</h1>
    <div class="meta">${escHtml(when)}</div>
  </header>
  <main>${messagesHtml}</main>
  <footer class="meta" style="margin-top:18px">Exported from your chat UI</footer>
</div>`;

    return `<!doctype html><html>${head}<body>${body}</body></html>`;
}

export default function ExportPDF({ chatTitle, messages, sourcesByIndex = {} }: Props) {
    const busyRef = React.useRef(false);

    const exportNow = async () => {
        if (busyRef.current) return;
        busyRef.current = true;

        try {
            const mod = await import("html2pdf.js");
            const html2pdf = mod.default; // typed via our .d.ts

            const safeTitle = (chatTitle?.trim() || "שיחה").replace(/[\\/:*?"<>|]+/g, "-");
            const now = new Date().toLocaleString();

            // Build a full HTML document in an isolated iframe (no global CSS flicker)
            const docHtml = buildExportHtml(safeTitle, now, messages, sourcesByIndex);

            const iframe = document.createElement("iframe");
            iframe.style.position = "fixed";
            iframe.style.left = "-10000px";
            iframe.style.top = "0";
            iframe.style.width = "820px";   // a bit wider than page for safety
            iframe.style.height = "1200px"; // plenty to lay out content
            iframe.setAttribute("sandbox", "allow-same-origin"); // we need same-origin to read its DOM
            document.body.appendChild(iframe);

            // Write srcdoc safely
            const iwin = iframe.contentWindow!;
            const idoc = iframe.contentDocument!;
            idoc.open();
            idoc.write(docHtml);
            idoc.close();

            // Wait for layout to settle
            await new Promise<void>((res) => {
                if (idoc.readyState === "complete") res();
                else iframe.addEventListener("load", () => res(), { once: true });
            });
            await new Promise((r) => requestAnimationFrame(r));

            // Pick the element to render (the whole document or body)
            const target = idoc.body;

            const options = {
                margin: 10,
                filename: `${safeTitle}.pdf`,
                image: { type: "jpeg" as const, quality: 0.98 },
                html2canvas: {
                    backgroundColor: "#ffffff",
                    scale: Math.min(2, window.devicePixelRatio || 1),
                    useCORS: true,
                    scrollX: 0,
                    scrollY: 0,
                    windowWidth: 900,
                    windowHeight: 1400,
                },
                jsPDF: { unit: "mm" as const, format: "a4" as const, orientation: "portrait" as const },
                pagebreak: { mode: ["css"] as const },
            };

            await html2pdf().set(options).from(target as unknown as HTMLElement).save();

            // Cleanup
            document.body.removeChild(iframe);
            busyRef.current = false;
        } catch (e) {
            console.error(e);
            busyRef.current = false;
            toast.error("Failed to export PDF");
        }
    };

    return (
        <button
            type="button"
            onClick={exportNow}
            className="inline-flex items-center gap-2 rounded-xl border px-3 py-1.5 text-sm bg-[var(--panel)]"
            aria-label="Export to PDF"
            title="Export conversation to PDF"
        >
            <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true">
                <path d="M19 14v7H5v-7H3v9h18v-9h-2zM12 16l4-4h-3V3h-2v9H8l4 4z"></path>
            </svg>
            <span>Export PDF</span>
        </button>
    );
}
