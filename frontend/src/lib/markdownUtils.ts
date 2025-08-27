// src/lib/markdownUtils.ts
export type FencedSegment = { kind: "text" | "code"; lang?: string; body: string };

/** Split markdown by fences ```lang ... ``` into text/code segments. */
export function splitFences(md: string): FencedSegment[] {
    const out: FencedSegment[] = [];
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
