// src/lib/text.ts
export type Dir = "rtl" | "ltr";

export const hebrewRegex = /[\u0590-\u05FF]/;
const hebrewRegexGlobal = /[\u0590-\u05FF]/g;

/** Heuristic: presence of code fences or common code tokens */
export const looksLikeCode = (s: string) =>
    s.includes("```") ||
    /\/\/|\/\*|\bclass\b|\bdef\b|\bfunction\b|\bimport\b|\b#include\b|=>|::/.test(s);

/**
 * Decide visual direction from message content.
 * Priority tweak: if there is visible Hebrew, prefer RTL even if the text also contains
 * brackets or punctuation that might look "code-like". Code blocks themselves are rendered LTR.
 */
export const detectDir = (s: string): Dir => {
    // Count Hebrew characters (ignoring whitespace)
    const hebCount = (s.match(hebrewRegexGlobal) || []).length;
    if (hebCount >= 2) return "rtl"; // prefer RTL when Hebrew is clearly present

    // Otherwise, default as before
    return looksLikeCode(s) ? "ltr" : "ltr";
};
