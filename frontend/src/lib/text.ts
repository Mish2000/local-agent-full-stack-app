export type Dir = "rtl" | "ltr";

export const hebrewRegex = /[\u0590-\u05FF]/;

export const looksLikeCode = (s: string) =>
    s.includes("```") ||
    /[#;{}()[\]<>=]|\/\/|\/\*|\bclass\b|\bdef\b|\bfunction\b|\bimport\b|\b#include\b/i.test(s);

/** Decide visual direction from message content. Code is always LTR. */
export const detectDir = (s: string): Dir =>
    looksLikeCode(s) ? "ltr" : hebrewRegex.test(s) ? "rtl" : "ltr";
