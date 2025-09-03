// src/lib/theme.ts
// Single source-of-truth for the header background across the app.
// We persist in localStorage so it survives navigation and can also broadcast cross-tab updates.

const LS_KEY = "theme:headerBg";

export function getHeaderBackground(): string | null {
    try {
        const v = localStorage.getItem(LS_KEY);
        return v && v.trim().length > 0 ? v : null;
    } catch {
        return null;
    }
}

// Accepts raw CSS background string: "#RRGGBB", "rgb(...)", "hsl(...)", "linear-gradient(...)", etc.
export function setHeaderBackground(css: string): void {
    const value = (css || "").trim();
    if (!value) return;
    try {
        localStorage.setItem(LS_KEY, value);
        // Fire both a storage-like event and a custom event so same-tab + cross-tab get notified.
        window.dispatchEvent(new StorageEvent("storage", { key: LS_KEY, newValue: value }));
        window.dispatchEvent(new CustomEvent(LS_KEY, { detail: value }));
    } catch {
        /* noop */
    }
}

// Best-effort extractor from a profile settings object you already send to the backend.
// We try common field names you may already have.
export function extractHeaderBgFromSettings(settings: unknown): string | null {
    // Do NOT use `any`. Keep this typesafe-ish.
    const obj = (settings ?? {}) as Record<string, unknown>;
    const candidates = [
        "headerBg", "bgCss", "backgroundCss", "background", "backgroundColor", "bgColor", "bgGradient"
    ] as const;

    for (const k of candidates) {
        const v = obj[k];
        if (typeof v === "string" && v.trim().length > 0) return v.trim();
    }
    return null;
}

// Stable deterministic fallback if we don't have a stored background.
// Uses a small palette but *stable* by hashing a seed (e.g., user id).
export function stableFallbackBackground(seed: string): string {
    const palette = [
        "linear-gradient(90deg, #0ea5e9, #22c55e)",
        "linear-gradient(90deg, #8b5cf6, #06b6d4)",
        "linear-gradient(90deg, #f97316, #ef4444)",
        "linear-gradient(90deg, #14b8a6, #22c55e)",
        "linear-gradient(90deg, #ec4899, #8b5cf6)"
    ];
    let h = 0;
    for (let i = 0; i < seed.length; i += 1) h = (h * 31 + seed.charCodeAt(i)) >>> 0;
    return palette[h % palette.length];
}
