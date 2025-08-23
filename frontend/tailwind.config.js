/** @type {import('tailwindcss').Config} */
export default {
    content: ["./index.html", "./src/**/*.{ts,tsx}"],
    theme: {
        extend: {
            container: { center: true, padding: "1rem" },
            boxShadow: {
                soft: "0 6px 24px rgba(0,0,0,.08), 0 2px 6px rgba(0,0,0,.06)",
                softDark: "0 6px 24px rgba(0,0,0,.40), 0 2px 6px rgba(0,0,0,.30)",
            },
            colors: {
                bg: { DEFAULT: "#ffffff", dark: "#0b0f19" },
                panel: { DEFAULT: "#f6f8fa", dark: "#0f172a" },
                border: { DEFAULT: "#e5e7eb", dark: "#374151" },
                text: { DEFAULT: "#111827", strong: "#0b0f19", light: "#6b7280" },
            },
            borderRadius: { xl2: "1rem" },
        },
    },
    darkMode: ["class", '[data-theme="dark"]'],
    plugins: [],
};
