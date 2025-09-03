// src/components/StepsToggle.tsx
import * as React from "react";
import { Button } from "@/components/ui/button.tsx";
import { Eye, EyeOff } from "lucide-react";
import type {JSX} from "react";

const STORAGE_KEY = "show-reasoning-steps";

export default function StepsToggle(): JSX.Element {
    const [enabled, setEnabled] = React.useState<boolean>(() => {
        try {
            return (localStorage.getItem(STORAGE_KEY) ?? "1") === "1";
        } catch {
            return true;
        }
    });

    // Persist and broadcast whenever the state changes
    React.useEffect(() => {
        try {
            const newValue = enabled ? "1" : "0";
            localStorage.setItem(STORAGE_KEY, newValue);
            // Broadcast to any listeners in this tab (ToolCalls, TracesBar)
            window.dispatchEvent(
                new StorageEvent("storage", { key: STORAGE_KEY, newValue })
            );
        } catch {
            /* ignore */
        }
    }, [enabled]);

    // Keep in sync if other components flip the switch (e.g., “hide” button inside ToolCalls)
    React.useEffect(() => {
        const onStorage = (e: StorageEvent) => {
            if (e.key === STORAGE_KEY) {
                setEnabled((e.newValue ?? "1") === "1");
            }
        };
        window.addEventListener("storage", onStorage);
        return () => window.removeEventListener("storage", onStorage);
    }, []);

    return (
        <Button
            variant="outline"
            size="sm"
            className="h-9 rounded-2xl"
            onClick={() => setEnabled((v) => !v)}
            title={enabled ? "הסתרת צעדים" : "הצגת צעדים"}
            aria-pressed={enabled}
        >
            {enabled ? <Eye className="size-4 mr-2" /> : <EyeOff className="size-4 mr-2" />}
            {enabled ? "שלבים: פעיל" : "שלבים: כבוי"}
        </Button>
    );
}
