// src/components/StepsToggle.tsx
import * as React from "react";
import { Button } from "@/components/ui/button";
import { Eye, EyeOff } from "lucide-react";

export default function StepsToggle() {
    const [enabled, setEnabled] = React.useState<boolean>(() => {
        try {
            return (localStorage.getItem("show-reasoning-steps") ?? "1") === "1";
        } catch {
            return true;
        }
    });

    React.useEffect(() => {
        try {
            localStorage.setItem("show-reasoning-steps", enabled ? "1" : "0");
        } catch {
            /* ignore */
        }
        // Notify listeners (ToolCalls subscribes to 'storage')
        try {
            window.dispatchEvent(
                new StorageEvent("storage", {
                    key: "show-reasoning-steps",
                    newValue: enabled ? "1" : "0",
                })
            );
        } catch {
            /* ignore */
        }
    }, [enabled]);

    return (
        <Button
            variant="outline"
            size="sm"
            className="h-9 rounded-2xl"
            onClick={() => setEnabled((v) => !v)}
            title="Show/Hide reasoning steps"
            aria-pressed={enabled}
        >
            {enabled ? <Eye className="size-4 mr-2" /> : <EyeOff className="size-4 mr-2" />}
            {enabled ? "Steps: On" : "Steps: Off"}
        </Button>
    );
}
