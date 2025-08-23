import * as React from "react";
import { cn } from "@/lib/utils";
import { assessPassword } from "@/lib/validators";

type Props = { value: string; className?: string };

export default function PasswordMeter({ value, className }: Props) {
    const { level, percent, label } = React.useMemo(() => assessPassword(value), [value]);

    const barColor =
        level === "invalid" ? "bg-neutral-300 dark:bg-neutral-700"
            : level === "weak"   ? "bg-red-500"
                : level === "medium" ? "bg-amber-500"
                    : "bg-green-500";

    return (
        <div className={cn("grid gap-1", className)}>
            <div className="h-2 w-full rounded-full bg-neutral-200 dark:bg-neutral-800 overflow-hidden">
                <div
                    className={cn("h-full transition-all duration-300", barColor)}
                    style={{ width: `${percent}%` }}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={percent}
                    role="progressbar"
                    aria-label="Password strength"
                />
            </div>
            <div
                className={cn(
                    "text-xs font-medium",
                    level === "invalid" && "text-neutral-500",
                    level === "weak" && "text-red-600",
                    level === "medium" && "text-amber-600",
                    level === "strong" && "text-green-600"
                )}
            >
                {level === "invalid" ? "Invalid password" : label}
            </div>
        </div>
    );
}
