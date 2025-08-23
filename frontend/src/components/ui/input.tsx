import * as React from "react";
import { cn } from "@/lib/utils";

export type InputProps = React.InputHTMLAttributes<HTMLInputElement>

export const Input = React.forwardRef<HTMLInputElement, InputProps>(function Input(
    { className, ...props },
    ref
) {
    return (
        <input
            ref={ref}
            className={cn(
                "w-full rounded-2xl border border-[var(--border)] bg-[var(--bg)] px-4 py-2.5",
                "outline-none focus:ring-2 focus:ring-neutral-300 dark:focus:ring-neutral-700",
                className
            )}
            {...props}
        />
    );
});
