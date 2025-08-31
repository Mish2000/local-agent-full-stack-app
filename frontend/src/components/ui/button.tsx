// src/components/ui/button.tsx
import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

/**
 * Unified Button with subtle hover/press animations (applies across the app).
 * - Hover: raise by 1px and add shadow
 * - Active: slight press (scale 0.98)
 * - Respects “prefers-reduced-motion”
 */
const buttonVariants = cva(
    [
        "inline-flex items-center justify-center whitespace-nowrap select-none",
        "rounded-2xl text-sm font-medium",
        "transition-all duration-200 will-change-transform [transform:translateZ(0)]",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2",
        "disabled:pointer-events-none disabled:opacity-50",
        // micro-interactions
        "hover:-translate-y-px hover:shadow-[var(--shadow)] active:translate-y-0 active:scale-[0.98]",
        "motion-reduce:transition-none",
        // default sizing
        "h-10 px-4",
    ].join(" "),
    {
        variants: {
            variant: {
                default:
                    "bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900",
                outline:
                    "border border-neutral-300 dark:border-neutral-700 bg-transparent text-[var(--text)]",
                ghost:
                    "bg-transparent text-[var(--text)] hover:bg-neutral-100 dark:hover:bg-neutral-800",
            },
            size: {
                sm: "h-9 px-3",
                md: "h-10 px-4",
                lg: "h-11 px-6 text-base",
                icon: "h-10 w-10 p-0",
            },
        },
        defaultVariants: { variant: "default", size: "md" },
    }
);

export interface ButtonProps
    extends React.ButtonHTMLAttributes<HTMLButtonElement>,
        VariantProps<typeof buttonVariants> {}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
    ({ className, variant, size, ...props }, ref) => (
        <button
            ref={ref}
            className={cn(buttonVariants({ variant, size }), className)}
            {...props}
        />
    )
);
Button.displayName = "Button";
