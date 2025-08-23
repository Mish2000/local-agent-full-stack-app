import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
    "inline-flex items-center justify-center whitespace-nowrap rounded-2xl text-sm font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none h-10 px-4 py-2",
    {
        variants: {
            variant: {
                default: "bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900",
                outline: "border border-neutral-300 dark:border-neutral-700 bg-transparent",
                ghost: "hover:bg-neutral-100 dark:hover:bg-neutral-800",
            },
            size: {
                sm: "h-9 px-3",
                md: "h-10 px-4",
                lg: "h-11 px-5 text-base",
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
        <button ref={ref} className={cn(buttonVariants({ variant, size }), className)} {...props} />
    )
);
Button.displayName = "Button";
