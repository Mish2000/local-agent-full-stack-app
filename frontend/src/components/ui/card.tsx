import * as React from "react";

export function Card({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
    return (
        <div
            className={[
                "rounded-2xl border",
                "border-[var(--border)] bg-[var(--bg)] shadow-[var(--shadow)]",
                className || ""
            ].join(" ")}
            {...props}
        />
    );
}
export function CardHeader(props: React.HTMLAttributes<HTMLDivElement>) {
    return <div className="p-4 border-b border-[var(--border)]" {...props} />;
}
export function CardContent(props: React.HTMLAttributes<HTMLDivElement>) {
    return <div className="p-4" {...props} />;
}
