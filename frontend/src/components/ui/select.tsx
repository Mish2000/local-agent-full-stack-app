import * as React from "react";
import * as SelectPrimitive from "@radix-ui/react-select";
import { ChevronDown, Check } from "lucide-react";

export function Select(props: SelectPrimitive.SelectProps) {
    return <SelectPrimitive.Root {...props} />;
}

export function SelectTrigger({
                                  // eslint-disable-next-line @typescript-eslint/no-unused-vars
                                  className,
                                  children,
                                  ...props
                              }: React.ComponentPropsWithoutRef<typeof SelectPrimitive.Trigger>) {
    return (
        <SelectPrimitive.Trigger
            className={[
                "inline-flex items-center justify-between gap-2 min-w-[260px]",
                "rounded-xl border px-3 py-2 text-sm font-semibold",
                "bg-[var(--bg)] border-[var(--border)] text-[var(--text-strong)]"
            ].join(" ")}
            {...props}
        >
            <SelectPrimitive.Value />
            {children}
            <SelectPrimitive.Icon>
                <ChevronDown className="size-4 opacity-80" />
            </SelectPrimitive.Icon>
        </SelectPrimitive.Trigger>
    );
}

export const SelectContent = React.forwardRef<
    React.ElementRef<typeof SelectPrimitive.Content>,
    React.ComponentPropsWithoutRef<typeof SelectPrimitive.Content>
// eslint-disable-next-line @typescript-eslint/no-unused-vars
>(({ className, children, ...props }, ref) => (
    <SelectPrimitive.Portal>
        <SelectPrimitive.Content
            ref={ref}
            className={[
                "z-50 overflow-hidden rounded-xl border",
                "bg-[var(--panel)] border-[var(--border)]"
            ].join(" ")}
            {...props}
        >
            <SelectPrimitive.Viewport className="p-1 text-[var(--text-strong)]">{children}</SelectPrimitive.Viewport>
        </SelectPrimitive.Content>
    </SelectPrimitive.Portal>
));
SelectContent.displayName = "SelectContent";

export const SelectItem = React.forwardRef<
    React.ElementRef<typeof SelectPrimitive.Item>,
    React.ComponentPropsWithoutRef<typeof SelectPrimitive.Item>
// eslint-disable-next-line @typescript-eslint/no-unused-vars
>(({ className, children, ...props }, ref) => (
    <SelectPrimitive.Item
        ref={ref}
        className={[
            "relative flex cursor-pointer select-none items-center rounded-lg px-3 py-2 text-sm outline-none",
            "text-[var(--text-strong)]",
            "data-[highlighted]:bg-[var(--user)]"
        ].join(" ")}
        {...props}
    >
        <SelectPrimitive.ItemText>{children}</SelectPrimitive.ItemText>
        <SelectPrimitive.ItemIndicator className="absolute right-2 inline-flex items-center">
            <Check className="size-4" />
        </SelectPrimitive.ItemIndicator>
    </SelectPrimitive.Item>
));
SelectItem.displayName = "SelectItem";
