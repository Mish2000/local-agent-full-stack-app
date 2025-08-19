export type ToastItem = {
    id: number;
    type: "error" | "info" | "success";
    text: string;
};

export default function Toasts({ items }: { items: ToastItem[] }) {
    if (!items || items.length === 0) return null;
    return (
        <div className="toasts-wrap" dir="rtl" aria-live="polite" aria-atomic="true">
            {items.map((t) => (
                <div key={t.id} className={`toast ${t.type}`} role="status">
                    <div className="toast-text">{t.text}</div>
                </div>
            ))}
        </div>
    );
}

