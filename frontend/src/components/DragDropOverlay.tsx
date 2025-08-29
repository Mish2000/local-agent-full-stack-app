// src/components/DragDropOverlay.tsx
import * as React from "react";
import { UploadCloud } from "lucide-react";

type Props = {
    /** Show/hide the overlay */
    visible: boolean;
    /** Called when files are dropped inside the overlay */
    onDropFiles: (files: File[]) => void;
    /** Close the overlay (e.g., user presses ESC or leaves the window) */
    onCancel: () => void;
};

/**
 * Full-bleed overlay that sits on top of the chat column only.
 * It accepts drops and forwards the files for staging.
 */
export default function DragDropOverlay({ visible, onDropFiles, onCancel }: Props) {
    const [pulse, setPulse] = React.useState(0);

    React.useEffect(() => {
        if (!visible) return;
        let raf = 0;
        const t0 = performance.now();
        const loop = (t: number) => {
            const dt = (t - t0) / 1000;
            setPulse(Math.abs(Math.sin(dt * Math.PI)));
            raf = requestAnimationFrame(loop);
        };
        raf = requestAnimationFrame(loop);
        return () => cancelAnimationFrame(raf);
    }, [visible]);

    if (!visible) return null;

    return (
        <div
            className="absolute inset-0 z-[50] grid place-items-center"
            style={{ background: "rgba(0,0,0,0.25)", backdropFilter: "blur(1px)" }}
            onDragOver={(e) => {
                e.preventDefault();
            }}
            onDrop={(e) => {
                e.preventDefault();
                const fl = e.dataTransfer?.files;
                if (fl && fl.length > 0) {
                    onDropFiles(Array.from(fl));
                }
                onCancel();
            }}
            onClick={onCancel}
            role="region"
            aria-label="Drop files to upload"
        >
            <div
                className="rounded-3xl border-2 border-dashed text-center shadow-xl px-10 py-9"
                style={{
                    borderColor: "rgba(255,255,255,0.8)",
                    background: "var(--panel)",
                }}
                onClick={(e) => e.stopPropagation()}
            >
                <div
                    className="mx-auto mb-4"
                    style={{
                        width: 84,
                        height: 84,
                        borderRadius: "9999px",
                        background: `conic-gradient(var(--text) ${Math.round(pulse * 360)}deg, transparent 0deg)`,
                        mask: "radial-gradient(circle at center, transparent 68%, black 70%)",
                    }}
                    aria-hidden
                />
                <UploadCloud className="size-7 mx-auto mb-2 opacity-90" />
                <div className="text-lg font-semibold">Drop files to upload</div>
                <div className="text-sm opacity-80">TXT, MD, PDF, DOCX…</div>
                <div className="text-xs opacity-70 mt-3">Tip: You can also click the paperclip</div>
            </div>
        </div>
    );
}
