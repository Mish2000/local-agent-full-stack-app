import React, { useState } from "react";
import { Link } from "react-router-dom";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { validateEmail } from "@/lib/validators";
import { Sparkles } from "lucide-react";

export default function Forgot() {
    const [email, setEmail] = useState("");
    const [busy, setBusy] = useState(false);
    const [resetUrl, setResetUrl] = useState<string | null>(null);

    async function submit(e: React.FormEvent) {
        e.preventDefault();
        if (!validateEmail(email)) {
            toast.error("Please enter a valid email address (e.g., a@b.com)");
            return;
        }
        setBusy(true);
        try {
            const res = await api<{ ok: boolean; reset_url?: string }>("/auth/forgot", {
                method: "POST",
                body: { email },
            });
            setResetUrl(res.reset_url ?? null);
            toast.success("If an account exists, a reset link was sent.");
        } catch (err: unknown) {
            const msg = err instanceof Error ? err.message : "Request failed";
            toast.error(msg);
        } finally {
            setBusy(false);
        }
    }

    return (
        <div className="min-h-dvh bg-[var(--bg)] text-[var(--text)] grid grid-rows-[auto_1fr]">
            <header className="sticky top-0 z-40">
                <div className="container py-3 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <Sparkles className="size-5" />
                        <h1 className="text-[20px] font-bold">Agent 2.0 — שחזור סיסמה</h1>
                    </div>
                    <nav className="flex items-center gap-3">
                        <Link to="/" className="hover:underline">בית</Link>
                        <Link to="/login" className="hover:underline">התחברות</Link>
                    </nav>
                </div>
            </header>

            <main className="container grid content-center">
                <form onSubmit={submit} className="w-full max-w-lg place-self-center grid gap-5 rounded-2xl border border-[var(--border)] p-8 bg-[var(--panel)] shadow-[var(--shadow)]">
                    <div>
                        <Label htmlFor="email">אימייל</Label>
                        <Input id="email" type="email" dir="ltr" inputMode="email" autoComplete="email"
                               placeholder="you@example.com" value={email} onChange={(e) => setEmail(e.target.value)} />
                    </div>
                    <Button disabled={busy} className="h-11 rounded-2xl">{busy ? "שולח…" : "שלח קישור שחזור"}</Button>

                    {resetUrl && (
                        <div className="text-sm mt-2 p-3 rounded-xl border border-[var(--border)] bg-[var(--bg)]">
                            <div className="font-semibold mb-1">DEV: Reset link</div>
                            <div className="break-all" dir="ltr">{resetUrl}</div>
                            <div className="text-xs opacity-70 mt-1">In production, this would be emailed to you.</div>
                            <div className="mt-2">
                                <Link to={new URL(resetUrl).pathname + new URL(resetUrl).search} className="underline">Open reset page</Link>
                            </div>
                        </div>
                    )}
                </form>
            </main>
        </div>
    );
}
