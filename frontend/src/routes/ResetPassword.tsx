import React, { useMemo, useState } from "react";
import { useNavigate, Link, useSearchParams } from "react-router-dom";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { api } from "@/lib/api";
import PasswordMeter from "@/components/PasswordMeter";
import { assessPassword } from "@/lib/validators";
import { Sparkles } from "lucide-react";

export default function ResetPassword() {
    const [sp] = useSearchParams();
    const nav = useNavigate();

    const [password, setPassword] = useState("");
    const [busy, setBusy] = useState(false);

    const stats = useMemo(() => assessPassword(password), [password]);
    const invalid = stats.level === "invalid";

    async function submit(e: React.FormEvent) {
        e.preventDefault();
        const token = sp.get("token") || ""; // read when needed; avoids unused warning
        if (!token) {
            toast.error("Invalid reset link");
            return;
        }
        if (invalid) {
            toast.error("Password must be 8–20 chars with at least one uppercase letter and one number");
            return;
        }
        setBusy(true);
        try {
            await api<{ ok: boolean }, { token: string; new_password: string }>("/auth/reset", {
                method: "POST",
                body: { token, new_password: password },
            });
            toast.success("Password updated. Redirecting to login…");
            nav("/login", { replace: true });
        } catch (err: unknown) {
            const msg = err instanceof Error ? err.message : "Reset failed";
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
                        <h1 className="text-[20px] font-bold">Agent 2.0 — איפוס סיסמה</h1>
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
                        <Label htmlFor="password">סיסמה חדשה</Label>
                        <Input id="password" type="password" autoComplete="new-password" maxLength={20}
                               value={password} onChange={(e) => setPassword(e.target.value)} />
                        <div className="mt-2"><PasswordMeter value={password} /></div>
                    </div>
                    <Button disabled={busy} className="h-11 rounded-2xl">{busy ? "מעדכן…" : "עדכן סיסמה"}</Button>
                </form>
            </main>
        </div>
    );
}
