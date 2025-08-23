// src/routes/Login.tsx
import React, { useState, useMemo } from "react";
import { useNavigate, Link } from "react-router-dom";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { Sparkles } from "lucide-react";
import { validateEmail } from "@/lib/validators";
import { useAuth } from "@/lib/useAuth";

type Me = { id: number; email: string; display_name?: string | null };

export default function Login() {
    const nav = useNavigate();
    const { refresh } = useAuth();
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [busy, setBusy] = useState(false);
    const [touched, setTouched] = useState<{ email?: boolean }>({});

    const emailOk = useMemo(() => validateEmail(email), [email]);

    async function submit(e: React.FormEvent) {
        e.preventDefault();
        if (!emailOk) {
            toast.error("Please enter a valid email address (e.g., a@b.com)");
            return;
        }
        if (!password) {
            toast.error("נא להזין סיסמה");
            return;
        }
        setBusy(true);
        try {
            await api<Me, { email: string; password: string }>("/auth/login", {
                method: "POST",
                body: { email, password },
            });
            await refresh(); // ensure /auth/me is set BEFORE route guard runs
            toast.success("מחובר! מעביר לצ'אט…");
            nav("/chat", { replace: true });
        } catch (err: unknown) {
            const msg = err instanceof Error ? err.message : "שם משתמש או סיסמה שגויים";
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
                        <h1 className="text-[20px] font-bold">Agent 2.0 — התחברות</h1>
                    </div>
                    <nav className="flex items-center gap-3">
                        <Link to="/" className="hover:underline">בית</Link>
                        <Link to="/register" className="hover:underline">הרשמה</Link>
                    </nav>
                </div>
            </header>

            <main className="container grid content-center">
                <form
                    onSubmit={submit}
                    className="w-full max-w-lg place-self-center grid gap-5 rounded-2xl border border-[var(--border)] p-8 bg-[var(--panel)] shadow-[var(--shadow)]"
                >
                    <div>
                        <Label htmlFor="email">אימייל</Label>
                        <Input
                            id="email"
                            type="email"
                            dir="ltr"
                            inputMode="email"
                            autoComplete="email"
                            placeholder="you@example.com"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            onBlur={() => setTouched((t) => ({ ...t, email: true }))}
                            className={!emailOk && touched.email ? "ring-2 ring-red-500" : ""}
                        />
                        {!emailOk && touched.email && (
                            <div className="text-xs text-red-600 mt-1">Invalid email format</div>
                        )}
                    </div>

                    <div>
                        <Label htmlFor="password">סיסמה</Label>
                        <Input
                            id="password"
                            type="password"
                            autoComplete="current-password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                        />
                        <div className="text-xs opacity-80 text-right mt-2">
                            <Link to="/forgot" className="underline">שכחתי סיסמה</Link>
                        </div>
                    </div>

                    <Button disabled={busy} className="h-11 rounded-2xl">
                        {busy ? "מחבר…" : "התחברות"}
                    </Button>

                    <div className="text-sm opacity-80 text-right">
                        חדש פה? <Link to="/register" className="underline">להרשמה</Link>
                    </div>
                </form>
            </main>
        </div>
    );
}
