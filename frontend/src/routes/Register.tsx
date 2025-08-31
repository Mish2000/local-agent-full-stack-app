// src/routes/Register.tsx
import React, { useMemo, useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { Sparkles } from "lucide-react";
import PasswordMeter from "@/components/PasswordMeter";
import { validateEmail, assessPassword } from "@/lib/validators";
import {APP_NAME} from "@/config/app.ts";

type Me = { id: number; email: string; display_name?: string | null };

export default function Register() {
    const nav = useNavigate();
    const [email, setEmail] = useState("");
    const [displayName, setDisplayName] = useState("");
    const [password, setPassword] = useState("");
    const [busy, setBusy] = useState(false);
    const [touched, setTouched] = useState<{ email?: boolean; password?: boolean }>({});

    const emailOk = useMemo(() => validateEmail(email), [email]);
    const pwdStats = useMemo(() => assessPassword(password), [password]);
    const pwdInvalid = pwdStats.level === "invalid";

    async function submit(e: React.FormEvent) {
        e.preventDefault();
        if (!emailOk) {
            toast.error("Please enter a valid email address (e.g., a@b.com)");
            return;
        }
        if (pwdInvalid) {
            toast.error("Password must be 8–20 chars with at least one uppercase letter and one number");
            return;
        }
        setBusy(true);
        try {
            await api<Me, { email: string; password: string; display_name?: string }>("/auth/register", {
                method: "POST",
                body: { email, password, display_name: displayName || undefined },
            });
            toast.success("Registered! Redirecting to login…");
            nav("/login", { replace: true });
        } catch (err: unknown) {
            const msg = err instanceof Error ? err.message : "Registration failed";
            toast.error(msg);
        } finally {
            setBusy(false);
        }
    }

    return (
        <div className="min-h-dvh bg-[var(--bg)] text-[var(--text)] grid grid-rows-[auto_1fr]">
            <header className="header sticky top-0 z-40">
                <div className="container py-3 flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                        <Sparkles className="size-5 opacity-80" />
                        <h1 className="text-[20px] font-bold leading-tight">{APP_NAME} — הרשמה</h1>
                    </div>
                    <nav className="flex items-center gap-3">
                        <Link to="/" className="text-[15px] font-semibold hover:underline">בית</Link>
                        <Link to="/login" className="text-[15px] font-semibold hover:underline">התחברות</Link>
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
                        <Label htmlFor="displayName">שם לתצוגה (רשות)</Label>
                        <Input
                            id="displayName"
                            autoComplete="nickname"
                            value={displayName}
                            onChange={(e) => setDisplayName(e.target.value)}
                        />
                    </div>

                    <div>
                        <Label htmlFor="password">סיסמה</Label>
                        <Input
                            id="password"
                            type="password"
                            autoComplete="new-password"
                            maxLength={20}
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            onBlur={() => setTouched((t) => ({ ...t, password: true }))}
                            className={pwdInvalid && touched.password ? "ring-2 ring-red-500" : ""}
                        />
                        <div className="mt-2">
                            <PasswordMeter value={password} />
                        </div>
                        <div className="text-xs opacity-75 mt-1">
                            Required: 8–20 chars, at least one uppercase letter and one number.
                        </div>
                    </div>

                    <Button disabled={busy} className="h-11 rounded-2xl">
                        {busy ? "מבצע…" : "הרשמה"}
                    </Button>

                    <div className="text-sm opacity-80 text-right">
                        כבר רשום? <Link to="/login" className="underline">להתחברות</Link>
                    </div>
                </form>
            </main>
        </div>
    );
}
