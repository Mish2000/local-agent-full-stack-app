// src/components/AccountSection.tsx
import * as React from "react";
import { Button } from "@/components/ui/button.tsx";
import { Input } from "@/components/ui/input.tsx";
import { Label } from "@/components/ui/label.tsx";
import PasswordMeter from "@/components/PasswordMeter.tsx";
import { assessPassword } from "@/lib/validators.ts";
import { apiGetProfileSettings, apiUpdateAccount } from "@/lib/api.ts";
import { useAuth } from "@/lib/useAuth.ts";

export default function AccountSection() {
    const { refresh } = useAuth();

    const [loading, setLoading] = React.useState(true);
    const [busy, setBusy] = React.useState(false);
    const [msg, setMsg] = React.useState<string>("");

    const [displayName, setDisplayName] = React.useState("");
    const [initialDisplayName, setInitialDisplayName] = React.useState("");

    const [currentPassword, setCurrentPassword] = React.useState("");
    const [newPassword, setNewPassword] = React.useState("");

    const pwdInfo = React.useMemo(() => assessPassword(newPassword), [newPassword]);
    const pwdValid = newPassword.length === 0 || pwdInfo.level !== "invalid";
    const canSave =
        (!loading &&
            (displayName.trim() !== initialDisplayName.trim() ||
                (newPassword.length > 0 && pwdValid && currentPassword.length > 0))) &&
        !busy;

    React.useEffect(() => {
        let mounted = true;
        (async () => {
            try {
                const s = await apiGetProfileSettings();
                if (!mounted) return;
                const dn = (s.display_name || "").trim();
                setDisplayName(dn);
                setInitialDisplayName(dn);
            } catch {
                // ignore
            } finally {
                if (mounted) setLoading(false);
            }
        })();
        return () => {
            mounted = false;
        };
    }, []);

    const onSave = async (ev: React.FormEvent) => {
        ev.preventDefault();
        if (!canSave) return;

        setBusy(true);
        setMsg("");
        try {
            await apiUpdateAccount({
                display_name:
                    displayName.trim() !== initialDisplayName.trim() ? displayName.trim() : undefined,
                current_password: newPassword ? currentPassword : undefined,
                new_password: newPassword || undefined,
            });

            // Refresh "me" so header, initials, etc. update
            await refresh();

            // Reset password inputs after success
            setCurrentPassword("");
            setNewPassword("");
            setInitialDisplayName(displayName.trim());

            setMsg("נשמר בהצלחה.");
        } catch (e: unknown) {
            setMsg(e instanceof Error ? e.message : "שמירה נכשלה");
        } finally {
            setBusy(false);
        }
    };

    if (loading) {
        return (
            <section className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] shadow-[var(--shadow)] p-4">
                <div className="animate-pulse opacity-70 text-right">טוען…</div>
            </section>
        );
    }

    return (
        <section
            className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] shadow-[var(--shadow)] p-4"
            dir="rtl"
        >
            <h2 className="text-right text-lg font-semibold mb-3">חשבון</h2>

            <form onSubmit={onSave} className="grid gap-4">
                <div>
                    <Label htmlFor="displayName">שם לתצוגה (רשות)</Label>
                    <Input
                        id="displayName"
                        autoComplete="nickname"
                        value={displayName}
                        onChange={(e) => setDisplayName(e.target.value)}
                    />
                </div>

                <div className="grid gap-2">
                    <Label htmlFor="newPassword">סיסמה חדשה</Label>
                    <Input
                        id="newPassword"
                        type="password"
                        autoComplete="new-password"
                        maxLength={128}
                        value={newPassword}
                        onChange={(e) => setNewPassword(e.target.value)}
                    />
                    <PasswordMeter value={newPassword} />
                    <p className="text-xs opacity-75">
                        דרישות מינימום: 8–20 תווים, לפחות אות גדולה אחת ומספר אחד.
                    </p>
                </div>

                <div>
                    <Label htmlFor="currentPassword">סיסמה נוכחית (נדרש בעת שינוי סיסמה)</Label>
                    <Input
                        id="currentPassword"
                        type="password"
                        autoComplete="current-password"
                        maxLength={128}
                        value={currentPassword}
                        onChange={(e) => setCurrentPassword(e.target.value)}
                    />
                </div>

                <div className="flex items-center justify-end gap-3">
                    <Button type="submit" disabled={!canSave} className="rounded-2xl h-10 px-6">
                        {busy ? "שומר…" : "שמירה"}
                    </Button>
                </div>

                {msg && (
                    <div className="text-right text-sm opacity-80" role="status" aria-live="polite">
                        {msg}
                    </div>
                )}
            </form>
        </section>
    );
}
