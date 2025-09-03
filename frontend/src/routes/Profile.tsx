// src/routes/Profile.tsx
import {useEffect, useMemo, useRef, useState} from "react";
import {
    apiGetProfileSettings,
    apiPutProfileSettings,
    apiUploadAvatar,
    type ProfileSettings,
    avatarUrl,
} from "@/lib/api";
import {useAuth} from "@/lib/useAuth";
import {Button} from "@/components/ui/button";
import {Link, useNavigate} from "react-router-dom";
import AccountSection from "@/features/layout/AccountSection.tsx";


type Preset = { id: string; label: string; grad: string };

const PRESETS: Preset[] = [
    {id: "sys-1", label: "כחול", grad: "from-indigo-500 to-sky-500"},
    {id: "sys-2", label: "סגול/ורוד", grad: "from-fuchsia-500 to-pink-500"},
    {id: "sys-3", label: "ירקרק", grad: "from-emerald-500 to-teal-500"},
    {id: "sys-4", label: "כתום", grad: "from-amber-500 to-orange-500"},
    {id: "sys-5", label: "סגול", grad: "from-violet-500 to-purple-500"},
    {id: "sys-6", label: "אדום", grad: "from-rose-500 to-red-500"},
    {id: "sys-7", label: "ליים/ירוק", grad: "from-lime-500 to-green-500"},
    {id: "sys-8", label: "אפור/שחור", grad: "from-slate-500 to-zinc-700"},
];

export default function Profile() {
    useNavigate();
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [msg, setMsg] = useState<string>("");

    const [form, setForm] = useState<ProfileSettings>({
        instruction_enabled: true,
        instruction_text: "",
        avatar_kind: "",
        avatar_value: "",
    });

    const [previewUrl, setPreviewUrl] = useState<string>("");
    const [showPresets, setShowPresets] = useState<boolean>(false);
    const fileRef = useRef<HTMLInputElement | null>(null);

    useEffect(() => {
        let mounted = true;
        (async () => {
            try {
                const s = await apiGetProfileSettings();
                if (mounted) {
                    setForm({
                        instruction_enabled: !!s.instruction_enabled,
                        instruction_text: s.instruction_text ?? "",
                        avatar_kind: s.avatar_kind ?? "",
                        avatar_value: s.avatar_value ?? "",
                    });
                }
            } catch {
                /* ignore */
            } finally {
                if (mounted) setLoading(false);
            }
        })();
        return () => {
            mounted = false;
            if (previewUrl) URL.revokeObjectURL(previewUrl);
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);


    const onSave = async () => {
        setSaving(true);
        setMsg("");
        try {
            await apiPutProfileSettings({
                instruction_enabled: !!form.instruction_enabled,
                instruction_text: form.instruction_text || "",
                avatar_kind: form.avatar_kind || "",
                avatar_value: form.avatar_value || "",
            });
            await refresh();
            setMsg("נשמר בהצלחה.");
            // If using system avatar or after upload, drop any object URL and reload from server
            if (previewUrl) {
                URL.revokeObjectURL(previewUrl);
                setPreviewUrl("");
            }
        } catch (e: unknown) {
            setMsg(e instanceof Error ? e.message : "שמירה נכשלה");
        } finally {
            setSaving(false);
        }
    };

    const onPickSystemAvatar = (id: string) => {
        setForm((f) => ({...f, avatar_kind: "system", avatar_value: id}));
        setPreviewUrl("");
        setShowPresets(true);
    };

    const onUploadClick = () => fileRef.current?.click();

    const onUploadFile = async (ev: React.ChangeEvent<HTMLInputElement>) => {
        const f = ev.target.files?.[0];
        if (!f) return;
        setMsg("");
        try {
            const url = URL.createObjectURL(f);
            setPreviewUrl(url);
            await apiUploadAvatar(f);
            setForm((prev) => ({...prev, avatar_kind: "upload", avatar_value: ""}));
            setMsg("התמונה הועלתה.");
        } catch (e: unknown) {
            setMsg(e instanceof Error ? e.message : "העלאה נכשלה");
            setPreviewUrl("");
        } finally {
            ev.target.value = "";
        }
    };

    const { me, refresh } = useAuth();


    const avatarNode = useMemo(() => {
        const initial =
            ((me?.display_name || me?.email || "").trim().charAt(0) || "?").toUpperCase();

        if (form.avatar_kind === "upload") {
            const src = previewUrl || avatarUrl(true);
            return (
                <img
                    src={src}
                    alt="תמונת פרופיל"
                    className="size-20 rounded-full object-cover"
                />
            );
        }
        if (form.avatar_kind === "system" && form.avatar_value) {
            const preset = PRESETS.find((a) => a.id === form.avatar_value) ?? PRESETS[0];
            return (
                <div
                    className={`size-20 rounded-full bg-gradient-to-br ${preset.grad} grid place-items-center text-white`}
                    title={me?.display_name || ""}
                    aria-label={`Avatar ${initial}`}
                >
                    <span className="text-xl font-semibold select-none">{initial}</span>
                </div>
            );
        }
        return (
            <div className="size-20 rounded-full bg-neutral-200 dark:bg-neutral-800 grid place-items-center">
                <span className="text-lg opacity-70">?</span>
            </div>
        );
    }, [form.avatar_kind, form.avatar_value, previewUrl, me?.display_name, me?.email]);


    if (loading) {
        return (
            <div className="container mx-auto p-4">
                <div className="animate-pulse opacity-70">טוען…</div>
            </div>
        );
    }

    return (
        <div className="container mx-auto px-4" dir="rtl">
            {/* Title at the top-right */}
            <h1 className="text-2xl font-semibold text-right mt-6 mb-4">פרופיל והגדרות</h1>

            {/* Account card — centered under the two cards */}
            <div className="max-w-5xl mx-auto my-5">
                <AccountSection/>
            </div>

            {/* Centered grid: Avatar + Personal instruction */}
            <div className="grid gap-5 md:grid-cols-2 max-w-5xl mx-auto">
                {/* Avatar card */}
                <section
                    className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] shadow-[var(--shadow)] p-4">
                    <h2 className="text-right text-lg font-semibold mb-3">תמונת פרופיל</h2>
                    <div className="flex items-center gap-4">
                        {avatarNode}
                        <div className="flex-1">
                            <div className="mt-2 flex flex-wrap items-center gap-2">
                                <Button onClick={onUploadClick} className="rounded-2xl h-9 px-3">
                                    העלאה מהמחשב
                                </Button>
                                <input
                                    ref={fileRef}
                                    type="file"
                                    accept="image/png,image/jpeg,image/webp"
                                    className="hidden"
                                    onChange={onUploadFile}
                                />
                                <Button
                                    variant="outline"
                                    onClick={() => setShowPresets((v) => !v)}
                                    className="rounded-2xl h-9 px-3"
                                    title="בחרו אווטאר מוכן"
                                >
                                    רקעים מוכנים
                                </Button>
                                {form.avatar_kind ? (
                                    <Button
                                        variant="outline"
                                        onClick={() => {
                                            setForm((f) => ({...f, avatar_kind: "", avatar_value: ""}));
                                            if (previewUrl) {
                                                URL.revokeObjectURL(previewUrl);
                                                setPreviewUrl("");
                                            }
                                        }}
                                        className="rounded-2xl h-9 px-3"
                                    >
                                        הסר תמונה
                                    </Button>
                                ) : null}
                            </div>
                        </div>
                    </div>

                    {/* Presets bar — hidden by default; perfectly circular tiles */}
                    {showPresets && (
                        <div className="mt-4">
                            <div className="text-sm opacity-80 mb-2">בחרו רקע מוכן</div>
                            <div className="grid grid-cols-4 sm:grid-cols-8 gap-3">
                                {PRESETS.map((p) => {
                                    const active = form.avatar_kind === "system" && form.avatar_value === p.id;
                                    return (
                                        <button
                                            key={p.id}
                                            type="button"
                                            onClick={() => onPickSystemAvatar(p.id)}
                                            className={[
                                                "relative h-16 w-16 rounded-full ring-2 transition",
                                                active ? "ring-indigo-400" : "ring-[var(--border)] hover:ring-neutral-400 dark:hover:ring-neutral-600",
                                            ].join(" ")}
                                            title={p.label}
                                            aria-label={p.label}
                                        >
                      <span
                          className={`absolute inset-0 rounded-full bg-gradient-to-br ${p.grad}`}
                          aria-hidden
                      />
                                        </button>
                                    );
                                })}
                            </div>
                        </div>
                    )}
                </section>

                {/* Personal instruction card */}
                <section
                    className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] shadow-[var(--shadow)] p-4">
                    <h2 className="text-right text-lg font-semibold mb-3">הנחייה אישית</h2>

                    <label className="flex items-center justify-end gap-2 mb-2">
                        <span>הפעל הנחייה אישית</span>
                        <input
                            type="checkbox"
                            checked={!!form.instruction_enabled}
                            onChange={(e) =>
                                setForm((f) => ({...f, instruction_enabled: e.target.checked}))
                            }
                        />
                    </label>

                    <textarea
                        className="min-h-[180px] w-full rounded-2xl border border-[var(--border)] bg-[var(--bg)] p-3 text-right"
                        placeholder="כאן תוכלו לכתוב הנחיות כלליות שיחולו על כל השיחות. לדוגמה, פנה אליי בשם מיכאל וענה על הכל בסגנון הומוריסטי."
                        value={form.instruction_text}
                        onChange={(e) =>
                            setForm((f) => ({...f, instruction_text: e.target.value}))
                        }
                    />
                </section>
            </div>

            {/* Bottom actions — Save + Back side-by-side and centered */}
            <div className="mt-8 mb-10 flex items-center justify-center gap-4">
                <Button
                    type="button"
                    disabled={saving}
                    onClick={onSave}
                    className="rounded-2xl h-10 px-6"
                    title="שמור"
                >
                    {saving ? "שומר…" : "שמור"}
                </Button>
                <Link to="/chat" className="no-underline">
                    <Button variant="outline" className="rounded-2xl h-10 px-6">
                        חזרה למסך השיחות
                    </Button>
                </Link>
            </div>

            {msg && (
                <div className="max-w-5xl mx-auto text-center text-sm opacity-80 mb-6" role="status">
                    {msg}
                </div>
            )}
        </div>
    );
}
