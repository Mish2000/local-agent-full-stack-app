// src/components/Header.tsx
import { forwardRef, useEffect, useMemo, useState } from "react";
import { useAuth } from "@/lib/useAuth.ts";
import { Button } from "@/components/ui/button.tsx";
import { NavLink, useNavigate } from "react-router-dom";
import { type RagMode } from "@/lib/sse.ts";
import { LogOut, LogIn, UserPlus, Sparkles } from "lucide-react";
import StepsToggle from "@/features/chat/StepsToggle.tsx";
import { APP_NAME } from "@/config/app.ts";
import { avatarUrl } from "@/lib/api.ts";


type Props = {
    ragMode: RagMode;
    onChangeMode: (m: RagMode) => void;
    onNewChat: () => void;
    showModePicker?: boolean;
    /** Optional callback used by Chat screen to clear guest cid before auth nav */
    onAuthNav?: () => void | Promise<void>;
};

const MODE_LABELS: Record<"offline" | "web" | "auto", string> = {
    offline: "לא מקוון",
    web: "חיפוש ברשת",
    auto: "אוטומטי",
};

const MODE_DESCRIPTIONS: Record<"offline" | "web" | "auto", string> = {
    offline: "קבצים מקומיים והנחיה אישית. בלי אינטרנט.",
    web: "תמיד מחפש ברשת לכל פנייה.",
    auto: "מחליט לבד אם צריך חיפוש.",
};

const Header = forwardRef<HTMLElement, Props>(function Header(
    { ragMode, onChangeMode, onNewChat, showModePicker = true, onAuthNav },
    _ref
) {
    const { me, logout } = useAuth();
    const nav = useNavigate();


    const [avatarSrc, setAvatarSrc] = useState<string | null>(null);
    useEffect(() => {
        // Try loading user's avatar from backend; if it 404s we'll show initials bubble.
        const src = avatarUrl(true);
        setAvatarSrc(src);
    }, [me?.id]);

    // Refresh avatar whenever profile broadcasts an update (system preset or upload)
    useEffect(() => {
        const onStorage = (e: StorageEvent) => {
            if (e.key === "avatar:v") setAvatarSrc(avatarUrl(true));
        };
        const onCustom = () => setAvatarSrc(avatarUrl(true));

        window.addEventListener("storage", onStorage);
        window.addEventListener("avatar:updated", onCustom as EventListener);
        return () => {
            window.removeEventListener("storage", onStorage);
            window.removeEventListener("avatar:updated", onCustom as EventListener);
        };
    }, []);


    const initials = useMemo(() => {
        const src = (me?.display_name || me?.email || "").trim();
        if (!src) return "?";
        const parts = src.split(/\s+/);
        const a = parts[0]?.[0] || "";
        const b = parts[1]?.[0] || "";
        return (a + b).toUpperCase() || a.toUpperCase() || "?";
    }, [me]);

    const displayName = useMemo(() => {
        if (me?.display_name && me.display_name.trim()) return me.display_name.trim();
        const email = me?.email || "";
        return email.split("@")[0] || "פרופיל שלי";
    }, [me]);

    const ModePill = ({ id }: { id: "offline" | "web" | "auto" }) => (
        <button
            type="button"
            onClick={() => onChangeMode(id as unknown as RagMode)}
            className={[
                "px-3 py-1 rounded-full text-sm transition",
                ragMode === id ? "bg-white dark:bg-neutral-700 shadow" : "opacity-70 hover:opacity-100",
            ].join(" ")}
        >
            {MODE_LABELS[id]}
        </button>
    );

    return (
        <header className="header sticky top-0 z-30">
            <div className="container mx-auto px-4">
                <div className="h-16 flex items-center justify-between gap-6">
                    {/* Left: Brand + New chat */}
                    <div className="flex items-center gap-3">
                        <NavLink to={me ? "/chat" : "/"} className="flex items-center gap-2">
                            <Sparkles className="size-5 text-indigo-600" />
                            <span className="text-[20px] font-bold">{APP_NAME}</span>
                        </NavLink>
                        <Button variant="outline" className="rounded-2xl h-9 px-4" onClick={onNewChat}>
                            שיחה חדשה
                        </Button>
                    </div>

                    {/* Middle: Modes — drop a few “mm” with mt-1 and align baseline */}
                    <div className="flex-1 flex justify-center">
                        {showModePicker && (
                            <div className="flex flex-col items-center mt-1">
                                <div className="flex items-center gap-1 bg-neutral-100 dark:bg-neutral-800 p-1 rounded-full">
                                    <ModePill id="offline" />
                                    <ModePill id="web" />
                                    <ModePill id="auto" />
                                </div>
                                <div className="mt-1 text-[11px] text-center opacity-70 min-h-[14px]">
                                    {MODE_DESCRIPTIONS[(["offline", "web", "auto"] as const).includes(ragMode as never) ? (ragMode as "offline" | "web" | "auto") : "auto"]}
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Right: Auth + Profile + Steps */}
                    <div className="flex items-center gap-3">
                        {me ? (
                            <NavLink
                                to="/profile/settings"
                                className="group flex items-center gap-2 rounded-full px-2.5 py-1.5 hover:bg-neutral-100 dark:hover:bg-neutral-800"
                                title="פרופיל והגדרות"
                            >
                                {avatarSrc ? (
                                    <img
                                        src={avatarSrc}
                                        alt="avatar"
                                        className="size-10 rounded-full object-cover ring-2 ring-white/10"
                                        onError={() => setAvatarSrc(null)}
                                    />
                                ) : (
                                    <div
                                        aria-label="Avatar"
                                        className="size-10 rounded-full bg-gradient-to-br from-indigo-500 to-sky-500 text-white grid place-items-center font-semibold"
                                    >
                                        {initials}
                                    </div>
                                )}
                                <span className="hidden sm:block text-sm opacity-90 group-hover:opacity-100">{displayName}</span>
                            </NavLink>
                        ) : null}

                        {me ? (
                            <Button
                                variant="outline"
                                className="rounded-2xl h-9"
                                onClick={async () => {
                                    await logout();
                                    nav("/", { replace: true });
                                }}
                                title="התנתקות"
                            >
                                <LogOut className="size-4 ml-1" />
                                יציאה
                            </Button>
                        ) : (
                            <div className="flex items-center gap-2">
                                <Button
                                    variant="outline"
                                    className="rounded-2xl h-9"
                                    onClick={async () => {
                                        await onAuthNav?.();
                                        nav("/login");
                                    }}
                                >
                                    <LogIn className="size-4 ml-1" />
                                    התחברות
                                </Button>
                                <Button
                                    className="rounded-2xl h-9"
                                    onClick={async () => {
                                        await onAuthNav?.();
                                        nav("/register");
                                    }}
                                >
                                    <UserPlus className="size-4 ml-1" />
                                    הרשמה
                                </Button>
                            </div>
                        )}

                        <StepsToggle />
                    </div>
                </div>
            </div>
        </header>
    );
});

export default Header;
