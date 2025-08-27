// src/components/Header.tsx
import { forwardRef } from "react";
import { useAuth } from "@/lib/useAuth";
import { Button } from "@/components/ui/button";
import { NavLink, useNavigate } from "react-router-dom";
import { type RagMode } from "@/lib/sse";
import { LogOut, LogIn, UserPlus, Sparkles } from "lucide-react";
import StepsToggle from "@/components/StepsToggle";

type Props = {
    ragMode: RagMode;
    onChangeMode: (m: RagMode) => void;
    onNewChat: () => void;
    showModePicker?: boolean;
    onAuthNav?: () => void;
};

const MODE_DESCRIPTIONS: Record<RagMode, string> = {
    auto: "מצב חכם: המערכת בוחרת כלים באופן דינמי.",
    none: "מודל בלבד — ללא כלים (No Tools).",
    dense: "RAG דחוס: שליפה מבוססת אמבדינגס.",
    rerank: "RAG עם Re-Ranker לשיפור הרלוונטיות.",
    web: "חיפוש רשת חי משולב בתשובה.",
};

const Header = forwardRef<HTMLElement, Props>(function Header(
    { ragMode, onChangeMode, onNewChat, showModePicker = true, onAuthNav },
    ref
) {
    const { me, logout } = useAuth();
    const nav = useNavigate();

    return (
        <header ref={ref} className="header shadow-sm">
            <div className="container mx-auto px-4">
                <div className="header-inner py-3">
                    {/* Left: Brand */}
                    <div className="header-left flex items-center gap-2">
                        <Sparkles className="size-5 opacity-80" />
                        <NavLink to="/" className="font-extrabold tracking-tight">
                            Agent 2.0
                        </NavLink>
                    </div>

                    {/* Center: Modes (optional) */}
                    <div className="header-center">
                        {showModePicker && (
                            <div className="flex flex-col items-center">
                                <div className="inline-flex items-center gap-1 p-1 rounded-full border border-[var(--border)] bg-[var(--bg)]">
                                    {(["auto", "none", "dense", "rerank", "web"] as RagMode[]).map((m) => (
                                        <button
                                            key={m}
                                            onClick={() => onChangeMode(m)}
                                            className={[
                                                "px-3 py-1 rounded-full text-sm transition",
                                                ragMode === m
                                                    ? "bg-[var(--panel)] font-semibold border border-[var(--border)]"
                                                    : "hover:bg-[var(--panel)]",
                                            ].join(" ")}
                                            title={m}
                                        >
                                            {m}
                                        </button>
                                    ))}
                                    <Button className="ml-2 rounded-full h-9" onClick={onNewChat} title="שיחה חדשה">
                                        חדשה
                                    </Button>
                                </div>
                                {/* Small description under the bar */}
                                <div className="mt-1 text-[11px] text-center opacity-70 min-h-[14px]">
                                    {MODE_DESCRIPTIONS[ragMode]}
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Right: Auth + steps toggle (same heights for symmetry) */}
                    <div className="header-right flex items-center gap-2">
                        {me ? (
                            <Button
                                variant="outline"
                                className="rounded-full h-9"
                                onClick={async () => {
                                    await logout();
                                    nav("/", { replace: true });
                                }}
                                title="התנתקות"
                            >
                                <LogOut className="size-4 mr-1" />
                                יציאה
                            </Button>
                        ) : (
                            <div className="flex items-center gap-2" onClick={onAuthNav}>
                                <Button variant="outline" className="rounded-full h-9" onClick={() => nav("/login")}>
                                    <LogIn className="size-4 mr-1" />
                                    התחברות
                                </Button>
                                <Button className="rounded-full h-9" onClick={() => nav("/register")}>
                                    <UserPlus className="size-4 mr-1" />
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
