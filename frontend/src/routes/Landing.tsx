import { Link } from "react-router-dom";
import { Sparkles, LockKeyholeOpen, History, Wand2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import React from "react";

export default function Landing() {
    return (
        <div className="min-h-dvh bg-[var(--bg)] text-[var(--text)] grid grid-rows-[auto_1fr]">
            <header className="header sticky top-0 z-40">
                <div className="container py-3 flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                        <Sparkles className="size-5 opacity-80" />
                        <h1 className="text-[20px] font-bold leading-tight">Agent 2.0</h1>
                    </div>
                    <nav className="flex items-center gap-3">
                        <Link to="/login" className="text-sm font-semibold hover:underline">התחברות</Link>
                        <Link to="/register" className="text-sm font-semibold hover:underline">הרשמה</Link>
                    </nav>
                </div>
            </header>

            <main className="container grid content-center py-8">
                <section className="grid gap-8">
                    <div className="grid gap-4 text-center">
                        <h2 className="text-4xl font-extrabold">ברוכים הבאים ל-Agent 2.0</h2>
                        <p className="text-lg opacity-90 max-w-3xl mx-auto">
                            מערכת עוזר חכם עם RAG, חיפוש ווב וכלים — ניתן לנסות מיד ללא הרשמה במצב No Tools עם זיכרון זמני בלבד.
                        </p>
                    </div>

                    <div className="grid sm:grid-cols-3 gap-5">
                        <Feature
                            icon={<Wand2 className="size-6" />}
                            title="שיחה מהירה — ללא כלים"
                            text="נסו מיד, ללא הרשמה. מצב No Tools מהיר ומבוסס זיכרון זמני."
                        />
                        <Feature
                            icon={<History className="size-6" />}
                            title="היסטוריה וקבצים (לאחר התחברות)"
                            text="קבלו היסטוריית שיחות ו-RAG על המסמכים שלכם."
                        />
                        <Feature
                            icon={<LockKeyholeOpen className="size-6" />}
                            title="פרטיות ושליטה"
                            text="שיחות אורח נמחקות ברענון או במעבר להתחברות/הרשמה."
                        />
                    </div>

                    <div className="flex flex-wrap gap-3 justify-center">
                        <Link to="/guest"><Button className="h-11 px-6 rounded-2xl">התחילו בלי הרשמה</Button></Link>
                        <Link to="/login"><Button variant="outline" className="h-11 px-6 rounded-2xl">להתחברות</Button></Link>
                        <Link to="/register"><Button variant="outline" className="h-11 px-6 rounded-2xl">להרשמה</Button></Link>
                    </div>
                </section>
            </main>
        </div>
    );
}

function Feature({ icon, title, text }: { icon: React.ReactNode; title: string; text: string }) {
    return (
        <div className="rounded-2xl border border-[var(--border)] bg-[var(--panel)] p-6 shadow-[var(--shadow)] grid place-items-center text-center gap-2">
            <div className="opacity-90">{icon}</div>
            <div className="text-base font-bold">{title}</div>
            <p className="text-sm opacity-90">{text}</p>
        </div>
    );
}
