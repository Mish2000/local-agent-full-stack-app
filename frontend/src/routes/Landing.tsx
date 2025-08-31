// src/routes/Landing.tsx
import { Link } from "react-router-dom";
import { Sparkles, LockKeyholeOpen, FileText, Globe2, Gauge } from "lucide-react";
import { Button } from "@/components/ui/button";
import React from "react";
import { APP_NAME } from "@/config/app";

export default function Landing() {
    return (
        <div className="min-h-dvh bg-[var(--bg)] text-[var(--text)] grid grid-rows-[auto_1fr]">
            <header className="header sticky top-0 z-40">
                <div className="container py-3 flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                        <Sparkles className="size-5 opacity-80" />
                        <h1 className="text-[20px] font-bold leading-tight">{APP_NAME}</h1>
                    </div>
                    <nav className="flex items-center gap-3">
                        <Link to="/login" className="text-[15px] font-semibold hover:underline">התחברות</Link>
                        <Link to="/register" className="text-[15px] font-semibold hover:underline">הרשמה</Link>
                    </nav>
                </div>
            </header>

            <main className="container grid content-center py-8">
                <section className="grid gap-8">
                    <div className="grid gap-4 text-center">
                        <h2 className="text-4xl font-extrabold">סוכן מקומי חכם — מותאם בדיוק אליכם</h2>
                        <p className="text-lg opacity-90 max-w-3xl mx-auto">
                            בחרו מצב עבודה לכל שיחה: לא מקוון, חיפוש ברשת או אוטומטי. צרו הנחיה אישית גלובלית
                            וצרפו קבצים ישירות בתוך השיחה לקבלת תוצאות מותאמות ומדויקות
                        </p>
                    </div>

                    <div className="grid sm:grid-cols-3 gap-5">
                        <Feature
                            icon={<FileText className="size-6" />}
                            title="מצב לא מקוון"
                            text="ללא חיפוש ברשת — משתמש בזיכרון שיחה, הנחייה אישית ובמסמכים שתצרפו לשיחה"
                        />
                        <Feature
                            icon={<Globe2 className="size-6" />}
                            title="חיפוש רשת"
                            text="מבצע חיפוש עדכני בכל פנייה לקבלת מידע מעודכן ממקורות גלויים"
                        />
                        <Feature
                            icon={<Gauge className="size-6" />}
                            title="מצב אוטומטי"
                            text="הסוכן מחליט לבד אם לבצע חיפוש - תלוי בתוכן הקלט "
                        />
                    </div>

                    <div className="grid sm:grid-cols-3 gap-5">
                        <Feature
                            icon={<LockKeyholeOpen className="size-6" />}
                            title="פרטיות ושליטה"
                            text="שליטה מלאה: הנחיה אישית גלובלית לכל השיחות שניתן לכבות בהגדרות, וצירוף קבצים מקומיים בכל שיחה "
                        />
                        <Feature
                            icon={<Sparkles className="size-6" />}
                            title="התאמה אישית"
                            text="העלו תמונת פרופיל לבחירתכם מהמחשב, או בחרו רקע מתוך המאגר הקיים"
                        />
                        <Feature
                            icon={<FileText className="size-6" />}
                            title="צירוף קבצים"
                            text="צרפו מסמכים לשיחה בגרירה או העלאה, והסוכן יתחשב בהם בתשובתו"
                        />
                    </div>

                    <div className="flex flex-wrap gap-3 justify-center">
                        <Link to="/guest"><Button className="h-11 px-6 rounded-2xl">נסה ללא הרשמה !</Button></Link>
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
