import { forwardRef } from "react";
import type { RagMode } from "@/lib/sse";
import { Select, SelectContent, SelectItem, SelectTrigger } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Sparkles, Plus } from "lucide-react";
import { Link } from "react-router-dom";

type Props = {
    backendStatus: string;
    ragMode: RagMode;
    onChangeMode: (m: RagMode) => void;
    onNewChat: () => void;
};

const modeMeta: Record<RagMode, { label: string; desc: string }> = {
    auto:   { label: "Smart (Auto Tools)", desc: "Balances local files & web when helpful" },
    none:   { label: "No Tools",           desc: "Very fast, best for basic chatting" },
    dense:  { label: "Docs (Dense)",       desc: "Grounded strictly on your indexed files" },
    rerank: { label: "Docs (+Rerank)",     desc: "Adds reranker for higher precision" },
    web:    { label: "Web Search",         desc: "Checks the internet for up-to-date info" },
};

const Header = forwardRef<HTMLElement, Props>(({ backendStatus, ragMode, onChangeMode, onNewChat }, ref) => {
    const meta = modeMeta[ragMode];

    return (
        <header ref={ref} className="header sticky top-0 z-40">
            <div className="container py-3 flex items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                    <Sparkles className="size-5 opacity-80" />
                    <div className="grid">
                        <h1 className="text-[20px] font-bold leading-tight">Local AI Agent — Chat</h1>
                        <div className="text-xs opacity-80">
                            Backend: <b>{backendStatus}</b>
                        </div>
                    </div>
                </div>

                <nav className="hidden md:flex items-center gap-4">
                    <Link to="/" className="text-sm font-semibold hover:underline">בית</Link>
                    <Link to="/login" className="text-sm font-semibold hover:underline">התחברות</Link>
                    <Link to="/register" className="text-sm font-semibold hover:underline">הרשמה</Link>
                </nav>

                <div className="flex items-end gap-3">
                    <div className="grid gap-1 justify-items-end">
                        <Select value={ragMode} onValueChange={(v) => onChangeMode(v as RagMode)}>
                            <SelectTrigger />
                            <SelectContent>
                                <SelectItem value="auto">{modeMeta.auto.label}</SelectItem>
                                <SelectItem value="none">{modeMeta.none.label}</SelectItem>
                                <SelectItem value="dense">{modeMeta.dense.label}</SelectItem>
                                <SelectItem value="rerank">{modeMeta.rerank.label}</SelectItem>
                                <SelectItem value="web">{modeMeta.web.label}</SelectItem>
                            </SelectContent>
                        </Select>
                        <div className="text-[12px] opacity-70 max-w-[360px] text-right">{meta.desc}</div>
                    </div>

                    <Button onClick={onNewChat} className="whitespace-nowrap" title="שיחה חדשה">
                        <Plus className="size-4 ml-1" /> שיחה חדשה
                    </Button>
                </div>
            </div>
        </header>
    );
});
Header.displayName = "Header";
export default Header;
