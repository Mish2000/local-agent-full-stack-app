import { useCallback, useEffect, useRef, useState } from "react";
import { openChatSSE, type RagMode } from "./lib/sse";
import Header from "./components/Header";
import ChatMessage from "./components/ChatMessage";
import Composer from "./components/Composer";
import SourcesBar from "./components/SourcesBar";
import ToolCalls from "./components/ToolCalls";
import { type Dir, detectDir, looksLikeCode } from "./lib/text";

type Msg = { role: "user" | "assistant"; content: string; dir: Dir };
type Source = { id: number; source: string; preview: string; score: number };
type ToolEvent = { name: string; args?: Record<string, never>; result?: never; error?: string };

export default function App() {
    const [backendStatus, setBackendStatus] = useState<string>("checking...");
    const [messages, setMessages] = useState<Msg[]>([]);
    const [streaming, setStreaming] = useState(false);
    const [ragMode, setRagMode] = useState<RagMode>("auto");

    const [lastSources, setLastSources] = useState<Source[] | null>(null);
    const [lastTools, setLastTools] = useState<ToolEvent[]>([]);

    const headerRef = useRef<HTMLElement | null>(null);
    const composerH = useRef<number>(96);
    const headerH = useRef<number>(64);

    const viewportRef = useRef<HTMLDivElement | null>(null);
    const chatRef = useRef<HTMLDivElement | null>(null);
    const esRef = useRef<EventSource | null>(null);

    const [pendingScrollId, setPendingScrollId] = useState<string | null>(null);
    const [showJumpDown, setShowJumpDown] = useState(false);
    const [jumpBtnBottom, setJumpBtnBottom] = useState<number>(84);

    useEffect(() => {
        fetch("http://localhost:8000/healthz")
            .then((r) => r.json())
            .then((j) => setBackendStatus(j.status ?? "unknown"))
            .catch(() => setBackendStatus("backend unreachable"));
    }, []);

    const layoutViewport = useCallback(() => {
        const vp = viewportRef.current;
        if (!vp) return;
        const top = headerH.current;
        const bottom = composerH.current;
        vp.style.top = `${top}px`;
        vp.style.bottom = `${bottom}px`;
        setJumpBtnBottom(Math.max(bottom + 12, 84));
    }, []);

    useEffect(() => {
        const measure = () => {
            if (headerRef.current) {
                headerH.current = headerRef.current.getBoundingClientRect().height;
            }
            layoutViewport();
        };
        measure();
        window.addEventListener("resize", measure);
        return () => window.removeEventListener("resize", measure);
    }, [layoutViewport]);

    const onComposerHeight = (h: number) => {
        composerH.current = h;
        layoutViewport();
    };

    const recomputeBottomState = useCallback(() => {
        const el = chatRef.current;
        if (!el) return;
        const atBottom = el.scrollHeight - (el.scrollTop + el.clientHeight) < 24;
        setShowJumpDown(!atBottom);
    }, []);

    useEffect(() => {
        const el = chatRef.current;
        if (!el) return;
        const onScroll = () => recomputeBottomState();
        el.addEventListener("scroll", onScroll);
        return () => el.removeEventListener("scroll", onScroll);
    }, [recomputeBottomState]);

    useEffect(() => {
        recomputeBottomState();
    }, [messages, recomputeBottomState]);

    useEffect(() => {
        const el = chatRef.current;
        if (!el || typeof ResizeObserver === "undefined") return;
        const ro = new ResizeObserver(() => recomputeBottomState());
        ro.observe(el);
        return () => ro.disconnect();
    }, [recomputeBottomState]);

    const snapMessageToTop = (msgEl: HTMLElement) => {
        msgEl.scrollIntoView({ block: "start", inline: "nearest", behavior: "auto" });
        const scroller = chatRef.current!;
        const padTop = parseFloat(getComputedStyle(scroller).paddingTop || "0");
        scroller.scrollTop = Math.max(0, msgEl.offsetTop - padTop - 1);
    };

    useEffect(() => {
        if (!pendingScrollId) return;
        const scroller = chatRef.current;
        if (!scroller) return;
        requestAnimationFrame(() => {
            const target = document.getElementById(pendingScrollId) as HTMLElement | null;
            if (!target) return;
            snapMessageToTop(target);
            setPendingScrollId(null);
            recomputeBottomState();
        });
    }, [messages.length, pendingScrollId, recomputeBottomState]);

    const handleSend = (text: string, kbdDir: Dir) => {
        if (streaming) return;

        setLastSources(null);
        setLastTools([]);

        const userIdx = messages.length;
        const userMsg: Msg = { role: "user", content: text, dir: kbdDir };
        const assistantMsg: Msg = { role: "assistant", content: "", dir: kbdDir };

        setMessages((prev) => [...prev, userMsg, assistantMsg]);
        setPendingScrollId(`msg-${userIdx}`);
        setStreaming(true);

        esRef.current = openChatSSE(
            text,
            {
                onToken: (t) => {
                    setMessages((prev) => {
                        const next = [...prev];
                        const lastIdx = next.length - 1;
                        if (lastIdx >= 0 && next[lastIdx].role === "assistant") {
                            const combined = next[lastIdx].content + t;
                            const dir: Dir = looksLikeCode(combined) ? "ltr" : detectDir(combined);
                            next[lastIdx] = { ...next[lastIdx], content: combined, dir };
                        }
                        return next;
                    });
                },
                onSources: (arr) => { setLastSources(arr); },
                onTool: (ev) => { setLastTools((prev) => [...prev, ev]); },
                onDone: () => setStreaming(false),
                onError: (msg) => {
                    setStreaming(false);
                    setMessages((prev) => [
                        ...prev,
                        { role: "assistant", content: `שגיאת סטרימינג: ${msg}`, dir: "rtl" },
                    ]);
                },
            },
            { mode: ragMode }
        );
    };

    useEffect(() => () => { if (esRef.current) esRef.current.close(); }, []);

    const jumpToBottom = () => {
        const el = chatRef.current;
        if (!el) return;
        el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    };

    return (
        <div style={{ minHeight: "100vh", display: "grid", gridTemplateRows: "auto 1fr" }}>
            <Header
                ref={headerRef}
                backendStatus={backendStatus}
                ragMode={ragMode}
                onChangeMode={setRagMode}
            />

            <div ref={viewportRef} className="chat-viewport">
                <div ref={chatRef} className="container chat">
                    {messages.map((m, idx) => (
                        <ChatMessage key={idx} id={`msg-${idx}`} role={m.role} content={m.content} />
                    ))}
                    {/* Day-6 panel: tool calls (if any), then citations (if any) */}
                    {lastTools.length > 0 && <ToolCalls items={lastTools} />}
                    {lastSources && lastSources.length > 0 && <SourcesBar items={lastSources} />}
                </div>
            </div>

            {showJumpDown && (
                <button
                    className="jump-down"
                    onClick={jumpToBottom}
                    aria-label="לקפוץ לתגובה העדכנית"
                    style={{ bottom: jumpBtnBottom }}
                >
                    ↓
                </button>
            )}

            <Composer disabled={streaming} onSend={handleSend} onHeightChange={onComposerHeight} />
        </div>
    );
}
