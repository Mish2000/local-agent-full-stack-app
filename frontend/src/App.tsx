import { useCallback, useEffect, useRef, useState } from "react";
import { openChatSSE } from "./lib/sse";
import Header from "./components/Header";
import ChatMessage from "./components/ChatMessage";
import Composer from "./components/Composer";
import {type Dir, detectDir, looksLikeCode } from "./lib/text";

type Msg = { role: "user" | "assistant"; content: string; dir: Dir };

export default function App() {
    const [backendStatus, setBackendStatus] = useState<string>("checking...");
    const [messages, setMessages] = useState<Msg[]>([]);
    const [streaming, setStreaming] = useState(false);

    // Layout refs + state
    const headerRef = useRef<HTMLElement | null>(null);
    const composerH = useRef<number>(96);
    const headerH = useRef<number>(64);

    const viewportRef = useRef<HTMLDivElement | null>(null); // fixed region
    const chatRef = useRef<HTMLDivElement | null>(null);     // inner scroller
    const esRef = useRef<EventSource | null>(null);

    const [pendingScrollId, setPendingScrollId] = useState<string | null>(null);
    const [showJumpDown, setShowJumpDown] = useState(false);
    const [jumpBtnBottom, setJumpBtnBottom] = useState<number>(84);

    // Health check once
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

    // Measure header on mount + resize
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

    // Bottom-state (for “down arrow”)
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

    useEffect(() => { recomputeBottomState(); }, [messages, recomputeBottomState]);

    useEffect(() => {
        const el = chatRef.current;
        if (!el || typeof ResizeObserver === "undefined") return;
        const ro = new ResizeObserver(() => recomputeBottomState());
        ro.observe(el);
        return () => ro.disconnect();
    }, [recomputeBottomState]);

    // Helper: snap a message to the very top of the chat scroller
    const snapMessageToTop = (msgEl: HTMLElement) => {
        // First, use scroll-snap to lock it at the top reliably
        msgEl.scrollIntoView({ block: "start", inline: "nearest", behavior: "auto" });
        // Tiny nudge so the previous message is fully out of view on all browsers
        const scroller = chatRef.current!;
        const padTop = parseFloat(getComputedStyle(scroller).paddingTop || "0");
        scroller.scrollTop = Math.max(0, msgEl.offsetTop - padTop - 1);
    };

    // After messages render, if there's a pending target, snap it
    useEffect(() => {
        if (!pendingScrollId) return;
        const scroller = chatRef.current;
        if (!scroller) return;
        // Let React paint both the new user bubble and the assistant placeholder
        requestAnimationFrame(() => {
            const target = document.getElementById(pendingScrollId) as HTMLElement | null;
            if (!target) return;
            snapMessageToTop(target);
            setPendingScrollId(null);
            recomputeBottomState();
        });
    }, [messages, pendingScrollId, recomputeBottomState]);

    // Send flow
    const handleSend = (text: string, kbdDir: Dir) => {
        if (streaming) return;

        const userIdx = messages.length; // index of the new user message
        const userMsg: Msg = { role: "user", content: text, dir: kbdDir };
        const assistantMsg: Msg = { role: "assistant", content: "", dir: kbdDir };

        setMessages((prev) => [...prev, userMsg, assistantMsg]);
        setPendingScrollId(`msg-${userIdx}`); // snap user's bubble to the very top
        setStreaming(true);

        esRef.current = openChatSSE(text, {
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
            onDone: () => setStreaming(false),
            onError: (msg) => {
                setStreaming(false);
                setMessages((prev) => [
                    ...prev,
                    { role: "assistant", content: `שגיאת סטרימינג: ${msg}`, dir: "rtl" },
                ]);
            },
        });
    };

    useEffect(() => {
        return () => { if (esRef.current) esRef.current.close(); };
    }, []);

    const jumpToBottom = () => {
        const el = chatRef.current;
        if (!el) return;
        el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    };

    return (
        <div style={{ minHeight: "100vh", display: "grid", gridTemplateRows: "auto 1fr" }}>
            <Header ref={headerRef} backendStatus={backendStatus} />

            {/* Fixed chat viewport */}
            <div ref={viewportRef} className="chat-viewport">
                <div ref={chatRef} className="container chat">
                    {messages.map((m, idx) => (
                        <ChatMessage key={idx} id={`msg-${idx}`} role={m.role} content={m.content} />
                    ))}
                </div>
            </div>

            {/* Floating “jump to latest” */}
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

            {/* Fixed composer */}
            <Composer disabled={streaming} onSend={handleSend} onHeightChange={onComposerHeight} />
        </div>
    );
}
