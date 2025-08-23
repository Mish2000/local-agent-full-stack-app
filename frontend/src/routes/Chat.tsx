// src/routes/Chat.tsx
import { useCallback, useEffect, useRef, useState } from "react";
import { openChatSSE, type RagMode, type Source, type ToolEvent } from "@/lib/sse";
import Header from "@/components/Header";
import ChatMessage from "@/components/ChatMessage";
import Composer from "@/components/Composer";
import SourcesBar from "@/components/SourcesBar";
import ToolCalls from "@/components/ToolCalls";
import TracesBar from "@/components/TracesBar";
import { type Dir, detectDir, looksLikeCode } from "@/lib/text";
import { ArrowDown } from "lucide-react";
import { toast } from "sonner";

type Msg = { role: "user" | "assistant"; content: string; dir: Dir };
type Variant = "guest" | "full";

function makeCid(): string {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
        return (crypto as Crypto).randomUUID();
    }
    return Math.random().toString(36).slice(2, 10);
}

export default function Chat({ variant = "full" }: { variant?: Variant }) {
    const isGuest = variant === "guest";

    const [backendStatus, setBackendStatus] = useState<string>("checking...");
    const [messages, setMessages] = useState<Msg[]>([]);
    const [streaming, setStreaming] = useState(false);

    // In guest, we force No Tools; in full we start with "auto".
    const forcedMode: RagMode | null = isGuest ? "none" : null;
    const [ragMode, setRagMode] = useState<RagMode>(forcedMode ?? "auto");

    const [lastSources, setLastSources] = useState<Source[] | null>(null);
    const [lastTools, setLastTools] = useState<ToolEvent[]>([]);
    const [lastTraceId, setLastTraceId] = useState<string | null>(null);

    // Conversation id:
    // - guest: NOT persisted (new on every refresh)
    // - full : persisted in localStorage
    const [cid, setCid] = useState<string>(() => {
        if (isGuest) return makeCid();
        const k = "chat_cid";
        const existing = localStorage.getItem(k);
        if (existing) return existing;
        const fresh = makeCid();
        localStorage.setItem(k, fresh);
        return fresh;
    });

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
        setLastTraceId(null);

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
                onSources: (arr) => setLastSources(arr),
                onTool: (ev) => setLastTools((prev) => [...prev, ev]),
                onTrace: (id) => setLastTraceId(id),
                onDone: () => setStreaming(false),
                onError: (msg) => {
                    setStreaming(false);
                    toast.error(`שגיאת סטרימינג: ${msg}`);
                },
            },
            {
                mode: forcedMode ?? ragMode,
                cid,
            }
        );
    };

    useEffect(() => {
        return () => {
            if (esRef.current) esRef.current.close();
        };
    }, []);

    const jumpToBottom = () => {
        const el = chatRef.current;
        if (!el) return;
        el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    };

    const clearBackendCid = async (id: string) => {
        try {
            await fetch(`http://localhost:8000/chat/clear?cid=${encodeURIComponent(id)}`, { method: "POST" });
        } catch {
            /* ignore */
        }
    };

    const newChat = async () => {
        if (isGuest) {
            setCid(makeCid());
        } else {
            await clearBackendCid(cid);
            const fresh = makeCid();
            setCid(fresh);
            localStorage.setItem("chat_cid", fresh);
        }
        setMessages([]);
        setLastSources(null);
        setLastTools([]);
        setLastTraceId(null);
    };

    // If user clicks Login/Register in header while in guest mode, clear guest memory immediately.
    const onAuthNavigate = async () => {
        if (isGuest && cid) {
            await clearBackendCid(cid);
            setMessages([]);
            setLastSources(null);
            setLastTools([]);
            setLastTraceId(null);
        }
    };

    return (
        <div className="min-h-dvh">
            <Header
                ref={headerRef}
                backendStatus={backendStatus}
                ragMode={forcedMode ?? ragMode}
                onChangeMode={setRagMode}
                onNewChat={newChat}
                showModePicker={!isGuest}
                onAuthNav={onAuthNavigate}
            />

            {/* Fixed viewport between header & composer */}
            <div ref={viewportRef} className="chat-viewport">
                <div ref={chatRef} className="container chat">
                    {messages.map((m, idx) => (
                        <ChatMessage key={idx} id={`msg-${idx}`} role={m.role} content={m.content} />
                    ))}
                    <TracesBar traceId={lastTraceId} />
                    {lastTools.length > 0 && <ToolCalls items={lastTools} />}
                    {lastSources && lastSources.length > 0 && <SourcesBar items={lastSources} />}
                </div>
            </div>

            {/* Jump-to-bottom floating button */}
            {showJumpDown && (
                <button
                    className="jump-down"
                    onClick={jumpToBottom}
                    aria-label="לקפוץ לתגובה העדכנית"
                    style={{ bottom: jumpBtnBottom }}
                    title="לתחתית"
                >
                    <ArrowDown className="size-5" />
                </button>
            )}

            {/* Fixed composer */}
            <Composer disabled={streaming} onSend={handleSend} onHeightChange={onComposerHeight} />
        </div>
    );
}
