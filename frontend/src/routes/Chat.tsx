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
import Sidebar from "@/components/Sidebar";
import {
    createChat,
    deleteChat,
    listChats,
    listMessages,
    renameChat,
    type ChatSummary,
    autoTitle,
} from "@/lib/chats";
import { useAuth } from "@/lib/useAuth";

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
    const { me } = useAuth();

    const [backendStatus, setBackendStatus] = useState<string>("checking...");
    const [messages, setMessages] = useState<Msg[]>([]);
    const [streaming, setStreaming] = useState(false);

    const forcedMode: RagMode | null = isGuest ? "none" : null;
    const [ragMode, setRagMode] = useState<RagMode>(forcedMode ?? "none");

    const [lastSources, setLastSources] = useState<Source[] | null>(null);
    const [lastTools, setLastTools] = useState<ToolEvent[]>([]);
    const [lastTraceId, setLastTraceId] = useState<string | null>(null);

    const [cid, setCid] = useState<string>(() => (isGuest ? makeCid() : ""));

    const [chats, setChats] = useState<ChatSummary[]>([]);
    const [activeChatId, setActiveChatId] = useState<number | null>(null);
    const [loadingChats, setLoadingChats] = useState<boolean>(!isGuest);

    // When we create the chat at first send, we remember it here for auto-rename
    const firstTurnTitlePendingRef = useRef<{ chatId: number | null; firstUserText: string | null }>({
        chatId: null,
        firstUserText: null,
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

    // ------- Backend status -------
    useEffect(() => {
        fetch("http://localhost:8000/healthz")
            .then((r) => r.json())
            .then((j) => setBackendStatus(j.status ?? "unknown"))
            .catch(() => setBackendStatus("backend unreachable"));
    }, []);

    // ------- Layout -------
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

    // ------- Load existing chats (do NOT auto-create; show blank on entry) -------
    useEffect(() => {
        if (isGuest) return;
        let cancelled = false;

        (async () => {
            try {
                setLoadingChats(true);
                const existing = await listChats();
                if (cancelled) return;
                setChats(existing);
                setActiveChatId(null); // fresh blank screen on entry
                setMessages([]);
                setLastSources(null);
                setLastTools([]);
                setLastTraceId(null);
            } catch (e) {
                console.error(e);
                toast.error("Failed to load chats");
            } finally {
                if (!cancelled) setLoadingChats(false);
            }
        })();

        return () => {
            cancelled = true;
        };
    }, [isGuest, me?.id]);

    // --- Load messages for the selected chat (but NEVER during streaming) ---
    useEffect(() => {
        // Guest mode has no server-side messages list
        if (variant !== "full") return;

        // While streaming, we MUST NOT reload, or we’ll erase the assistant placeholder
        if (streaming) return;

        if (!activeChatId) {
            setMessages([]);
            setLastSources(null);
            setLastTools([]);
            setLastTraceId(null);
            return;
        }

        let cancelled = false;
        (async () => {
            try {
                const rows = await listMessages(activeChatId);
                if (cancelled) return;
                const msgs = rows.map<Msg>((r) => ({ role: r.role, content: r.content, dir: detectDir(r.content) }));
                setMessages(msgs);
                setLastSources(null);
                setLastTools([]);
                setLastTraceId(null);
            } catch (e) {
                console.error(e);
                toast.error("Failed to load messages");
            }
        })();

        return () => {
            cancelled = true;
        };
        // IMPORTANT: depend on `streaming` as well
    }, [variant, activeChatId, streaming]);

    // ------- Send message -------
    const handleSend = async (text: string, kbdDir: Dir) => {
        if (streaming) return;

        // For authenticated users: if no active chat yet, create it NOW (first prompt)
        let ensureChatId: number | undefined = activeChatId ?? undefined;
        if (!isGuest && !ensureChatId) {
            try {
                // minimal placeholder; will be auto-titled when first assistant reply completes
                const created = await createChat("…");
                ensureChatId = created.id;
                setChats((prev) => [created, ...prev]);
                setActiveChatId(created.id);
                // mark this chat for auto-title after first assistant message
                firstTurnTitlePendingRef.current = { chatId: created.id, firstUserText: text };
                // eslint-disable-next-line @typescript-eslint/no-unused-vars
            } catch (e) {
                toast.error("Failed to start a new chat");
                return;
            }
        } else {
            // If this is the first message of an already-selected (empty) chat, set pending rename
            if (!isGuest && messages.length === 0 && ensureChatId) {
                firstTurnTitlePendingRef.current = { chatId: ensureChatId, firstUserText: text };
            } else {
                firstTurnTitlePendingRef.current = { chatId: null, firstUserText: null };
            }
        }

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
                // NOTE: onDone takes NO arguments per SSEHandlers; do not destructure anything here.
                onDone: () => {
                    setStreaming(false);

                    // If this was the first turn of a brand-new chat, ask the backend to auto-title it
                    const pending = firstTurnTitlePendingRef.current;
                    if (!isGuest && pending.chatId && pending.firstUserText) {
                        (async () => {
                            try {
                                const res = await autoTitle(pending.chatId!);
                                // reflect the new title in the sidebar
                                setChats((prev) => prev.map((c) => (c.id === res.id ? { ...c, title: res.title } : c)));
                            } catch (e) {
                                console.warn("auto-title failed:", e);
                                // it's okay if this fails; we keep the default name
                            } finally {
                                firstTurnTitlePendingRef.current = { chatId: null, firstUserText: null };
                            }
                        })();
                    }
                },
                onError: (msg) => {
                    setStreaming(false);
                    toast.error(`שגיאת סטרימינג: ${msg}`);
                },
            },
            {
                mode: forcedMode ?? ragMode,
                cid: isGuest ? cid : undefined,
                chatId: !isGuest ? ensureChatId : undefined,
                scope: "user",
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

    const doNewChat = () => {
        // Do NOT create on click. Just show a fresh blank screen.
        if (isGuest) {
            setCid(makeCid());
        }
        setActiveChatId(null);
        setMessages([]);
        setLastSources(null);
        setLastTools([]);
        setLastTraceId(null);
    };

    const doRenameChat = async (id: number) => {
        const current = chats.find((c) => c.id === id);
        const title = prompt("שם חדש לשיחה:", current?.title || "");
        if (!title) return;
        try {
            const updated = await renameChat(id, title);
            setChats((prev) => prev.map((c) => (c.id === id ? updated : c)));
        } catch {
            toast.error("Rename failed");
        }
    };

    const doDeleteChat = async (id: number) => {
        if (!confirm("למחוק את השיחה?")) return;
        try {
            await deleteChat(id);
            setChats((prev) => prev.filter((c) => c.id !== id));
            if (activeChatId === id) {
                setActiveChatId(null);
                setMessages([]);
            }
        } catch {
            toast.error("Delete failed");
        }
    };

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
                onNewChat={doNewChat}
                showModePicker={!isGuest}
                onAuthNav={onAuthNavigate}
            />

            {/* Fixed viewport between header & composer */}
            <div ref={viewportRef} className="chat-viewport">
                <div className="h-full flex">
                    {!isGuest && (
                        <Sidebar
                            items={chats}
                            activeId={activeChatId}
                            onSelect={setActiveChatId}
                            onNew={doNewChat}
                            onRename={doRenameChat}
                            onDelete={doDeleteChat}
                        />
                    )}

                    {/* Messages scroller */}
                    <div ref={chatRef} className="chat flex-1 min-w-0 px-4">
                        {!isGuest && loadingChats && <div className="opacity-70 py-4">טוען היסטוריית שיחות…</div>}

                        {/* Blank state when no chat selected (or new chat draft) */}
                        {!isGuest && !activeChatId && messages.length === 0 && (
                            <div className="w-full h-full grid place-items-center">
                                <div className="text-center opacity-80">
                                    <div className="text-xl font-bold mb-2">שיחה חדשה</div>
                                    <div className="text-sm mb-4">כתבו את ההודעה הראשונה כדי להתחיל</div>
                                </div>
                            </div>
                        )}

                        {messages.map((m, idx) => (
                            <ChatMessage key={idx} id={`msg-${idx}`} role={m.role} content={m.content} />
                        ))}
                        <TracesBar traceId={lastTraceId} />
                        {lastTools.length > 0 && <ToolCalls items={lastTools} />}
                        {lastSources && lastSources.length > 0 && <SourcesBar items={lastSources} />}
                    </div>
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
