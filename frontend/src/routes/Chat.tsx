// src/routes/Chat.tsx
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
import UploadDocs from "@/components/UploadDocs";
import ExportPDF from "@/components/ExportPDF";
import { usePerChatMode } from "@/lib/usePerChatMode";
import { setChatMode } from "@/lib/modeStorage";
import { saveLastForChat, loadLastForChat, clearLastForChat } from "@/lib/reasoningStorage";

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

    const [messages, setMessages] = useState<Msg[]>([]);
    const [streaming, setStreaming] = useState(false);

    // Guest is hard-forced to 'none'; authenticated users persist per-chat mode.
    const forcedMode: RagMode | null = isGuest ? "none" : null;

    // Per-conversation mode (persisted in localStorage by chat id).
    const [activeChatId, setActiveChatId] = useState<number | null>(null);
    const [ragMode, setRagMode] = usePerChatMode(activeChatId ? String(activeChatId) : null);

    const [lastSources, setLastSources] = useState<Source[] | null>(null);
    const [lastTools, setLastTools] = useState<ToolEvent[]>([]);
    const [lastTraceId, setLastTraceId] = useState<string | null>(null);

    const [cid, setCid] = useState<string>(() => (isGuest ? makeCid() : ""));

    const [chats, setChats] = useState<ChatSummary[]>([]);
    const [loadingChats, setLoadingChats] = useState<boolean>(!isGuest);

    const firstTurnTitlePendingRef = useRef<{ chatId: number | null; firstUserText: string | null }>({
        chatId: null,
        firstUserText: null,
    });
    const suppressNextLoadRef = useRef<boolean>(false);

    const headerRef = useRef<HTMLElement | null>(null);
    const composerH = useRef<number>(96);
    const headerH = useRef<number>(64);

    const viewportRef = useRef<HTMLDivElement | null>(null);
    const chatRef = useRef<HTMLDivElement | null>(null);
    const esRef = useRef<EventSource | null>(null);
    /** Track which chat the current SSE belongs to (prevents auto-closing the just-opened stream). */
    const esChatIdRef = useRef<number | null>(null);

    const [pendingScrollId, setPendingScrollId] = useState<string | null>(null);
    const [showJumpDown, setShowJumpDown] = useState(false);
    const [jumpBtnBottom, setJumpBtnBottom] = useState<number>(84);

    // Sources per assistant message index (used for PDF export)
    const [sourcesByIdx, setSourcesByIdx] = useState<Record<number, Source[]>>({});
    const currentAssistantIndexRef = useRef<number | null>(null);

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

    // ------- Load existing chats -------
    useEffect(() => {
        if (isGuest) return;
        let cancelled = false;
        (async () => {
            try {
                setLoadingChats(true);
                const existing = await listChats();
                if (cancelled) return;
                setChats(existing);
                setActiveChatId(null);
                setMessages([]);
                setLastSources(null);
                setLastTools([]);
                setLastTraceId(null);
                setSourcesByIdx({});
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

    // --- When the active chat changes, hydrate last steps/sources/trace.
    //     Only close an existing SSE if it's for a DIFFERENT chat.
    useEffect(() => {
        if (esRef.current && esChatIdRef.current !== activeChatId) {
            esRef.current.close();
            esRef.current = null;
            setStreaming(false);
        }

        if (!activeChatId) {
            setLastSources(null);
            setLastTools([]);
            setLastTraceId(null);
            setSourcesByIdx({});
            return;
        }

        const cached = loadLastForChat(activeChatId);
        setLastTools(cached?.tools ?? []);
        setLastSources((cached?.sources?.length ?? 0) ? cached!.sources : null);
        setLastTraceId(cached?.traceId ?? null);
        setSourcesByIdx({});
    }, [activeChatId]);

    // --- Load messages for the selected chat ---
    useEffect(() => {
        if (variant !== "full") return;

        if (!activeChatId) {
            setMessages([]);
            return;
        }

        if (suppressNextLoadRef.current) {
            suppressNextLoadRef.current = false;
            return;
        }

        let cancelled = false;
        (async () => {
            try {
                const rows = await listMessages(activeChatId);
                if (cancelled) return;
                const msgs = rows.map<Msg>((r) => ({
                    role: r.role,
                    content: r.content,
                    dir: detectDir(r.content),
                }));
                setMessages(msgs);
            } catch (e) {
                console.error(e);
                toast.error("Failed to load messages");
            }
        })();

        return () => {
            cancelled = true;
        };
    }, [variant, activeChatId]);

    // --- After messages change, attach cached sources to the latest assistant msg (for Export PDF)
    useEffect(() => {
        if (!activeChatId) return;
        if (!lastSources || lastSources.length === 0) return;
        if (!messages.length) return;

        let lastAssistantIdx: number | null = null;
        for (let i = messages.length - 1; i >= 0; i--) {
            if (messages[i].role === "assistant") {
                lastAssistantIdx = i;
                break;
            }
        }
        if (lastAssistantIdx == null) return;

        setSourcesByIdx((prev) => {
            if (prev[lastAssistantIdx!] && prev[lastAssistantIdx!].length) return prev;
            return { ...prev, [lastAssistantIdx!]: lastSources };
        });
    }, [messages, lastSources, activeChatId]);

    // --- Persist last steps/sources/trace changes to sessionStorage (per chat)
    useEffect(() => {
        if (!activeChatId) return;
        if (lastTools.length === 0 && !lastSources && !lastTraceId) {
            clearLastForChat(activeChatId);
            return;
        }
        saveLastForChat(activeChatId, {
            tools: lastTools,
            sources: lastSources ?? [],
            traceId: lastTraceId ?? null,
        });
    }, [activeChatId, lastTools, lastSources, lastTraceId]);

    // ------- Send message -------
    const handleSend = async (text: string, kbdDir: Dir) => {
        if (streaming) return;

        let ensureChatId: number | undefined = activeChatId ?? undefined;
        if (!isGuest && !ensureChatId) {
            try {
                // Create the chat only on first user message of the session
                const created = await createChat("…");
                suppressNextLoadRef.current = true;
                ensureChatId = created.id;
                setChats((prev) => [created, ...prev]);
                setActiveChatId(created.id);

                // Persist the *currently selected* mode for this new chat (fixes auto-reset-to-none bug).
                setChatMode(String(created.id), forcedMode ?? ragMode);

                firstTurnTitlePendingRef.current = { chatId: created.id, firstUserText: text };
            } catch {
                toast.error("Failed to start a new chat");
                return;
            }
        } else {
            if (!isGuest && messages.length === 0 && ensureChatId) {
                firstTurnTitlePendingRef.current = { chatId: ensureChatId, firstUserText: text };
            } else {
                firstTurnTitlePendingRef.current = { chatId: null, firstUserText: null };
            }
        }

        // Clear last turn panels immediately for this chat to avoid stale display.
        setLastSources(null);
        setLastTools([]);
        setLastTraceId(null);
        if (ensureChatId) clearLastForChat(ensureChatId);

        const assistantIdx = messages.length + 1;
        currentAssistantIndexRef.current = assistantIdx;

        const userMsg: Msg = { role: "user", content: text, dir: kbdDir };
        const assistantMsg: Msg = { role: "assistant", content: "", dir: kbdDir };
        setMessages((prev) => [...prev, userMsg, assistantMsg]);
        setPendingScrollId(`msg-${assistantIdx}`);
        setStreaming(true);

        setSourcesByIdx((prev) => {
            const next = { ...prev };
            delete next[assistantIdx];
            return next;
        });

        // Mark which chat the SSE belongs to (so the activeChatId effect won't kill it).
        esChatIdRef.current = !isGuest ? (ensureChatId ?? null) : null;

        esRef.current = openChatSSE(
            text,
            {
                onToken: (t) => {
                    setMessages((prev) => {
                        const next = [...prev];
                        const last = next.length - 1;
                        if (last >= 0 && next[last].role === "assistant") {
                            const combined = next[last].content + t;
                            const dir: Dir = looksLikeCode(combined) ? "ltr" : detectDir(combined);
                            next[last] = { ...next[last], content: combined, dir };
                        }
                        return next;
                    });
                },
                onSources: (arr) => {
                    setLastSources(arr);
                    const idx = currentAssistantIndexRef.current;
                    if (typeof idx === "number") {
                        setSourcesByIdx((prev) => ({ ...prev, [idx]: arr }));
                    }
                },
                onTool: (ev) => setLastTools((prev) => [...prev, ev]),
                onTrace: (id) => setLastTraceId(id),
                onDone: () => {
                    setStreaming(false);
                    const pending = firstTurnTitlePendingRef.current;
                    if (!isGuest && pending.chatId && pending.firstUserText) {
                        (async () => {
                            try {
                                const res = await autoTitle(pending.chatId!);
                                setChats((prev) => prev.map((c) => (c.id === res.id ? { ...c, title: res.title } : c)));
                            } catch (e) {
                                console.warn("auto-title failed:", e);
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
            esChatIdRef.current = null;
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

    const doNewChat = useCallback(() => {
        if (isGuest) {
            setCid(makeCid());
        }
        setActiveChatId(null);
        setMessages([]);
        setLastSources(null);
        setLastTools([]);
        setLastTraceId(null);
        setSourcesByIdx({});
        // Note: no chat id yet -> mode naturally shows 'none' via usePerChatMode(null)
    }, [isGuest]);

    useEffect(() => {
        const onKey = (e: KeyboardEvent) => {
            if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
                e.preventDefault();
                doNewChat();
            }
        };
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, [doNewChat]);

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
                setSourcesByIdx({});
                setLastSources(null);
                setLastTools([]);
                setLastTraceId(null);
                clearLastForChat(id);
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
            setSourcesByIdx({});
        }
    };

    const activeTitle = useMemo(
        () => (!isGuest ? chats.find((c) => c.id === activeChatId)?.title ?? "שיחה" : "שיחה"),
        [chats, activeChatId, isGuest]
    );

    return (
        <div className="min-h-dvh">
            <Header
                ref={headerRef}
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
                        {/* sticky tools row at top */}
                        <div className="sticky top-2 z-10 flex justify-end gap-2 pr-1 pb-2">
                            {!isGuest && <UploadDocs chatId={activeChatId} />}
                            <ExportPDF
                                chatTitle={activeTitle}
                                messages={messages.map((m) => ({ role: m.role, content: m.content }))}
                                sourcesByIndex={sourcesByIdx}
                            />
                        </div>

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

                        {/* Trace + Reasoning steps + Sources are inside the scroller */}
                        <TracesBar traceId={lastTraceId} />
                        {lastTools.length > 0 && <ToolCalls items={lastTools} />}
                        {lastSources && lastSources.length > 0 && <SourcesBar items={lastSources} />}
                    </div>
                </div>
            </div>

            {/* Jump-to-bottom floating button (CSS places it on the left) */}
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
