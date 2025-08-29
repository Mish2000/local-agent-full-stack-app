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
import ExportPDF from "@/components/ExportPDF";
import { usePerChatMode } from "@/lib/usePerChatMode";
import { setChatMode } from "@/lib/modeStorage";
import { saveLastForChat, loadLastForChat, clearLastForChat } from "@/lib/reasoningStorage";
import AttachmentsShelf from "@/components/AttachmentsShelf";
import DragDropOverlay from "@/components/DragDropOverlay";

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

import { stageUpload, listStaged, commitStaged, type StagedItem } from "@/lib/api";

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

    const forcedMode: RagMode | null = isGuest ? "none" : null;

    const [activeChatId, setActiveChatId] = useState<number | null>(null);
    const [ragMode, setRagMode] = usePerChatMode(activeChatId ? String(activeChatId) : null);

    const [lastSources, setLastSources] = useState<Source[] | null>(null);
    const [lastTools, setLastTools] = useState<ToolEvent[]>([]);
    const [lastTraceId, setLastTraceId] = useState<string | null>(null);

    const [cid, setCid] = useState<string>(() => (isGuest ? makeCid() : ""));

    const [chats, setChats] = useState<ChatSummary[]>([]);
    const [loadingChats, setLoadingChats] = useState<boolean>(!isGuest);

    // Auto-title trigger for the very first user turn (text OR files)
    const firstTurnRef = useRef<{ chatId: number | null; shouldAuto: boolean }>({
        chatId: null,
        shouldAuto: false,
    });
    const suppressNextLoadRef = useRef<boolean>(false);

    const headerRef = useRef<HTMLElement | null>(null);
    const composerH = useRef<number>(96);
    const headerH = useRef<number>(64);

    const viewportRef = useRef<HTMLDivElement | null>(null);
    const chatColumnRef = useRef<HTMLDivElement | null>(null); // wrapper (relative)
    const chatRef = useRef<HTMLDivElement | null>(null); // scroller
    const esRef = useRef<EventSource | null>(null);
    const esChatIdRef = useRef<number | null>(null);

    const [pendingScrollId, setPendingScrollId] = useState<string | null>(null);
    const [showJumpDown, setShowJumpDown] = useState(false);
    const [jumpBtnBottom, setJumpBtnBottom] = useState<number>(84);

    const [sourcesByIdx, setSourcesByIdx] = useState<Record<number, Source[]>>({});
    const currentAssistantIndexRef = useRef<number | null>(null);

    // Staged uploads (used for BOTH new & existing chats)
    const [draftId, setDraftId] = useState<string | null>(null);
    const [staged, setStaged] = useState<StagedItem[]>([]);
    const [uploadingP, setUploadingP] = useState<number | null>(null);
    const [filesRefreshKey, setFilesRefreshKey] = useState<number>(0);

    // Global drag/drop → show overlay over the chat column
    const [dragVisible, setDragVisible] = useState<boolean>(false);

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
                // clear staged when switching accounts
                setDraftId(null);
                setStaged([]);
            } catch (e) {
                // eslint-disable-next-line no-console
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

    // --- When active chat changes, hydrate last panels
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
                // eslint-disable-next-line no-console
                console.error(e);
                toast.error("Failed to load messages");
            }
        })();

        return () => {
            cancelled = true;
        };
    }, [variant, activeChatId]);

    // --- Attach last sources to latest assistant message (for Export PDF)
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

    // --- Persist last tool/sources/trace
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

    // ===== Upload / Stage files =====
    const refreshStaged = useCallback(async () => {
        if (!draftId) return;
        try {
            const items = await listStaged(draftId);
            setStaged(items);
        } catch {
            setStaged([]);
        }
    }, [draftId]);

    const onPickFiles = async (files: File[]) => {
        if (streaming) return;
        if (!files || files.length === 0) return;

        setUploadingP(0);
        try {
            const res = await stageUpload(files, draftId ?? undefined, (p) => setUploadingP(p));
            setUploadingP(null);
            setDraftId(res.draft_id);
            setStaged(res.items);
        } catch (e: unknown) {
            setUploadingP(null);
            toast.error(e instanceof Error ? e.message : "Upload failed");
        }
    };

    // Keep latest onPickFiles in a ref for drag/drop handlers
    const onPickFilesRef = useRef(onPickFiles);
    useEffect(() => {
        onPickFilesRef.current = onPickFiles;
    }, [onPickFiles]);

    // Global drag listeners (overlay appears over chat column)
    useEffect(() => {
        const onDragOver = (e: DragEvent) => {
            if (!e.dataTransfer) return;
            const hasFiles = Array.from(e.dataTransfer.items || []).some((i) => i.kind === "file");
            if (hasFiles) {
                e.preventDefault();
                setDragVisible(true);
            }
        };
        const onDrop = () => {
            // If drop happens outside our overlay/chat column, just hide
            setDragVisible(false);
        };
        const onKey = (e: KeyboardEvent) => {
            if (e.key === "Escape") setDragVisible(false);
        };
        const onLeave = (e: DragEvent) => {
            if ((e.relatedTarget as Node | null) === null) setDragVisible(false);
        };

        window.addEventListener("dragover", onDragOver);
        window.addEventListener("drop", onDrop);
        window.addEventListener("keyup", onKey);
        window.addEventListener("dragleave", onLeave);
        return () => {
            window.removeEventListener("dragover", onDragOver);
            window.removeEventListener("drop", onDrop);
            window.removeEventListener("keyup", onKey);
            window.removeEventListener("dragleave", onLeave);
        };
    }, []);

    // ------- Send message -------
    const handleSend = async (text: string, kbdDir: Dir) => {
        if (streaming) return;

        let ensureChatId: number | undefined = activeChatId ?? undefined;

        const hasStaged = staged.length > 0;
        const textToSend = text.trim(); // no default prompt injection

        // Create the chat if needed
        if (!isGuest && !ensureChatId) {
            try {
                const created = await createChat("…");
                suppressNextLoadRef.current = true;
                ensureChatId = created.id;
                setChats((prev) => [created, ...prev]);
                setActiveChatId(created.id);
                setChatMode(String(created.id), forcedMode ?? ragMode);
                // Trigger auto-title if first turn has text OR files
                firstTurnRef.current = { chatId: created.id, shouldAuto: (textToSend.length > 0) || hasStaged };
            } catch {
                toast.error("Failed to start a new chat");
                return;
            }
        } else {
            if (!isGuest && messages.length === 0 && ensureChatId) {
                firstTurnRef.current = { chatId: ensureChatId, shouldAuto: (textToSend.length > 0) || hasStaged };
            } else {
                firstTurnRef.current = { chatId: null, shouldAuto: false };
            }
        }

        // Commit staged files BEFORE streaming
        if (hasStaged && ensureChatId) {
            try {
                const res = await commitStaged(draftId!, ensureChatId);
                if (res.count > 0) setFilesRefreshKey((v) => v + 1);
                setDraftId(null);
                setStaged([]);
            } catch (e: unknown) {
                toast.error(e instanceof Error ? e.message : "Failed to attach files");
                return;
            }
        }

        // Reset last turn panels
        setLastSources(null);
        setLastTools([]);
        setLastTraceId(null);
        if (ensureChatId) clearLastForChat(ensureChatId);

        const assistantIdx = messages.length + 1;
        currentAssistantIndexRef.current = assistantIdx;

        const userMsg: Msg = { role: "user", content: textToSend, dir: kbdDir };
        const assistantMsg: Msg = { role: "assistant", content: "", dir: kbdDir };
        setMessages((prev) => [...prev, userMsg, assistantMsg]);
        setPendingScrollId(`msg-${assistantIdx}`);
        setStreaming(true);

        setSourcesByIdx((prev) => {
            const next = { ...prev };
            delete next[assistantIdx];
            return next;
        });

        esChatIdRef.current = !isGuest ? (ensureChatId ?? null) : null;

        const scopeForStream: "user" | "chat" = ensureChatId ? "chat" : "user";

        openChatSSE(
            textToSend,
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
                    const ft = firstTurnRef.current;
                    if (!isGuest && ft.chatId && ft.shouldAuto) {
                        (async () => {
                            try {
                                const res = await autoTitle(ft.chatId!);
                                setChats((prev) => prev.map((c) => (c.id === res.id ? { ...c, title: res.title } : c)));
                            } catch (e) {
                                // eslint-disable-next-line no-console
                                console.warn("auto-title failed:", e);
                            } finally {
                                firstTurnRef.current = { chatId: null, shouldAuto: false };
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
                scope: scopeForStream,
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
        setDraftId(null);
        setStaged([]);
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

    const canSendEmpty = staged.length > 0;

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

                    {/* Chat column (relative to host overlay) */}
                    <div ref={chatColumnRef} className="relative flex-1 min-w-0">
                        {/* Messages scroller */}
                        <div ref={chatRef} className="chat px-4">
                            {!isGuest && loadingChats && <div className="opacity-70 py-4">טוען היסטוריית שיחות…</div>}

                            {/* Left attachments shelf */}
                            <AttachmentsShelf
                                chatId={activeChatId}
                                draftId={draftId}
                                staged={staged}
                                onRefreshStaged={refreshStaged}
                                refreshKey={filesRefreshKey}
                            />

                            {/* Blank state */}
                            {!isGuest && !activeChatId && messages.length === 0 && (
                                <div className="w-full h-full grid place-items-center">
                                    <div className="text-center opacity-80">
                                        <div className="text-xl font-bold mb-2">שיחה חדשה</div>
                                        <div className="text-sm mb-4">כתוב את ההודעה הראשונה כדי להתחיל</div>
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

                        {/* Drag & Drop overlay sits above the chat column only */}
                        <DragDropOverlay
                            visible={dragVisible}
                            onDropFiles={(files) => onPickFilesRef.current(files)}
                            onCancel={() => setDragVisible(false)}
                        />
                    </div>
                </div>
            </div>

            {/* Jump-to-bottom */}
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

            {/* Fixed composer with Upload + Export inside it */}
            <Composer
                disabled={streaming}
                onSend={handleSend}
                onHeightChange={onComposerHeight}
                canSendEmpty={canSendEmpty}
                onPickFiles={onPickFiles}
                uploadingProgress={uploadingP}
                rightActions={
                    <ExportPDF
                        chatTitle={activeTitle}
                        messages={messages.map((m) => ({ role: m.role, content: m.content }))}
                        sourcesByIndex={sourcesByIdx}
                    />
                }
            />
        </div>
    );
}
