// src/routes/Chat.tsx
import {useCallback, useEffect, useMemo, useRef, useState} from "react";
import {openChatSSE, type RagMode, type Source, type ToolEvent} from "@/lib/sse";
import Header from "@/features/layout/Header.tsx";
import ChatMessage from "@/features/chat/ChatMessage.tsx";
import Composer from "@/features/chat/Composer.tsx";
import SourcesBar from "@/features/chat/SourcesBar.tsx";
import ToolCalls from "@/features/chat/ToolCalls.tsx";
import TracesBar from "@/features/chat/TracesBar.tsx";
import {type Dir, detectDir, looksLikeCode} from "@/lib/text";
import {ArrowDown} from "lucide-react";
import {toast} from "sonner";
import Sidebar from "@/features/layout/Sidebar.tsx";
import ExportPDF from "@/features/chat/ExportPDF.tsx";
import {saveLastForChat, loadLastForChat, clearLastForChat} from "@/lib/reasoningStorage";
import AttachmentsShelf from "@/features/chat/AttachmentsShelf.tsx";
import DragDropOverlay from "@/components/DragDropOverlay";
import {useChatMode} from "@/lib/modes";
import ConfirmDialog from "@/components/ConfirmDialog";

import {
    createChat,
    deleteChat,
    listChats,
    listMessages,
    type ChatSummary,
    autoTitle,
    renameChat,
} from "@/lib/chats";
import {useAuth} from "@/lib/useAuth";

import {stageUpload, listStaged, commitStaged, type StagedItem} from "@/lib/api";

type Msg = { role: "user" | "assistant"; content: string; dir: Dir };
type Variant = "guest" | "full";

function makeCid(): string {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
        return (crypto as Crypto).randomUUID();
    }
    return Math.random().toString(36).slice(2, 10);
}

// Ensure only valid backend modes ever go out
function validMode(m: string): RagMode {
    return (m === "offline" || m === "web" || m === "auto" ? m : "offline") as RagMode;
}

export default function Chat({variant = "full"}: { variant?: Variant }) {
    const isGuest = variant === "guest";
    const {me} = useAuth();

    const [messages, setMessages] = useState<Msg[]>([]);
    const [streaming, setStreaming] = useState(false);

    // Force a valid default mode for guests (backend expects one of: offline|web|auto)
    const forcedMode: RagMode | null = isGuest ? "offline" : null;

    const [activeChatId, setActiveChatId] = useState<number | null>(null);

    // New per-chat mode hook (defaults to "offline" and persists; never yields "none")
    const {mode: ragMode, setMode: setRagMode} = useChatMode(activeChatId ?? undefined);

    const [lastSources, setLastSources] = useState<Source[] | null>(null);
    const [lastTools, setLastTools] = useState<ToolEvent[]>([]);
    const [lastTraceId, setLastTraceId] = useState<string | null>(null);

    const [cid, setCid] = useState<string>(() => (isGuest ? makeCid() : ""));

    const [chats, setChats] = useState<ChatSummary[]>([]);
    const [loadingChats, setLoadingChats] = useState<boolean>(!isGuest);

    const [pendingDeleteId, setPendingDeleteId] = useState<number | null>(null);


    // Auto-title trigger for the very first user turn (text OR files)
    const firstTurnRef = useRef<{ chatId: number | null; shouldAuto: boolean; firstText: string }>({
        chatId: null,
        shouldAuto: false,
        firstText: "",
    });

    function firstTurnTitleSeed(text: string, staged: StagedItem[]): string {
        const t = (text ?? "").trim();
        if (t) return t;

        if (Array.isArray(staged) && staged.length > 0) {
            // Derive a short, human-friendly seed from up to 3 filenames
            const pretty = staged
                .slice(0, 3)
                .map((s) => String(s?.filename ?? "")
                    .replace(/\.[^.]+$/, "")         // drop extension
                    .replace(/[-_]+/g, " ")          // dashes/underscores -> spaces
                    .replace(/\s+/g, " ")            // collapse spaces
                    .trim())
                .filter((name) => name.length > 0);

            if (pretty.length > 0) return pretty.join(", ");
        }
        return "";
    }

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

    // Attachments are disabled for guests and in "web" mode
    const attachmentsEnabled = !isGuest && ragMode !== "web";

    const streamingRef = useRef<boolean>(false);
    useEffect(() => {
        streamingRef.current = streaming;
    }, [streaming]);

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
        msgEl.scrollIntoView({block: "start", inline: "nearest", behavior: "auto"});
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

    // --- When active chat changes, hydrate last panels & manage SSE lifecycle
    useEffect(() => {
        // If there is an open SSE that belongs to a DIFFERENT chat, close it.
        // Important guard: do not close if we haven't tagged the SSE with a chat yet.
        if (esRef.current && esChatIdRef.current != null && esChatIdRef.current !== activeChatId) {
            try {
                esRef.current.close();
            } catch {
                /* ignore */
            }
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
                setMessages((prev) => {
                    const hasAssistantPlaceholder =
                        prev.length > 0 &&
                        prev[prev.length - 1].role === "assistant" &&
                        prev[prev.length - 1].content === "";

                    if (streamingRef.current || hasAssistantPlaceholder) {
                        // keep optimistic state; background load is ignored once
                        return prev;
                    }
                    return msgs;
                });
            } catch (e) {
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
            return {...prev, [lastAssistantIdx!]: lastSources};
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

    const onPickFiles = useCallback(async (files: File[]) => {
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
    }, [streaming, draftId]);


    // Keep latest onPickFiles in a ref for drag/drop handlers
    const onPickFilesRef = useRef(onPickFiles);
    useEffect(() => {
        onPickFilesRef.current = onPickFiles;
    }, [onPickFiles]);

    // Global drag listeners (overlay appears over chat column)
    useEffect(() => {
        const onDragOver = (e: DragEvent) => {
            if (isGuest) return; // ⟵ block for guest
            if (!e.dataTransfer) return;
            const hasFiles = Array.from(e.dataTransfer.items || []).some((i) => i.kind === "file");
            if (hasFiles) {
                e.preventDefault();
                setDragVisible(true);
            }
        };
        const onDrop = () => {
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
    }, [isGuest]);

    // ------- Send message -------
    // REPLACE the whole function in src/routes/Chat.tsx
    const handleSend = async (text: string, kbdDir: Dir) => {
        if (streaming) return;

        let ensureChatId: number | undefined = activeChatId ?? undefined;

        const hasStaged = staged.length > 0;
        const textToSend = text.trim();

        // Create the chat if needed
        if (!isGuest && !ensureChatId) {
            try {
                const created = await createChat("…");
                suppressNextLoadRef.current = true;
                ensureChatId = created.id;
                setChats((prev) => [created, ...prev]);

                // Persist current mode for the new chat
                try {
                    localStorage.setItem(`agent-mode:${created.id}`, (forcedMode ?? ragMode) as RagMode);
                } catch {
                    /* ignore */
                }

                // IMPORTANT: tag the upcoming SSE with this chat id to avoid the close-on-change race
                esChatIdRef.current = created.id;

                setActiveChatId(created.id);

                // First-turn auto-title trigger
                firstTurnRef.current = {
                    chatId: created.id,
                    shouldAuto: textToSend.length > 0 || hasStaged,
                    firstText: firstTurnTitleSeed(textToSend, staged),
                };
            } catch {
                toast.error("Failed to start a new chat");
                return;
            }
        } else {
            if (!isGuest && messages.length === 0 && ensureChatId) {
                firstTurnRef.current = {
                    chatId: ensureChatId,
                    shouldAuto: textToSend.length > 0 || hasStaged,
                    firstText: firstTurnTitleSeed(textToSend, staged),
                };
            } else {
                firstTurnRef.current = { chatId: null, shouldAuto: false, firstText: "" };
            }
        }

        // Commit staged files BEFORE streaming
        try {
            if (draftId && staged.length) {
                await commitStaged(draftId, ensureChatId);
                setDraftId(null);
                setStaged([]);
                setFilesRefreshKey((v) => v + 1);
                refreshStaged();
            }
        } catch {
            toast.error("Failed to attach files");
            return;
        }

        // Optimistic messages
        setMessages((prev) => [
            ...prev,
            { role: "user", content: textToSend, dir: kbdDir },
            { role: "assistant", content: "", dir: kbdDir },
        ]);

        // Reset side panels
        setLastSources(null);
        setLastTools([]);
        setLastTraceId(null);
        if (ensureChatId) clearLastForChat(ensureChatId);

        const scopeForStream: "user" | "chat" = ensureChatId ? "chat" : "user";
        const sendMode: RagMode = validMode(forcedMode ?? ragMode);

        // --- Open SSE ---
        setStreaming(true);

        // Tag the SSE with its owning chat *for any* chat (new or existing)
        esChatIdRef.current = ensureChatId ?? null;

        const es = openChatSSE(
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
                onSources: (srcs) => setLastSources(srcs),
                onTool: (ev: ToolEvent) => setLastTools((prev) => [...prev, ev]),
                onTrace: (traceId) => setLastTraceId(traceId || null),
                onError: (msg) => {
                    setStreaming(false);
                    toast.error(`שגיאת סטרימינג: ${msg}`);
                },
                onDone: () => {
                    setStreaming(false);

                    // First-turn auto-title (if required)
                    const ft = firstTurnRef.current;
                    if (ft.shouldAuto && ft.chatId) {
                        (async () => {
                            try {
                                const res = await autoTitle(ft.chatId!, ft.firstText);
                                setChats((prev) => prev.map((c) => (c.id === res.id ? { ...c, title: res.title } : c)));
                            } catch (e) {
                                // eslint-disable-next-line no-console
                                console.warn("auto-title failed:", e);
                            } finally {
                                firstTurnRef.current = { chatId: null, shouldAuto: false, firstText: "" };
                            }
                        })();
                    }
                },
            },
            {
                mode: sendMode,
                cid: isGuest ? cid : undefined,
                chatId: !isGuest ? ensureChatId : undefined,
                scope: scopeForStream,
            }
        );

        esRef.current = es;
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
        el.scrollTo({top: el.scrollHeight, behavior: "smooth"});
    };

    const clearBackendCid = async (id: string) => {
        try {
            await fetch(`http://localhost:8000/chat/clear?cid=${encodeURIComponent(id)}`, {method: "POST"});
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
        // Mode is not reset to "none" — useChatMode will keep/persist per-chat and default to "offline" for new.
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

    const doDeleteChat = (id: number) => {
        setPendingDeleteId(id);
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
                        {/* Static attachments shelf (pinned) */}
                        {attachmentsEnabled && (
                            <div className="block absolute left-4 top-3 z-20">
                                <AttachmentsShelf
                                    chatId={activeChatId}
                                    draftId={draftId}
                                    staged={staged}
                                    onRefreshStaged={refreshStaged}
                                    refreshKey={filesRefreshKey}
                                />
                            </div>
                        )}

                        {/* Messages scroller */}
                        <div ref={chatRef} className="chat px-4">
                            {!isGuest && loadingChats && <div className="opacity-70 py-4">טוען היסטוריית שיחות…</div>}

                            {/* Blank state */}
                            {!isGuest && !activeChatId && messages.length === 0 && (
                                <div className="w-full h-full grid place-items-center">
                                    <div className="text-center">
                                        <div className="text-xl font-bold mb-2 ui-title-strong">שיחה חדשה</div>
                                        <div className="text-sm opacity-80 ui-title-normal">כתוב את ההודעה הראשונה כדי
                                            להתחיל
                                        </div>
                                    </div>
                                </div>
                            )}

                            {messages.map((m, idx) => (
                                <ChatMessage key={idx} id={`msg-${idx}`} role={m.role} content={m.content}/>
                            ))}

                            <TracesBar traceId={lastTraceId}/>
                            {lastTools.length > 0 && <ToolCalls items={lastTools}/>}
                            {lastSources && lastSources.length > 0 && <SourcesBar items={lastSources}/>}
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
                    style={{bottom: jumpBtnBottom}}
                    title="לתחתית"
                >
                    <ArrowDown className="size-5"/>
                </button>
            )}

            {/* Composer — hide paperclip + Export for guest */}
            <Composer
                disabled={streaming}
                onSend={handleSend}
                onHeightChange={onComposerHeight}
                canSendEmpty={canSendEmpty}
                onPickFiles={onPickFiles}
                uploadingProgress={uploadingP}
                showAttach={attachmentsEnabled}
                rightActions={
                    !isGuest ? (
                        <ExportPDF
                            chatTitle={activeTitle}
                            messages={messages.map((m) => ({role: m.role, content: m.content}))}
                            sourcesByIndex={sourcesByIdx}
                        />
                    ) : null
                }
            />

            <ConfirmDialog
                open={pendingDeleteId !== null}
                dir="rtl"
                title="למחוק את השיחה?"
                message="הפעולה תמחק לצמיתות את כל התוכן כולל נתונים משויכים. לא יתאפשר שחזור."
                confirmText="מחק שיחה"
                cancelText="ביטול"
                destructive
                onCancel={() => setPendingDeleteId(null)}
                onConfirm={async () => {
                    if (pendingDeleteId == null) return;
                    const id = pendingDeleteId;

                    // Close the dialog immediately for a snappy feel
                    setPendingDeleteId(null);

                    // If an SSE is streaming for this chat, close it to avoid orphaned streams
                    if (esRef.current && esChatIdRef.current === id) {
                        try { esRef.current.close(); } catch { /* ignore */ }
                        esRef.current = null;
                        setStreaming(false);
                    }

                    try {
                        await deleteChat(id);
                        setChats(prev => prev.filter(c => c.id !== id));

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
                }}
            />

        </div>
    );
}
