import React, { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Ctx, type Me, type AuthCtx } from "./auth-context";

export default function AuthProvider({ children }: { children: React.ReactNode }) {
    const [me, setMe] = useState<Me>(null);
    const [loading, setLoading] = useState(true);

    const refresh = useCallback(async () => {
        setLoading(true);
        try {
            const who = await api<Me>("/auth/me");
            setMe(who);
        } catch {
            setMe(null);
        } finally {
            setLoading(false);
        }
    }, []);

    const logout = useCallback<AuthCtx["logout"]>(async () => {
        try {
            await api<{ ok: true }>("/auth/logout", { method: "POST" });
        } finally {
            setMe(null);
        }
    }, []);

    useEffect(() => { void refresh(); }, [refresh]);

    return <Ctx.Provider value={{ me, loading, refresh, logout }}>{children}</Ctx.Provider>;
}
