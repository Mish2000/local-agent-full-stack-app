import { createContext } from "react";

export type Me = { id: number; email: string; display_name?: string | null } | null;

export type AuthCtx = {
    me: Me;
    loading: boolean;
    refresh: () => Promise<void>;
    logout: () => Promise<void>;
};

export const Ctx = createContext<AuthCtx | null>(null);
