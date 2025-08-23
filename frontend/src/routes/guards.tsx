import React from "react";
import { Navigate } from "react-router-dom";
import { useAuth } from "@/lib/useAuth";

export function RequireAuth({ children }: { children: React.ReactNode }) {
    const { me, loading } = useAuth();
    if (loading) return <div className="container py-10 text-center">Loading…</div>;
    if (!me) return <Navigate to="/login" replace />;
    return <>{children}</>;
}

export function AnonOnly({ children }: { children: React.ReactNode }) {
    const { me, loading } = useAuth();
    if (loading) return <div className="container py-10 text-center">Loading…</div>;
    if (me) return <Navigate to="/chat" replace />;
    return <>{children}</>;
}
