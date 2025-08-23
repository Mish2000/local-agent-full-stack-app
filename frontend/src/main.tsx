import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider, Navigate } from "react-router-dom";
import "./index.css";
import { Toaster } from "sonner";
import AuthProvider from "@/lib/auth-provider";
import { RequireAuth, AnonOnly } from "@/routes/guards";

import Landing from "./routes/Landing";
import Chat from "./routes/Chat";
import Login from "./routes/Login";
import Register from "./routes/Register";
import Forgot from "./routes/Forgot";
import ResetPassword from "./routes/ResetPassword";

const router = createBrowserRouter([
    { path: "/", element: <AnonOnly><Landing /></AnonOnly> },
    { path: "/guest", element: <Chat variant="guest" /> },
    { path: "/chat", element: <RequireAuth><Chat variant="full" /></RequireAuth> },
    { path: "/login", element: <AnonOnly><Login /></AnonOnly> },
    { path: "/register", element: <AnonOnly><Register /></AnonOnly> },
    { path: "/forgot", element: <AnonOnly><Forgot /></AnonOnly> },
    { path: "/reset", element: <AnonOnly><ResetPassword /></AnonOnly> },
    { path: "*", element: <Navigate to="/" replace /> },
]);

document.documentElement.setAttribute("dir", "rtl");
document.documentElement.setAttribute("lang", "he");

ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
        <Toaster richColors position="top-right" />
        <AuthProvider>
            <RouterProvider router={router} />
        </AuthProvider>
    </React.StrictMode>
);
