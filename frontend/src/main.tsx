import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import App from "./App";
import "./index.css";
import { Toaster } from "sonner";
import Login from "./routes/Login";
import Register from "./routes/Register";

const router = createBrowserRouter([
    { path: "/", element: <App /> },
    { path: "/login", element: <Login /> },
    { path: "/register", element: <Register /> },
]);

// Default to RTL (Hebrew-first)
document.documentElement.setAttribute("dir", "rtl");
document.documentElement.setAttribute("lang", "he");

ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
        <Toaster richColors position="top-right" />
        <RouterProvider router={router} />
    </React.StrictMode>
);
