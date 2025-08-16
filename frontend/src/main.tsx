import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles.css";

// Initial default: RTL (Hebrew-first)
document.documentElement.setAttribute("dir", "rtl");
document.documentElement.setAttribute("lang", "he");

ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
        <App />
    </React.StrictMode>
);
