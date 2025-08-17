import { forwardRef } from "react";
import type { RagMode } from "../lib/sse";

type Props = {
    backendStatus: string;
    ragMode: RagMode;
    onChangeMode: (m: RagMode) => void;
};

const Header = forwardRef<HTMLElement, Props>(({ backendStatus, ragMode, onChangeMode }, ref) => {
    return (
        <header ref={ref} className="header">
            <div className="container" style={{ padding: 16, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                    <h1 style={{ fontSize: 22, margin: 0 }}>Local AI Agent — Chat</h1>
                    <p style={{ margin: "6px 0 0 0", opacity: 0.85 }}>
                        Backend health: <b>{backendStatus}</b>
                    </p>
                </div>

                <label style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 14 }}>
                    <span>Context</span>
                    <select
                        value={ragMode}
                        onChange={(e) => onChangeMode(e.target.value as RagMode)}
                        style={{ padding: "6px 8px", borderRadius: 8 }}
                    >
                        <option value="auto">Auto (tools)</option>
                        <option value="none">None</option>
                        <option value="dense">Dense</option>
                        <option value="rerank">+Rerank</option>
                    </select>
                </label>
            </div>
        </header>
    );
});

export default Header;
