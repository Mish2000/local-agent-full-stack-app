import { forwardRef } from "react";

type Props = { backendStatus: string };

// Expose header height to App for layout calculations
const Header = forwardRef<HTMLElement, Props>(({ backendStatus }, ref) => {
    return (
        <header ref={ref} className="header">
            <div className="container" style={{ padding: 16 }}>
                <h1 style={{ fontSize: 22, margin: 0 }}>Local AI Agent — Chat</h1>
                <p style={{ margin: "6px 0 0 0", opacity: 0.85 }}>
                    Backend health: <b>{backendStatus}</b>
                </p>
            </div>
        </header>
    );
});

export default Header;
