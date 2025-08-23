import { detectDir, type Dir } from "@/lib/text";

type Props = {
    id: string;
    role: "user" | "assistant";
    content: string;
};

export default function ChatMessage({ id, role, content }: Props) {
    const dir: Dir = detectDir(content);

    if (role === "user") {
        return (
            <div id={id} className="msg bubble bubble-user" dir={dir} style={{ alignSelf: "flex-end", marginLeft: "auto", maxWidth: "80ch" }}>
                {content}
            </div>
        );
    }

    return (
        <div id={id} className="msg assistant" dir={dir} style={{ alignSelf: "flex-end", marginLeft: "auto", maxWidth: "80ch" }}>
            {content}
        </div>
    );
}
