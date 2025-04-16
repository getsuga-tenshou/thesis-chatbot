import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Send } from "lucide-react";
import styles from "../styles/homepage.module.css";
import { useEffect, useRef } from "react";

interface Message {
  id: string | number;
  role: "user" | "assistant";
  content: string;
}

interface NewChatProps {
  messages: Message[];
  input: string;
  onInputChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onSubmit: (e: React.FormEvent<HTMLFormElement>) => void;
  chatTitle: string;
  isLoading?: boolean;
}

export default function NewChat({
  messages = [],
  input = "",
  onInputChange,
  onSubmit,
  chatTitle = "Chat",
  isLoading = false,
}: NewChatProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onInputChange(e);
  };

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    onSubmit(e);
  };

  return (
    <div
      className={styles.chatContainer}
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        width: "100%",
      }}
    >
      <header className={styles.header}>
        <h1 className={styles.headerTitle}>{chatTitle}</h1>
      </header>

      <ScrollArea
        className={styles.messageArea}
        style={{ flex: 1, height: "calc(100vh - 140px)", position: "relative" }}
      >
        <div className={styles.messagesWrapper}>
          {messages.map((message) => (
            <div
              key={message.id}
              className={`${styles.message} ${
                message.role === "user"
                  ? styles.userMessage
                  : styles.assistantMessage
              }`}
            >
              <div
                className={`${styles.messageBubble} ${
                  message.role === "user"
                    ? styles.userMessageBubble
                    : styles.assistantMessageBubble
                }`}
              >
                {message.content}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className={`${styles.message} ${styles.assistantMessage}`}>
              <div
                className={`${styles.messageBubble} ${styles.assistantMessageBubble}`}
              >
                <div className={styles.loadingDots}>
                  <span>.</span>
                  <span>.</span>
                  <span>.</span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </ScrollArea>

      <div
        className={styles.inputArea}
        style={{
          position: "sticky",
          bottom: 0,
          backgroundColor: "#1a1a1a",
          padding: "1rem",
          borderTop: "1px solid #444",
        }}
      >
        <form
          onSubmit={handleSubmit}
          style={{ display: "flex", gap: "0.5rem" }}
        >
          <Input
            value={input}
            onChange={handleInputChange}
            placeholder="Type your message..."
            className="flex-grow"
            style={{
              backgroundColor: "#222",
              color: "#fff",
              padding: "0.5rem",
              border: "1px solid #444",
            }}
            disabled={isLoading}
          />
          <Button
            type="submit"
            disabled={isLoading}
            style={{
              backgroundColor: "#8a2be2",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: "40px",
              height: "40px",
              padding: "0",
            }}
          >
            <Send className="h-4 w-4" />
          </Button>
        </form>
      </div>
    </div>
  );
}
