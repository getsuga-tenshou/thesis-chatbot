import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Send } from "lucide-react";
import styles from "../styles/homepage.module.css";
import { useEffect, useRef } from "react";

interface Message {
  id: string | number;
  role: "user" | "assistant" | "system" | "data";
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
    <div className={styles.chatContainer}>
      <header className={styles.header}>
        <h1 className={styles.headerTitle}>{chatTitle}</h1>
      </header>

      <ScrollArea className={styles.messageArea}>
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

      <div className={styles.inputArea}>
        <form onSubmit={handleSubmit} className={styles.form}>
          <Input
            value={input}
            onChange={handleInputChange}
            placeholder="Type your message..."
            className="flex-grow"
            disabled={isLoading}
          />
          <Button type="submit" disabled={isLoading}>
            <Send className="h-4 w-4" />
          </Button>
        </form>
      </div>
    </div>
  );
}
