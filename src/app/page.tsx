"use client";

import { useState } from "react";
import { useChat } from "ai/react";
// (UI imports from your own library or Shadcn UI)
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageSquare, Send, User } from "lucide-react";

import Welcome from "../app/components/welcome";
import NewChat from "./components/newchat";
import styles from "../app/styles/homepage.module.css";

export default function ChatbotPage() {
  const { messages, input, handleInputChange, handleSubmit, isLoading } =
    useChat({
      api: "/api/chat",
      onError: (error) => {
        console.error("Error in chat:", error);
        // To be added here later...
      },
    });

  const [selectedChat, setSelectedChat] = useState<string | null>(null);

  const startNewChat = () => {
    setSelectedChat("New Chat");
  };

  const goToHome = () => {
    setSelectedChat(null);
  };

  const handleWelcomeSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (input.trim()) {
      startNewChat();
      handleSubmit(e);
    }
  };

  const handleNewChatSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (input.trim()) {
      handleSubmit(e);
    }
  };

  return (
    <div className={styles.container}>
      {/* Sidebar */}
      <div className={styles.sidebar}>
        <h2 className={styles.sidebarTitle} onClick={goToHome}>
          Socratic AI
        </h2>
        <ScrollArea className={styles.chatList}>
          <Button
            variant="ghost"
            className="w-full justify-start text-white hover:text-white"
            onClick={startNewChat}
          >
            <MessageSquare className="mr-2 h-4 w-4" />
            New Chat
          </Button>
        </ScrollArea>
        <div className={styles.userInfo}>
          <div className="flex items-center">
            <Avatar className="h-8 w-8">
              <AvatarImage src="/placeholder-avatar.jpg" alt="User" />
              <AvatarFallback>
                <User />
              </AvatarFallback>
            </Avatar>
            <span className="ml-2">User Name</span>
          </div>
        </div>
      </div>

      {/* Main chat area */}
      <div className={styles.mainArea}>
        {selectedChat ? (
          <NewChat
            messages={messages}
            input={input}
            onInputChange={handleInputChange}
            onSubmit={handleNewChatSubmit}
            chatTitle={selectedChat}
            isLoading={isLoading}
          />
        ) : (
          <Welcome
            input={input}
            onInputChange={handleInputChange}
            onSubmit={handleWelcomeSubmit}
          />
        )}
      </div>
    </div>
  );
}
