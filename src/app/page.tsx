"use client";

import { useState, useEffect } from "react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageSquare, User, RefreshCw } from "lucide-react";

import Welcome from "../app/components/welcome";
import NewChat from "./components/newchat";
import styles from "../app/styles/homepage.module.css";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
}

interface Conversation {
  _id: string;
  title: string;
  messages: Message[];
  lastUpdated: string;
}

export default function ChatbotPage() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversation, setActiveConversation] =
    useState<Conversation | null>(null);
  const [currentMessages, setCurrentMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showWelcome, setShowWelcome] = useState(true);
  const [isResetting, setIsResetting] = useState(false);

  // Fetch all conversations on component mount
  useEffect(() => {
    fetchConversations();
  }, []);

  const fetchConversations = async () => {
    try {
      const response = await fetch("/api/conversations");
      if (!response.ok) throw new Error("Failed to fetch conversations");
      const data = await response.json();

      // Verify we have valid conversations
      if (Array.isArray(data) && data.length > 0) {
        console.log("Conversations fetched:", data);
        setConversations(data);
      }
    } catch (error) {
      console.error("Error fetching conversations:", error);
    }
  };

  // Reset database
  const resetDatabase = async () => {
    try {
      setIsResetting(true);
      const response = await fetch("/api/reset-db");
      if (!response.ok) throw new Error("Failed to reset database");

      await fetchConversations();
      startNewChat();
    } catch (error) {
      console.error("Error resetting database:", error);
    } finally {
      setIsResetting(false);
    }
  };

  // Set messages when active conversation changes
  useEffect(() => {
    if (activeConversation && activeConversation.messages) {
      setCurrentMessages(activeConversation.messages);
    } else {
      setCurrentMessages([]);
    }
  }, [activeConversation]);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
    };

    // Add user message to the UI immediately
    setCurrentMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: [...currentMessages, userMessage],
          conversationId: activeConversation?._id,
        }),
      });

      if (!response.ok) throw new Error("Failed to fetch response");

      const data = await response.json();

      // Use the data returned from the API
      if (data.allMessages) {
        // If we have all messages, use those
        setCurrentMessages(data.allMessages);
      } else {
        // Otherwise just add the assistant response
        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: data.content,
        };
        setCurrentMessages((prev) => [...prev, assistantMessage]);
      }

      // Update conversations to reflect the new state
      await fetchConversations();

      // Update active conversation if needed
      if (data.conversationId) {
        const updatedConversation = conversations.find(
          (conv) => conv._id === data.conversationId
        );

        if (updatedConversation) {
          setActiveConversation(updatedConversation);
        }
      }
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleWelcomeSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
    };

    setCurrentMessages([userMessage]);
    setInput("");
    setIsLoading(true);
    setShowWelcome(false);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: [userMessage],
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch response");
      }

      const data = await response.json();
      console.log("Response from chat API:", data);

      // Use the allMessages from API if available
      if (data.allMessages) {
        setCurrentMessages(data.allMessages);
      } else {
        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: data.content,
        };
        setCurrentMessages([userMessage, assistantMessage]);
      }

      // Refresh the conversation list
      await fetchConversations();

      // Set the active conversation to the new one
      if (data.conversationId) {
        const newConversation = conversations.find(
          (conv) => conv._id === data.conversationId
        );

        if (newConversation) {
          setActiveConversation(newConversation);
        } else {
          // If conversation not found, fetch it again
          await fetchConversations();
          const updatedConversations = await (
            await fetch("/api/conversations")
          ).json();
          const foundConversation = updatedConversations.find(
            (conv: Conversation) => conv._id === data.conversationId
          );

          if (foundConversation) {
            setActiveConversation(foundConversation);
            setConversations(updatedConversations);
          }
        }
      }
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const startNewChat = () => {
    setActiveConversation(null);
    setCurrentMessages([]);
    setInput("");
    setShowWelcome(true);
  };

  const selectConversation = (conversation: Conversation) => {
    setActiveConversation(conversation);
    setCurrentMessages(conversation.messages || []);
    setShowWelcome(false);
  };

  return (
    <div
      className={styles.container}
      style={{ display: "flex", height: "100vh", backgroundColor: "#111" }}
    >
      {/* Sidebar */}
      <div
        className={styles.sidebar}
        style={{
          width: "16rem",
          backgroundColor: "#111",
          background: "linear-gradient(to bottom, #111, #222)",
          color: "#fff",
          padding: "1rem",
          display: "flex",
          flexDirection: "column",
          borderRight: "1px solid #444",
        }}
      >
        <h2 className={styles.sidebarTitle} onClick={startNewChat}>
          Socratic AI
        </h2>
        <div className="flex flex-col space-y-2 mb-4">
          <Button
            variant="ghost"
            className="w-full justify-start text-white hover:text-white"
            onClick={startNewChat}
          >
            <MessageSquare className="mr-2 h-4 w-4" />
            New Chat
          </Button>

          <Button
            variant="ghost"
            className="w-full justify-start text-white hover:text-white"
            onClick={resetDatabase}
            disabled={isResetting}
          >
            <RefreshCw
              className={`mr-2 h-4 w-4 ${isResetting ? "animate-spin" : ""}`}
            />
            Reset Database
          </Button>
        </div>

        <ScrollArea className={styles.chatList} style={{ flexGrow: 1 }}>
          {conversations.length > 0 ? (
            conversations.map((conversation) => (
              <Button
                key={conversation._id}
                variant="ghost"
                className={`w-full justify-start text-white hover:text-white mb-1 truncate ${
                  activeConversation?._id === conversation._id
                    ? styles.activeChat
                    : ""
                }`}
                onClick={() => selectConversation(conversation)}
              >
                <MessageSquare className="mr-2 h-4 w-4 flex-shrink-0" />
                <span className="truncate">
                  {conversation.title || "Untitled Chat"}
                </span>
              </Button>
            ))
          ) : (
            <div className="text-center text-white/50 p-4">
              No conversations yet
            </div>
          )}
        </ScrollArea>
        <div
          className={styles.userInfo}
          style={{
            marginTop: "auto",
            paddingTop: "1rem",
            borderTop: "1px solid #333",
          }}
        >
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
      <div
        className={styles.mainArea}
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          background: "linear-gradient(to bottom right, #111, #222)",
        }}
      >
        {showWelcome ? (
          <Welcome
            input={input}
            onInputChange={(e) => setInput(e.target.value)}
            onSubmit={handleWelcomeSubmit}
          />
        ) : (
          <NewChat
            messages={currentMessages}
            input={input}
            onInputChange={(e) => setInput(e.target.value)}
            onSubmit={handleSubmit}
            chatTitle={activeConversation?.title || "New Chat"}
            isLoading={isLoading}
          />
        )}
      </div>
    </div>
  );
}
