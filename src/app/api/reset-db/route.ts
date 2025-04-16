import { NextResponse } from "next/server";
import connectDB from "@/lib/mongodb";
import { Conversation } from "@/models/conversation";
import { Message } from "@/models/message";

export async function GET() {
  try {
    console.log("Starting database reset operation");
    await connectDB();

    // Delete all existing conversations and messages
    console.log("Deleting all existing conversations and messages");
    await Message.deleteMany({});
    await Conversation.deleteMany({});

    // Create a sample conversation
    console.log("Creating a sample conversation");
    const sampleConversation = await Conversation.create({
      title: "Welcome Conversation",
      lastUpdated: new Date(),
    });

    // Add sample messages
    console.log("Adding sample messages");
    const userMessage = await Message.create({
      role: "user",
      content: "Hello, how can you help me?",
      conversationId: sampleConversation._id,
    });

    const assistantMessage = await Message.create({
      role: "assistant",
      content:
        "Hello! I'm Socratic AI, your learning assistant. I can help you by answering questions, explaining concepts, and guiding your learning journey. What would you like to learn about today?",
      conversationId: sampleConversation._id,
    });

    // Add messages to conversation
    sampleConversation.messages.push(userMessage._id, assistantMessage._id);
    await sampleConversation.save();

    console.log("Database reset complete");

    return NextResponse.json({
      success: true,
      message: "Database reset successfully",
      conversation: {
        id: sampleConversation._id,
        title: sampleConversation.title,
      },
    });
  } catch (error: any) {
    console.error("Error resetting database:", error);
    return NextResponse.json(
      {
        success: false,
        error: error.message,
      },
      {
        status: 500,
      }
    );
  }
}
