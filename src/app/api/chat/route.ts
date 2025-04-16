import { InferenceClient } from "@huggingface/inference";
import { NextResponse } from "next/server";
import connectDB from "@/lib/mongodb";
import { Message } from "@/models/message";
import { Conversation } from "@/models/conversation";

const API_KEY = process.env.HUGGING_FACE_API_KEY;

if (!API_KEY) {
  throw new Error("HUGGINGFACE_API_KEY is not set in environment variables");
}

if (!API_KEY.startsWith("hf_")) {
  throw new Error("HUGGINGFACE_API_KEY should start with 'hf_'");
}

export async function POST(req: Request) {
  try {
    await connectDB();

    const { messages, conversationId } = await req.json();
    const lastMessage = messages[messages.length - 1];

    // Clean the message content for title use
    const sanitizedContent = lastMessage.content
      .replace(/db\.messages\.insertOne/g, "") // Remove database commands if present
      .trim();

    // Format a proper title from the first message content
    let displayTitle = sanitizedContent;
    if (displayTitle.length > 30) {
      displayTitle = displayTitle.substring(0, 30) + "...";
    }

    let conversation;
    if (conversationId) {
      conversation = await Conversation.findById(conversationId);
      if (!conversation) {
        throw new Error("Conversation not found");
      }
    } else {
      // Create a new conversation with a proper title
      conversation = await Conversation.create({
        title: displayTitle || "New Conversation",
      });
    }

    // Create user message
    const userMessage = await Message.create({
      role: "user",
      content: lastMessage.content,
      conversationId: conversation._id,
    });

    // Add message to conversation
    conversation.messages.push(userMessage._id);
    conversation.lastUpdated = new Date();
    await conversation.save();

    // Generate AI response
    const client = new InferenceClient(API_KEY);
    const response = await client.textGeneration({
      model: "google/flan-t5-large",
      inputs: lastMessage.content,
      parameters: {
        max_new_tokens: 250,
        temperature: 0.7,
      },
    });

    // Create assistant message
    const assistantMessage = await Message.create({
      role: "assistant",
      content: response.generated_text,
      conversationId: conversation._id,
    });

    // Add assistant message to conversation
    conversation.messages.push(assistantMessage._id);
    await conversation.save();

    // Get all messages for this conversation to return
    const allMessages = await Message.find({
      conversationId: conversation._id,
    }).sort({ createdAt: 1 });

    // Format messages for the response
    const formattedMessages = allMessages.map((msg) => ({
      id: msg._id,
      role: msg.role,
      content: msg.content,
    }));

    return NextResponse.json({
      content: response.generated_text,
      conversationId: conversation._id,
      title: conversation.title,
      allMessages: formattedMessages,
    });
  } catch (error: any) {
    console.error("API Error:", error);
    return NextResponse.json(
      { error: error.message || "Failed to process request" },
      { status: 500 }
    );
  }
}
