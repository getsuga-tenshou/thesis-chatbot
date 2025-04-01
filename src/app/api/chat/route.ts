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

    let conversation;
    if (conversationId) {
      conversation = await Conversation.findById(conversationId);
      if (!conversation) {
        throw new Error("Conversation not found");
      }
    } else {
      conversation = await Conversation.create({
        title: lastMessage.content.slice(0, 30) + "...",
      });
    }

    const userMessage = await Message.create({
      role: "user",
      content: lastMessage.content,
      conversationId: conversation._id,
    });

    conversation.messages.push(userMessage._id);
    conversation.lastUpdated = new Date();
    await conversation.save();

    const client = new InferenceClient(API_KEY);
    const response = await client.textGeneration({
      model: "google/flan-t5-large",
      inputs: lastMessage.content,
      parameters: {
        max_new_tokens: 250,
        temperature: 0.7,
      },
    });

    const assistantMessage = await Message.create({
      role: "assistant",
      content: response.generated_text,
      conversationId: conversation._id,
    });

    conversation.messages.push(assistantMessage._id);
    await conversation.save();

    return NextResponse.json({
      content: response.generated_text,
      conversationId: conversation._id,
    });
  } catch (error: any) {
    console.error("API Error:", error);
    return NextResponse.json(
      { error: error.message || "Failed to process request" },
      { status: 500 }
    );
  }
}
