import { NextResponse } from "next/server";
import connectDB from "@/lib/mongodb";
import { Conversation } from "@/models/conversation";
import { Message } from "@/models/message";

export async function GET() {
  try {
    await connectDB();

    // Get all conversations sorted by last updated
    const conversations = await Conversation.find().sort({ lastUpdated: -1 });

    // Prepare response with properly formatted messages
    const conversationsWithMessages = await Promise.all(
      conversations.map(async (conversation) => {
        const messages = await Message.find({
          conversationId: conversation._id,
        }).sort({ createdAt: 1 });

        return {
          _id: conversation._id,
          title: conversation.title,
          lastUpdated: conversation.lastUpdated,
          createdAt: conversation.createdAt,
          messages: messages.map((msg) => ({
            id: msg._id,
            role: msg.role,
            content: msg.content,
          })),
        };
      })
    );

    return NextResponse.json(conversationsWithMessages);
  } catch (error: any) {
    console.error("Error fetching conversations:", error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
