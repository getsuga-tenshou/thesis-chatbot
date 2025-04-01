import { NextResponse } from "next/server";
import connectDB from "@/lib/mongodb";
import { Conversation } from "@/models/conversation";

export async function GET() {
  try {
    await connectDB();
    const conversations = await Conversation.find()
      .sort({ lastUpdated: -1 })
      .populate("messages");

    return NextResponse.json(conversations);
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
