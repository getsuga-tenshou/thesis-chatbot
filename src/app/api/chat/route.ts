import { InferenceClient } from "@huggingface/inference";
import { NextResponse } from "next/server";

export const runtime = "edge";

const API_KEY = process.env.HUGGING_FACE_API_KEY;

if (!API_KEY) {
  throw new Error("HUGGINGFACE_API_KEY is not set in environment variables");
}

if (!API_KEY.startsWith("hf_")) {
  throw new Error("HUGGINGFACE_API_KEY should start with 'hf_'");
}

const client = new InferenceClient(API_KEY);

export async function POST(req: Request) {
  try {
    const { messages } = await req.json();

    if (!messages?.length) {
      return NextResponse.json(
        { error: "No messages provided" },
        { status: 400 }
      );
    }

    const lastMessage = messages[messages.length - 1];
    console.log("Processing message:", lastMessage.content);

    const response = await client.textGeneration({
      model: "google/flan-t5-large",
      inputs: lastMessage.content,
      parameters: {
        max_new_tokens: 250,
        temperature: 0.7,
      },
    });

    if (!response.generated_text) {
      throw new Error("No response generated from the model");
    }

    return NextResponse.json({ content: response.generated_text });
  } catch (error: any) {
    console.error("API Error:", {
      message: error.message,
      status: error.status,
      response: error.response?.data,
    });

    return NextResponse.json(
      { error: "Failed to generate response - " + error.message },
      { status: error.status || 500 }
    );
  }
}
