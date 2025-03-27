import { NextResponse } from "next/server";
import { HfInference } from "@huggingface/inference";

export const runtime = "edge";

const HF_API_TOKEN = process.env.HUGGING_FACE_API_KEY;

// Initialize the inference client
const inference = new HfInference(HF_API_TOKEN);

export async function POST(req: Request) {
  try {
    if (!HF_API_TOKEN) {
      return NextResponse.json(
        { error: "API key not configured" },
        { status: 500 }
      );
    }

    const { messages } = await req.json();
    const lastMessage = messages[messages.length - 1];

    // Generate text using the inference client
    const result = await inference.textGeneration({
      model: "google/flan-t5-large",
      inputs: lastMessage.content,
      parameters: {
        max_length: 100,
        temperature: 0.7,
      },
    });

    // Create a stream of the response
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        // Format message as per Vercel AI SDK requirements
        controller.enqueue(
          encoder.encode(
            `data: ${JSON.stringify({
              id: Date.now().toString(),
              role: "assistant",
              content: result.generated_text,
            })}\n\n`
          )
        );
        // Signal the end of the stream
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
      },
    });

    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Failed to get response from the model",
      },
      { status: 500 }
    );
  }
}
