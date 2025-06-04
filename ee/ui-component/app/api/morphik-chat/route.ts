import { NextRequest } from "next/server";
import { streamText } from "ai";
import { openai } from "@ai-sdk/openai";

export const runtime = "edge";
export const maxDuration = 30;

export async function POST(request: NextRequest) {
  try {
    const { messages } = await request.json();
    const { searchParams } = new URL(request.url);

    const chatId = searchParams.get("chatId");
    const apiBaseUrl = searchParams.get("apiBaseUrl");
    const authToken = searchParams.get("authToken");

    console.log("Morphik chat request:", { chatId, apiBaseUrl, messageCount: messages?.length });

    if (!messages || messages.length === 0) {
      return new Response("No messages provided", { status: 400 });
    }

    // Get the last message (user message)
    const lastMessage = messages[messages.length - 1];
    if (!lastMessage || lastMessage.role !== "user") {
      return new Response("Invalid message format", { status: 400 });
    }

    // Extract text content from the message
    const userQuery = Array.isArray(lastMessage.content)
      ? lastMessage.content
          .filter((part: any) => part.type === "text")
          .map((part: any) => part.text)
          .join("")
      : typeof lastMessage.content === "string"
      ? lastMessage.content
      : "";

    console.log("User query:", userQuery);

    // For now, let's use a simple mock response to test the integration
    // Later we can integrate with the actual Morphik API
    const result = streamText({
      model: openai("gpt-4o-mini"),
      messages: [
        {
          role: "system",
          content: "You are Morphik, a helpful AI assistant that helps with document queries. Respond helpfully and concisely."
        },
        ...messages
      ],
      temperature: 0.3,
      maxTokens: 1024,
    });

    return result.toDataStreamResponse();

  } catch (error) {
    console.error("Morphik chat API error:", error);
    return new Response(
      error instanceof Error ? error.message : "Internal server error",
      { status: 500 }
    );
  }
}
