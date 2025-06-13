import { NextRequest } from "next/server";
import { addClient, removeClient } from "@/lib/pdf-commands";

export const dynamic = "force-dynamic";

// Extend the controller interface to include our custom heartbeat property
interface ExtendedController extends ReadableStreamDefaultController {
  heartbeat?: NodeJS.Timeout;
}

export async function GET(request: NextRequest) {
  let streamController: ExtendedController | null = null;

  // Get session and user info from query parameters or headers
  const url = new URL(request.url);
  const sessionId = url.searchParams.get("sessionId") || request.headers.get("x-session-id") || "default";
  const userId = url.searchParams.get("userId") || request.headers.get("x-user-id") || "anonymous";

  console.log(`PDF events stream requested for session: ${sessionId}, user: ${userId}`);

  const stream = new ReadableStream({
    start(controller) {
      console.log(`PDF events stream started - new client connected for session: ${sessionId}, user: ${userId}`);
      streamController = controller as ExtendedController;
      addClient(controller, sessionId, userId);

      // Send initial heartbeat to keep connection alive
      const heartbeat = setInterval(() => {
        try {
          controller.enqueue(": heartbeat\n\n");
        } catch (error) {
          console.error("Error sending heartbeat:", error);
          clearInterval(heartbeat);
        }
      }, 30000); // Send heartbeat every 30 seconds

      // Store heartbeat timer for cleanup
      streamController.heartbeat = heartbeat;
    },
    cancel() {
      console.log(`PDF events stream cancelled - client disconnected from session: ${sessionId}`);
      if (streamController) {
        // Clear heartbeat timer
        if (streamController.heartbeat) {
          clearInterval(streamController.heartbeat);
        }
        removeClient(streamController, sessionId);
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Headers": "Cache-Control",
      "X-Accel-Buffering": "no", // Disable nginx buffering
    },
  });
}
