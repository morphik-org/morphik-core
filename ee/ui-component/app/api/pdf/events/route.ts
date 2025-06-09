import { NextRequest } from "next/server";
import { addClient, removeClient } from "@/lib/pdf-commands";

export const dynamic = "force-dynamic";

// Extend the controller interface to include our custom heartbeat property
interface ExtendedController extends ReadableStreamDefaultController {
  heartbeat?: NodeJS.Timeout;
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
export async function GET(_request: NextRequest) {
  let streamController: ExtendedController | null = null;

  const stream = new ReadableStream({
    start(controller) {
      console.log("PDF events stream started - new client connected");
      streamController = controller as ExtendedController;
      addClient(controller);

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
      console.log("PDF events stream cancelled - client disconnected");
      if (streamController) {
        // Clear heartbeat timer
        if (streamController.heartbeat) {
          clearInterval(streamController.heartbeat);
        }
        removeClient(streamController);
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
