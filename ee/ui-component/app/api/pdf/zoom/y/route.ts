import { NextRequest, NextResponse } from "next/server";
import { broadcastPDFCommand } from "@/lib/pdf-commands";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { top, bottom, sessionId, userId } = body;

    console.log("Zoom Y API called with:", { top, bottom, sessionId, userId });
    console.log("Headers:", {
      "x-session-id": request.headers.get("x-session-id"),
      "x-user-id": request.headers.get("x-user-id"),
    });

    if (typeof top !== "number" || typeof bottom !== "number") {
      return NextResponse.json(
        { error: "Invalid zoom bounds. Expected { top: number, bottom: number }" },
        { status: 400 }
      );
    }

    if (top >= bottom) {
      return NextResponse.json({ error: "Top bound must be less than bottom bound" }, { status: 400 });
    }

    // Get session and user info from body or headers
    const finalSessionId = sessionId || request.headers.get("x-session-id") || "default";
    const finalUserId = userId || request.headers.get("x-user-id") || "anonymous";

    console.log("Final session/user IDs:", { finalSessionId, finalUserId });

    // Broadcast command to all connected PDF viewers in this session
    broadcastPDFCommand(
      {
        type: "zoomToY",
        bounds: { top, bottom },
        timestamp: new Date().toISOString(),
      },
      finalSessionId,
      finalUserId
    );

    return NextResponse.json({
      success: true,
      message: `Zoomed to Y bounds: top=${top}, bottom=${bottom}`,
      bounds: { top, bottom },
      sessionId: finalSessionId,
      userId: finalUserId,
    });
  } catch (error) {
    console.error("Error zooming PDF (Y):", error);
    return NextResponse.json({ error: "Failed to zoom PDF" }, { status: 500 });
  }
}
