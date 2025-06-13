import { NextRequest, NextResponse } from "next/server";
import { broadcastPDFCommand } from "@/lib/pdf-commands";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { left, right, sessionId, userId } = body;

    console.log("Zoom X API called with:", { left, right, sessionId, userId });
    console.log("Headers:", {
      "x-session-id": request.headers.get("x-session-id"),
      "x-user-id": request.headers.get("x-user-id"),
    });

    if (typeof left !== "number" || typeof right !== "number") {
      return NextResponse.json(
        { error: "Invalid zoom bounds. Expected { left: number, right: number }" },
        { status: 400 }
      );
    }

    if (left >= right) {
      return NextResponse.json({ error: "Left bound must be less than right bound" }, { status: 400 });
    }

    // Get session and user info from body or headers
    const finalSessionId = sessionId || request.headers.get("x-session-id") || "default";
    const finalUserId = userId || request.headers.get("x-user-id") || "anonymous";

    console.log("Final session/user IDs:", { finalSessionId, finalUserId });

    // Broadcast command to all connected PDF viewers in this session
    broadcastPDFCommand(
      {
        type: "zoomToX",
        bounds: { left, right },
        timestamp: new Date().toISOString(),
      },
      finalSessionId,
      finalUserId
    );

    return NextResponse.json({
      success: true,
      message: `Zoomed to X bounds: left=${left}, right=${right}`,
      bounds: { left, right },
      sessionId: finalSessionId,
      userId: finalUserId,
    });
  } catch (error) {
    console.error("Error zooming PDF (X):", error);
    return NextResponse.json({ error: "Failed to zoom PDF" }, { status: 500 });
  }
}
