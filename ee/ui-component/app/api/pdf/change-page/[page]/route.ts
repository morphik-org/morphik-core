import { NextRequest, NextResponse } from "next/server";
import { broadcastPDFCommand } from "@/lib/pdf-commands";

export async function POST(request: NextRequest, { params }: { params: { page: string } }) {
  try {
    const page = parseInt(params.page);

    if (isNaN(page) || page < 1) {
      return NextResponse.json({ error: "Invalid page number" }, { status: 400 });
    }

    // Get session and user info from request headers or body
    const body = await request.json().catch(() => ({}));
    const sessionId = body.sessionId || request.headers.get("x-session-id") || "default";
    const userId = body.userId || request.headers.get("x-user-id") || "anonymous";

    // Broadcast command to all connected PDF viewers in this session
    broadcastPDFCommand(
      {
        type: "changePage",
        page: page,
        timestamp: new Date().toISOString(),
      },
      sessionId,
      userId
    );

    return NextResponse.json({
      success: true,
      message: `Changed to page ${page}`,
      page,
      sessionId,
      userId,
    });
  } catch (error) {
    console.error("Error changing PDF page:", error);
    return NextResponse.json({ error: "Failed to change page" }, { status: 500 });
  }
}
