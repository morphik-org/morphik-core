import { NextRequest, NextResponse } from "next/server";
import { broadcastPDFCommand } from "@/lib/pdf-commands";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { top, bottom } = body;

    if (typeof top !== "number" || typeof bottom !== "number") {
      return NextResponse.json(
        { error: "Invalid zoom bounds. Expected { top: number, bottom: number }" },
        { status: 400 }
      );
    }

    if (top >= bottom) {
      return NextResponse.json({ error: "Top bound must be less than bottom bound" }, { status: 400 });
    }

    // Broadcast command to all connected PDF viewers
    broadcastPDFCommand({
      type: "zoomToY",
      bounds: { top, bottom },
      timestamp: new Date().toISOString(),
    });

    return NextResponse.json({
      success: true,
      message: `Zoomed to Y bounds: top=${top}, bottom=${bottom}`,
      bounds: { top, bottom },
    });
  } catch (error) {
    console.error("Error zooming PDF (Y):", error);
    return NextResponse.json({ error: "Failed to zoom PDF" }, { status: 500 });
  }
}
