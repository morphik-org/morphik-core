import { NextRequest, NextResponse } from "next/server";
import { broadcastPDFCommand } from "@/lib/pdf-commands";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { left, right } = body;

    if (typeof left !== "number" || typeof right !== "number") {
      return NextResponse.json(
        { error: "Invalid zoom bounds. Expected { left: number, right: number }" },
        { status: 400 }
      );
    }

    if (left >= right) {
      return NextResponse.json({ error: "Left bound must be less than right bound" }, { status: 400 });
    }

    // Broadcast command to all connected PDF viewers
    broadcastPDFCommand({
      type: "zoomToX",
      bounds: { left, right },
      timestamp: new Date().toISOString(),
    });

    return NextResponse.json({
      success: true,
      message: `Zoomed to X bounds: left=${left}, right=${right}`,
      bounds: { left, right },
    });
  } catch (error) {
    console.error("Error zooming PDF (X):", error);
    return NextResponse.json({ error: "Failed to zoom PDF" }, { status: 500 });
  }
}
