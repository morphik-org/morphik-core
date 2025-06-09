import { NextRequest, NextResponse } from "next/server";
import { broadcastPDFCommand } from "@/lib/pdf-commands";

export async function POST(request: NextRequest, { params }: { params: { page: string } }) {
  try {
    const page = parseInt(params.page);

    if (isNaN(page) || page < 1) {
      return NextResponse.json({ error: "Invalid page number" }, { status: 400 });
    }

    // Broadcast command to all connected PDF viewers
    broadcastPDFCommand({
      type: "changePage",
      page: page,
      timestamp: new Date().toISOString(),
    });

    return NextResponse.json({
      success: true,
      message: `Changed to page ${page}`,
      page,
    });
  } catch (error) {
    console.error("Error changing PDF page:", error);
    return NextResponse.json({ error: "Failed to change page" }, { status: 500 });
  }
}
