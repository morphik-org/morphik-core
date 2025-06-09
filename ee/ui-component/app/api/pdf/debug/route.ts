import { NextResponse } from "next/server";

export async function GET() {
  const clientCount = globalThis.pdfClients ? globalThis.pdfClients.length : 0;
  const queueLength = globalThis.pdfCommandQueue ? globalThis.pdfCommandQueue.length : 0;

  return NextResponse.json({
    connectedClients: clientCount,
    queuedCommands: queueLength,
    commands: globalThis.pdfCommandQueue || [],
    timestamp: new Date().toISOString(),
  });
}
