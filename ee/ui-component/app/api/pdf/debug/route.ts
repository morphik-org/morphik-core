import { NextResponse } from "next/server";

export async function GET() {
  const sessions = globalThis.pdfSessions || new Map();

  const sessionInfo = Array.from(sessions.entries()).map(([sessionId, session]) => ({
    sessionId,
    userId: session.userId,
    connectedClients: session.clients.length,
    queuedCommands: session.commandQueue.length,
    lastActivity: session.lastActivity.toISOString(),
    commands: session.commandQueue,
  }));

  const totalClients = Array.from(sessions.values()).reduce((sum, session) => sum + session.clients.length, 0);
  const totalQueuedCommands = Array.from(sessions.values()).reduce(
    (sum, session) => sum + session.commandQueue.length,
    0
  );

  return NextResponse.json({
    totalSessions: sessions.size,
    totalConnectedClients: totalClients,
    totalQueuedCommands: totalQueuedCommands,
    sessions: sessionInfo,
    timestamp: new Date().toISOString(),
  });
}
