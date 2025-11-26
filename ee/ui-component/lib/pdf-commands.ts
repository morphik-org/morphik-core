interface PDFCommand {
  type: string;
  page?: number;
  bounds?: { top?: number; bottom?: number; left?: number; right?: number };
  timestamp: string;
  sessionId?: string;
  userId?: string;
}

interface PDFSession {
  sessionId: string;
  userId: string;
  clients: ReadableStreamDefaultController[];
  commandQueue: PDFCommand[];
  lastActivity: Date;
}

// Extend globalThis to include our PDF sessions
declare global {
  // eslint-disable-next-line no-var
  var pdfSessions: Map<string, PDFSession> | undefined;
}

// Global sessions map - use globalThis to persist across hot reloads
if (!globalThis.pdfSessions) {
  globalThis.pdfSessions = new Map<string, PDFSession>();
}

const sessions: Map<string, PDFSession> = globalThis.pdfSessions;

const MAX_LOG_VALUE_LENGTH = 200;

function sanitizeForLog(value: string): string {
  if (!value) return "";
  const trimmed = value.replace(/[\r\n\t]+/g, " ").replace(/[^\x20-\x7E]/g, "");
  return trimmed.length > MAX_LOG_VALUE_LENGTH ? `${trimmed.slice(0, MAX_LOG_VALUE_LENGTH)}...` : trimmed;
}

// Clean up inactive sessions (older than 1 hour)
function cleanupInactiveSessions() {
  const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000);
  for (const [sessionId, session] of Array.from(sessions.entries())) {
    if (session.lastActivity < oneHourAgo) {
      const safeSessionId = sanitizeForLog(sessionId);
      console.log("Cleaning up inactive session:", safeSessionId);
      // Close all clients in the session
      session.clients.forEach((client: ReadableStreamDefaultController) => {
        try {
          client.close();
        } catch (error) {
          console.error("Error closing client:", error);
        }
      });
      sessions.delete(sessionId);
    }
  }
}

// Run cleanup every 30 minutes
setInterval(cleanupInactiveSessions, 30 * 60 * 1000);

export function getOrCreateSession(sessionId: string, userId: string): PDFSession {
  const safeSessionId = sanitizeForLog(sessionId);
  const safeUserId = sanitizeForLog(userId);
  let session = sessions.get(sessionId);

  if (!session) {
    session = {
      sessionId,
      userId,
      clients: [],
      commandQueue: [],
      lastActivity: new Date(),
    };
    sessions.set(sessionId, session);
    console.log("Created new PDF session", safeSessionId, "for user", safeUserId);
  } else {
    // Verify user owns this session (allow anonymous/authenticated user mixing for development)
    const isUserMatch =
      session.userId === userId ||
      session.userId === "anonymous" ||
      userId === "anonymous" ||
      session.userId === "authenticated" ||
      userId === "authenticated";

    if (!isUserMatch) {
      throw new Error(
        `Session ${sessionId} belongs to different user (session: ${session.userId}, request: ${userId})`
      );
    }
    session.lastActivity = new Date();
  }

  return session;
}

export function addClient(controller: ReadableStreamDefaultController, sessionId: string, userId: string) {
  const safeSessionId = sanitizeForLog(sessionId);
  const safeUserId = sanitizeForLog(userId);
  console.log("Adding new PDF API client for session", safeSessionId, "user", safeUserId);

  const session = getOrCreateSession(sessionId, userId);
  session.clients.push(controller);

  console.log("Total clients in session", safeSessionId, session.clients.length);

  // Send initial connection message
  const connectionMessage = `data: ${JSON.stringify({ type: "connected", sessionId, userId })}\n\n`;
  console.log("Sending connection message:", connectionMessage);
  controller.enqueue(connectionMessage);

  // Send any queued commands for this session
  console.log("Sending queued commands for session", safeSessionId, session.commandQueue.length);
  session.commandQueue.forEach(command => {
    const message = `data: ${JSON.stringify(command)}\n\n`;
    console.log("Sending queued command:", message);
    controller.enqueue(message);
  });

  // Clear the queue after sending
  session.commandQueue = [];
}

export function removeClient(controller: ReadableStreamDefaultController, sessionId: string) {
  const safeSessionId = sanitizeForLog(sessionId);
  console.log("Removing PDF API client from session", safeSessionId);

  const session = sessions.get(sessionId);
  if (!session) {
    console.log("Session not found", safeSessionId);
    return;
  }

  const index = session.clients.indexOf(controller);
  if (index > -1) {
    session.clients.splice(index, 1);
    console.log("Client removed from session", safeSessionId, "Remaining clients:", session.clients.length);

    // If no clients left in session, we could optionally clean it up
    // For now, we'll keep it for a while in case the client reconnects
  } else {
    console.log("Client not found in session");
  }
}

// Function to broadcast commands to all connected clients in a specific session
export function broadcastPDFCommand(command: PDFCommand, sessionId: string, userId: string) {
  const safeSessionId = sanitizeForLog(sessionId);
  const safeUserId = sanitizeForLog(userId);
  console.log("Broadcasting PDF command to session", safeSessionId, "user", safeUserId, command);

  const session = sessions.get(sessionId);
  if (!session) {
    console.log("Session not found, creating new session", safeSessionId, "user", safeUserId);
    getOrCreateSession(sessionId, userId);
    return broadcastPDFCommand(command, sessionId, userId);
  }

  // Verify user owns this session (allow anonymous/authenticated user mixing for development)
  const isUserMatch =
    session.userId === userId ||
    session.userId === "anonymous" ||
    userId === "anonymous" ||
    session.userId === "authenticated" ||
    userId === "authenticated";

  if (!isUserMatch) {
    throw new Error(`Session ${sessionId} belongs to different user (session: ${session.userId}, request: ${userId})`);
  }

  // Add session and user info to command
  const scopedCommand = {
    ...command,
    sessionId,
    userId,
  };

  console.log("Connected clients in session:", session.clients.length);

  const message = `data: ${JSON.stringify(scopedCommand)}\n\n`;

  // Send to all connected clients in this session
  session.clients.forEach((controller, index) => {
    try {
      console.log("Sending command to client", index + 1, "in session", safeSessionId, message);
      controller.enqueue(message);
    } catch (error) {
      console.error("Error sending command to client", index + 1, "in session", safeSessionId, error);
      // Remove failed client
      const clientIndex = session.clients.indexOf(controller);
      if (clientIndex > -1) {
        session.clients.splice(clientIndex, 1);
        console.log("Removed failed client from session", safeSessionId, "Remaining clients:", session.clients.length);
      }
    }
  });

  // If no clients connected in this session, queue the command
  if (session.clients.length === 0) {
    console.log("No clients connected in session", safeSessionId, "queueing command");
    session.commandQueue.push(scopedCommand);
    // Keep only the last 10 commands to prevent memory issues
    if (session.commandQueue.length > 10) {
      session.commandQueue = session.commandQueue.slice(-10);
    }
  }

  // Update last activity
  session.lastActivity = new Date();
}

// Helper function to get session info
export function getSessionInfo(sessionId: string): PDFSession | null {
  return sessions.get(sessionId) || null;
}

// Helper function to list all sessions for a user
export function getUserSessions(userId: string): PDFSession[] {
  return Array.from(sessions.values()).filter(session => session.userId === userId);
}
