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

// Clean up inactive sessions (older than 1 hour)
function cleanupInactiveSessions() {
  const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000);
  for (const [sessionId, session] of Array.from(sessions.entries())) {
    if (session.lastActivity < oneHourAgo) {
      console.log(`Cleaning up inactive session: ${sessionId}`);
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
    console.log(`Created new PDF session: ${sessionId} for user: ${userId}`);
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
  console.log(`Adding new PDF API client for session: ${sessionId}, user: ${userId}`);

  const session = getOrCreateSession(sessionId, userId);
  session.clients.push(controller);

  console.log(`Total clients in session ${sessionId}:`, session.clients.length);

  // Send initial connection message
  const connectionMessage = `data: ${JSON.stringify({ type: "connected", sessionId, userId })}\n\n`;
  console.log("Sending connection message:", connectionMessage);
  controller.enqueue(connectionMessage);

  // Send any queued commands for this session
  console.log(`Sending queued commands for session ${sessionId}:`, session.commandQueue.length);
  session.commandQueue.forEach(command => {
    const message = `data: ${JSON.stringify(command)}\n\n`;
    console.log("Sending queued command:", message);
    controller.enqueue(message);
  });

  // Clear the queue after sending
  session.commandQueue = [];
}

export function removeClient(controller: ReadableStreamDefaultController, sessionId: string) {
  console.log(`Removing PDF API client from session: ${sessionId}`);

  const session = sessions.get(sessionId);
  if (!session) {
    console.log(`Session ${sessionId} not found`);
    return;
  }

  const index = session.clients.indexOf(controller);
  if (index > -1) {
    session.clients.splice(index, 1);
    console.log(`Client removed from session ${sessionId}. Remaining clients:`, session.clients.length);

    // If no clients left in session, we could optionally clean it up
    // For now, we'll keep it for a while in case the client reconnects
  } else {
    console.log("Client not found in session");
  }
}

// Function to broadcast commands to all connected clients in a specific session
export function broadcastPDFCommand(command: PDFCommand, sessionId: string, userId: string) {
  console.log(`Broadcasting PDF command to session ${sessionId}:`, command);

  const session = sessions.get(sessionId);
  if (!session) {
    console.log(`Session ${sessionId} not found, creating new session`);
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
      console.log(`Sending command to client ${index + 1} in session ${sessionId}:`, message);
      controller.enqueue(message);
    } catch (error) {
      console.error(`Error sending command to client ${index + 1} in session ${sessionId}:`, error);
      // Remove failed client
      const clientIndex = session.clients.indexOf(controller);
      if (clientIndex > -1) {
        session.clients.splice(clientIndex, 1);
        console.log(`Removed failed client from session ${sessionId}. Remaining clients: ${session.clients.length}`);
      }
    }
  });

  // If no clients connected in this session, queue the command
  if (session.clients.length === 0) {
    console.log(`No clients connected in session ${sessionId}, queueing command`);
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
