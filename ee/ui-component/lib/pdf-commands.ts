interface PDFCommand {
  type: string;
  page?: number;
  bounds?: { top?: number; bottom?: number; left?: number; right?: number };
  timestamp: string;
}

// Global command queue for PDF viewer commands - use globalThis to persist across hot reloads
if (!globalThis.pdfClients) {
  globalThis.pdfClients = [];
}
if (!globalThis.pdfCommandQueue) {
  globalThis.pdfCommandQueue = [];
}

const clients: ReadableStreamDefaultController[] = globalThis.pdfClients;
let commandQueue: PDFCommand[] = globalThis.pdfCommandQueue;

export function addClient(controller: ReadableStreamDefaultController) {
  console.log("Adding new PDF API client");
  clients.push(controller);
  console.log("Total clients:", clients.length);

  // Send initial connection message
  const connectionMessage = `data: ${JSON.stringify({ type: "connected" })}\n\n`;
  console.log("Sending connection message:", connectionMessage);
  controller.enqueue(connectionMessage);

  // Send any queued commands
  console.log("Sending queued commands:", commandQueue.length);
  commandQueue.forEach(command => {
    const message = `data: ${JSON.stringify(command)}\n\n`;
    console.log("Sending queued command:", message);
    controller.enqueue(message);
  });

  // Clear the queue after sending
  commandQueue = [];
  globalThis.pdfCommandQueue = commandQueue;
}

export function removeClient(controller: ReadableStreamDefaultController) {
  console.log("Removing PDF API client");
  const index = clients.indexOf(controller);
  if (index > -1) {
    clients.splice(index, 1);
    console.log("Client removed. Remaining clients:", clients.length);
  } else {
    console.log("Client not found in list");
  }
}

// Function to broadcast commands to all connected clients
export function broadcastPDFCommand(command: PDFCommand) {
  console.log("Broadcasting PDF command:", command);
  console.log("Connected clients:", clients.length);

  const message = `data: ${JSON.stringify(command)}\n\n`;

  // Send to all connected clients
  clients.forEach((controller, index) => {
    try {
      console.log(`Sending command to client ${index + 1}:`, message);
      controller.enqueue(message);
    } catch (error) {
      console.error(`Error sending command to client ${index + 1}:`, error);
      // Remove failed client
      const clientIndex = clients.indexOf(controller);
      if (clientIndex > -1) {
        clients.splice(clientIndex, 1);
        console.log(`Removed failed client. Remaining clients: ${clients.length}`);
      }
    }
  });

  // If no clients connected, queue the command
  if (clients.length === 0) {
    console.log("No clients connected, queueing command");
    commandQueue.push(command);
    globalThis.pdfCommandQueue = commandQueue;
    // Keep only the last 10 commands to prevent memory issues
    if (commandQueue.length > 10) {
      commandQueue = commandQueue.slice(-10);
      globalThis.pdfCommandQueue = commandQueue;
    }
  }
}
