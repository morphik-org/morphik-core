import { useEffect, createContext, useContext } from "react";

interface PDFSessionContext {
  sessionId: string;
  userId: string;
}

const PDFSessionContext = createContext<PDFSessionContext | null>(null);

export function usePDFSession() {
  const context = useContext(PDFSessionContext);
  if (!context) {
    throw new Error("usePDFSession must be used within a PDFAPIService");
  }
  return context;
}

interface PDFAPIServiceProps {
  children: React.ReactNode;
  sessionId?: string;
  userId?: string;
}

export function PDFAPIService({ children, sessionId = "default", userId = "anonymous" }: PDFAPIServiceProps) {
  useEffect(() => {
    console.log(`Starting PDF API service with session: ${sessionId}, user: ${userId}`);

    let eventSource: EventSource | null = null;
    let reconnectTimeout: NodeJS.Timeout | null = null;
    let isConnected = false;
    console.log("PDFAPIService: Starting with session:", sessionId, "and user:", userId, "connected:", isConnected);

    const connectToEventSource = () => {
      try {
        console.log("Connecting to PDF events stream...");

        // Include session and user info in the URL
        const url = new URL("/api/pdf/events", window.location.origin);
        url.searchParams.set("sessionId", sessionId);
        url.searchParams.set("userId", userId);

        eventSource = new EventSource(url.toString());
        console.log("EventSource created for URL:", url.toString());

        eventSource.onopen = () => {
          console.log("PDF events stream connected successfully");
          isConnected = true;
          // Clear any pending reconnection attempts
          if (reconnectTimeout) {
            clearTimeout(reconnectTimeout);
            reconnectTimeout = null;
          }
        };

        eventSource.onmessage = event => {
          console.log("Received PDF command:", event.data);
          try {
            const command = JSON.parse(event.data);
            console.log("Parsed command:", command);

            // Verify the command is for our session
            if (command.sessionId && command.sessionId !== sessionId) {
              console.log(`Ignoring command for different session: ${command.sessionId} (ours: ${sessionId})`);
              return;
            }

            if (window.pdfViewerControls) {
              const mode = window.pdfViewerControls.getMode ? window.pdfViewerControls.getMode() : "unknown";
              console.log("PDF viewer controls available, mode:", mode, "executing command:", command.type);

              if (mode === "api") {
                switch (command.type) {
                  case "changePage":
                    console.log("Changing page to:", command.page);
                    window.pdfViewerControls.changePage(command.page);
                    break;
                  case "zoomToY":
                    console.log("Zooming to Y bounds:", command.bounds);
                    window.pdfViewerControls.zoomToY(command.bounds);
                    break;
                  case "zoomToX":
                    console.log("Zooming to X bounds:", command.bounds);
                    window.pdfViewerControls.zoomToX(command.bounds);
                    break;
                  case "connected":
                    console.log("PDF API service connected successfully");
                    break;
                  default:
                    console.warn("Unknown PDF command:", command.type);
                }
              } else {
                console.warn("PDF viewer is in manual mode, ignoring API command:", command.type);
              }
            } else {
              console.warn("PDF viewer controls not available - ensure PDF is loaded and in API mode");
            }
          } catch (error) {
            console.error("Error processing PDF command:", error);
          }
        };

        eventSource.onerror = error => {
          console.error("PDF events stream error:", error);
          isConnected = false;

          // Close the current connection
          if (eventSource) {
            eventSource.close();
            eventSource = null;
          }

          // Attempt to reconnect after a delay
          if (!reconnectTimeout) {
            console.log("Attempting to reconnect in 3 seconds...");
            reconnectTimeout = setTimeout(() => {
              reconnectTimeout = null;
              connectToEventSource();
            }, 3000);
          }
        };
      } catch (error) {
        console.error("Error creating PDF events stream:", error);
      }
    };

    // Initial connection
    connectToEventSource();

    // Cleanup function
    return () => {
      console.log("Cleaning up PDF API service");
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }
      if (eventSource) {
        eventSource.close();
      }
    };
  }, [sessionId, userId]);

  return <PDFSessionContext.Provider value={{ sessionId, userId }}>{children}</PDFSessionContext.Provider>;
}
