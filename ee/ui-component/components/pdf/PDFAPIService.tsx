"use client";

import { useEffect } from "react";

interface PDFAPIServiceProps {
  children: React.ReactNode;
}

export function PDFAPIService({ children }: PDFAPIServiceProps) {
  useEffect(() => {
    console.log("Setting up PDF API service...");
    let eventSource: EventSource | null = null;
    let reconnectTimer: NodeJS.Timeout | null = null;
    let connectionAttempts = 0;
    const maxReconnectAttempts = 5;
    const baseReconnectDelay = 1000; // 1 second

    const connectToEventSource = () => {
      try {
        console.log(`Attempting to connect to EventSource (attempt ${connectionAttempts + 1}/${maxReconnectAttempts})`);

        if (eventSource) {
          eventSource.close();
        }

        eventSource = new EventSource("/api/pdf/events");
        console.log("EventSource created for /api/pdf/events");

        eventSource.onopen = () => {
          console.log("PDF API EventSource connection opened successfully");
          connectionAttempts = 0; // Reset counter on successful connection
        };

        eventSource.onmessage = event => {
          console.log("Received PDF command:", event.data);
          try {
            const command = JSON.parse(event.data);
            console.log("Parsed command:", command);

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
          console.error("PDF API EventSource error:", error);

          if (eventSource) {
            eventSource.close();
            eventSource = null;
          }

          // Attempt reconnection with exponential backoff
          if (connectionAttempts < maxReconnectAttempts) {
            connectionAttempts++;
            const delay = baseReconnectDelay * Math.pow(2, connectionAttempts - 1);
            console.log(
              `Attempting to reconnect in ${delay}ms (attempt ${connectionAttempts}/${maxReconnectAttempts})`
            );

            reconnectTimer = setTimeout(() => {
              connectToEventSource();
            }, delay);
          } else {
            console.error("Max reconnection attempts reached. PDF API service unavailable.");
          }
        };
      } catch (error) {
        console.error("Error setting up PDF API service:", error);
      }
    };

    // Initial connection
    connectToEventSource();

    // Cleanup function
    return () => {
      console.log("Cleaning up PDF API service...");

      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }

      if (eventSource) {
        console.log("Closing PDF API EventSource");
        eventSource.close();
        eventSource = null;
      }
    };
  }, []);

  return <>{children}</>;
}
