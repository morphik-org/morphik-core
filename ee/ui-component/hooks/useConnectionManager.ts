import { useCallback } from "react";
import { useMorphik } from "@/contexts/morphik-context";
import { clearLocalConnectionFromStorage } from "@/lib/connection-utils";

/**
 * Hook for managing connection state and clearing local connections
 */
export function useConnectionManager() {
  const { connectionInfo, updateConnectionUri, isLocal } = useMorphik();

  /**
   * Clear the current connection
   */
  const clearConnection = useCallback(() => {
    console.log("Clearing connection");
    updateConnectionUri("");

    // Also clear from localStorage if it's local
    if (isLocal) {
      clearLocalConnectionFromStorage();
    }
  }, [updateConnectionUri, isLocal]);

  /**
   * Reset to default local connection
   */
  const resetToDefault = useCallback(() => {
    console.log("Resetting to default local connection");
    updateConnectionUri("localhost:8000");
  }, [updateConnectionUri]);

  /**
   * Clear and reconnect (useful for debugging)
   */
  const reconnect = useCallback(() => {
    const currentUri = connectionInfo?.originalUri;
    clearConnection();

    // Small delay to ensure state updates
    setTimeout(() => {
      if (currentUri) {
        updateConnectionUri(currentUri);
      } else {
        resetToDefault();
      }
    }, 100);
  }, [connectionInfo, clearConnection, updateConnectionUri, resetToDefault]);

  return {
    connectionInfo,
    isLocal,
    clearConnection,
    resetToDefault,
    reconnect,
  };
}
