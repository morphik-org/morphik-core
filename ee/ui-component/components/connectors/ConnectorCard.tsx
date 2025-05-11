"use client";

import { useState, useEffect, useCallback } from "react";
import {
  getConnectorAuthStatus,
  initiateConnectorAuth,
  disconnectConnector,
  ingestConnectorFile,
  type ConnectorAuthStatus,
} from "@/lib/connectorsApi";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { PlugZap, Unplug, AlertCircle, Loader2, FileText } from "lucide-react";
import { FileBrowser } from "./FileBrowser";
import { Textarea } from "@/components/ui/textarea";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter, DialogClose } from "@/components/ui/dialog";

interface ConnectorCardProps {
  connectorType: string;
  displayName: string;
  icon?: React.ElementType;
  apiBaseUrl: string;
}

export function ConnectorCard({ connectorType, displayName, icon: ConnectorIcon, apiBaseUrl }: ConnectorCardProps) {
  const [authStatus, setAuthStatus] = useState<ConnectorAuthStatus | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [showFileBrowserModal, setShowFileBrowserModal] = useState<boolean>(false);

  // State for ingestion modal
  const [showIngestionModal, setShowIngestionModal] = useState<boolean>(false);
  const [ingestionTargetFileId, setIngestionTargetFileId] = useState<string | null>(null);
  const [ingestionTargetFileName, setIngestionTargetFileName] = useState<string | null>(null);
  const [ingestionMetadata, setIngestionMetadata] = useState<string>("{}"); // Default to empty JSON object
  const [ingestionRules, setIngestionRules] = useState<string>("[]"); // Default to empty JSON array

  const fetchStatus = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const status = await getConnectorAuthStatus(apiBaseUrl, connectorType);
      setAuthStatus(status);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unknown error occurred while fetching status.");
      setAuthStatus(null);
    } finally {
      setIsLoading(false);
    }
  }, [apiBaseUrl, connectorType]);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  const handleConnect = () => {
    setError(null);
    setIsSubmitting(true);
    try {
      // Construct the redirect URI to point to the main page with the connections section active
      const connectionsSectionUrl = new URL(window.location.origin);
      connectionsSectionUrl.pathname = "/"; // Ensure we are at the root path
      connectionsSectionUrl.searchParams.set("section", "connections");
      initiateConnectorAuth(apiBaseUrl, connectorType, connectionsSectionUrl.toString());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to initiate connection.");
      setIsSubmitting(false);
    }
  };

  const handleDisconnect = async () => {
    setError(null);
    setIsSubmitting(true);
    try {
      await disconnectConnector(apiBaseUrl, connectorType);
      await fetchStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to disconnect.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleFileIngest = async (fileId: string, fileName: string, ingestedConnectorType: string) => {
    if (ingestedConnectorType !== connectorType) return;

    // Set state for the modal instead of direct ingestion
    setIngestionTargetFileId(fileId);
    setIngestionTargetFileName(fileName);
    setIngestionMetadata("{}"); // Reset metadata
    setIngestionRules("[]"); // Reset rules
    setShowIngestionModal(true);
    setError(null); // Clear previous errors
  };

  const handleConfirmFileIngest = async () => {
    if (!ingestionTargetFileId || !ingestionTargetFileName) return;

    setIsSubmitting(true);
    setError(null);
    try {
      // Pass metadata and rules to ingestConnectorFile
      // Note: ingestConnectorFile in lib/connectorsApi.ts will need to be updated to accept these
      const result = await ingestConnectorFile(apiBaseUrl, connectorType, ingestionTargetFileId, {
        metadata: JSON.parse(ingestionMetadata),
        rules: JSON.parse(ingestionRules),
        // morphikFolderName and morphikEndUserId can be added here if there are UI elements to collect them
        // For now, they will be undefined and thus not sent if not explicitly set.
      });
      console.log("Ingestion successfully queued:", result);
      setShowIngestionModal(false); // Close modal on success
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to ingest file.";
      setError(errorMessage);
      console.error("Ingestion error:", errorMessage);
      // Keep modal open on error to allow correction or retry
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Card className="w-full max-w-2xl">
      <CardHeader>
        <CardTitle className="flex items-center">
          {ConnectorIcon ? <ConnectorIcon className="mr-2 h-6 w-6" /> : <FileText className="mr-2 h-6 w-6" />}
          {displayName}
        </CardTitle>
        <CardDescription>Manage your connection and browse files from the {displayName} service.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Simplified view for "Disconnected but Connectable" state */}
        {!isLoading && !error && authStatus && !authStatus.is_authenticated && authStatus.auth_url ? (
          <div className="flex min-h-[60px] items-center justify-center rounded-lg border bg-gray-50 p-4 dark:bg-gray-800/30">
            <Button onClick={handleConnect} disabled={isSubmitting || !authStatus.auth_url} size="lg">
              {isSubmitting ? <Loader2 className="mr-2 h-5 w-5 animate-spin" /> : <PlugZap className="mr-2 h-5 w-5" />}
              Connect to {displayName}
            </Button>
          </div>
        ) : (
          /* Original detailed view for other states (loading, error, connected, unconnectable) */
          <div
            className={`min-h-[60px] rounded-lg border p-4 ${authStatus?.is_authenticated ? "border-green-200 bg-green-50 dark:border-green-700 dark:bg-green-900/30" : "bg-gray-50 dark:bg-gray-800/30"}`}
          >
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-semibold">Connection Status</h3>
                {isLoading && (
                  <div className="flex items-center space-x-2 text-sm text-muted-foreground">
                    <Loader2 className="h-5 w-5 animate-spin" />
                    <span>Loading status...</span>
                  </div>
                )}
                {!isLoading && error && (
                  <div className="flex items-center space-x-2 text-sm text-red-600">
                    <AlertCircle className="h-5 w-5" />
                    <span>Error: {error}</span>
                  </div>
                )}
                {!isLoading && !error && authStatus && (
                  <div className="flex items-center space-x-2 text-sm">
                    {authStatus.is_authenticated ? (
                      <PlugZap className="h-5 w-5 text-green-600" />
                    ) : (
                      <Unplug className="h-5 w-5 text-gray-500" />
                    )}
                    <span>
                      {authStatus.is_authenticated ? "Connected" : "Not Connected"}
                      {authStatus.message && !authStatus.is_authenticated && ` - ${authStatus.message}`}
                    </span>
                  </div>
                )}
                {!isLoading && !error && !authStatus && (
                  <div className="flex items-center space-x-2 text-sm text-gray-500">
                    <AlertCircle className="h-5 w-5" />
                    <span>Status currently unavailable. Try refreshing.</span>
                  </div>
                )}
              </div>
              {!isLoading && authStatus && (
                <div>
                  {authStatus.is_authenticated ? (
                    <Button variant="outline" onClick={handleDisconnect} disabled={isSubmitting}>
                      {isSubmitting && !error ? (
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      ) : (
                        <Unplug className="mr-2 h-4 w-4" />
                      )}
                      Disconnect
                    </Button>
                  ) : (
                    <Button onClick={handleConnect} disabled={isSubmitting || !authStatus.auth_url}>
                      {isSubmitting ? (
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      ) : (
                        <PlugZap className="mr-2 h-4 w-4" />
                      )}
                      Connect
                    </Button>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Button to open the file browser modal */}
        {!isLoading && authStatus?.is_authenticated && (
          <div className="mt-4">
            <Button variant="outline" onClick={() => setShowFileBrowserModal(true)} className="mb-4">
              Open Files
            </Button>
          </div>
        )}
      </CardContent>

      {/* File Browser Modal */}
      <Dialog open={showFileBrowserModal} onOpenChange={setShowFileBrowserModal}>
        <DialogContent className="flex h-[80vh] w-[90vw] max-w-[1200px] flex-col">
          <DialogHeader>
            <DialogTitle>Browse Files: {displayName}</DialogTitle>
          </DialogHeader>
          <div className="flex-grow overflow-auto py-4">
            <FileBrowser connectorType={connectorType} apiBaseUrl={apiBaseUrl} onFileIngest={handleFileIngest} />
          </div>
          <DialogFooter className="mt-auto">
            <DialogClose asChild>
              <Button variant="outline">Close</Button>
            </DialogClose>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Ingestion Options Modal */}
      <Dialog open={showIngestionModal} onOpenChange={setShowIngestionModal}>
        <DialogContent className="sm:max-w-[625px]">
          <DialogHeader>
            <DialogTitle>Ingest File: {ingestionTargetFileName || "File"}</DialogTitle>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <label htmlFor="metadata" className="col-span-1 text-right">
                Metadata (JSON)
              </label>
              <Textarea
                id="metadata"
                value={ingestionMetadata}
                onChange={e => setIngestionMetadata(e.target.value)}
                className="col-span-3 h-24"
                placeholder='Enter metadata as JSON, e.g., { "source": "google_drive" }'
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <label htmlFor="rules" className="col-span-1 text-right">
                Rules (JSON)
              </label>
              <Textarea
                id="rules"
                value={ingestionRules}
                onChange={e => setIngestionRules(e.target.value)}
                className="col-span-3 h-24"
                placeholder='Enter rules as JSON array, e.g., [{ "type": "metadata_extraction", ... }]'
              />
            </div>
            {error && (
              <div className="col-span-4 rounded-md border border-red-200 bg-red-50 p-2 text-sm text-red-700 dark:border-red-700 dark:bg-red-900/30 dark:text-red-300">
                <AlertCircle className="mr-1 inline h-4 w-4" /> {error}
              </div>
            )}
          </div>
          <DialogFooter>
            <DialogClose asChild>
              <Button variant="outline">Cancel</Button>
            </DialogClose>
            <Button onClick={handleConfirmFileIngest} disabled={isSubmitting}>
              {isSubmitting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
              Confirm Ingest
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
}
