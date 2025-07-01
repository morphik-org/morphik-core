"use client";

import { useState, useEffect, useCallback } from "react";
import {
  getConnectorAuthStatus,
  initiateConnectorAuth,
  disconnectConnector,
  ingestConnectorFile,
  submitManualCredentials,
  type ConnectorAuthStatus,
  type CredentialField,
} from "@/lib/connectorsApi";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { PlugZap, Unplug, AlertCircle, Loader2, FileText } from "lucide-react";
import { FileBrowser } from "./FileBrowser";
import { Textarea } from "@/components/ui/textarea";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter, DialogClose } from "@/components/ui/dialog";

interface ConnectorCardProps {
  connectorType: string;
  displayName: string;
  icon?: React.ElementType;
  apiBaseUrl: string;
  authToken: string | null;
}

export function ConnectorCard({
  connectorType,
  displayName,
  icon: ConnectorIcon,
  apiBaseUrl,
  authToken,
}: ConnectorCardProps) {
  const [authStatus, setAuthStatus] = useState<ConnectorAuthStatus | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [showFileBrowserModal, setShowFileBrowserModal] = useState<boolean>(false);
  const [showIngestionModal, setShowIngestionModal] = useState<boolean>(false);

  // State for ingestion modal
  const [ingestionTargetFileId, setIngestionTargetFileId] = useState<string | null>(null);
  const [ingestionTargetFileName, setIngestionTargetFileName] = useState<string | null>(null);
  const [ingestionMetadata, setIngestionMetadata] = useState<string>("{}");
  const [ingestionRules, setIngestionRules] = useState<string>("[]");

  // State for manual credentials modal
  const [showCredentialsModal, setShowCredentialsModal] = useState<boolean>(false);
  const [credentialFields, setCredentialFields] = useState<CredentialField[]>([]);
  const [credentialValues, setCredentialValues] = useState<Record<string, string>>({});
  const [credentialInstructions, setCredentialInstructions] = useState<string>("");

  const fetchStatus = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const status = await getConnectorAuthStatus(apiBaseUrl, connectorType, authToken);
      setAuthStatus(status);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unknown error occurred while fetching status.");
      setAuthStatus(null);
    } finally {
      setIsLoading(false);
    }
  }, [apiBaseUrl, connectorType, authToken]);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);



  const handleConnect = async () => {
    setError(null);
    setIsSubmitting(true);
    try {
      // Construct the redirect URI to point to the main page with the connections section active
      const connectionsSectionUrl = new URL(window.location.origin);
      connectionsSectionUrl.pathname = "/"; // Ensure we are at the root path
      connectionsSectionUrl.searchParams.set("section", "connections");

      const authResponse = await initiateConnectorAuth(
        apiBaseUrl,
        connectorType,
        connectionsSectionUrl.toString(),
        authToken
      );

      // Check if this is a manual credentials flow
      if ("auth_type" in authResponse && authResponse.auth_type === "manual_credentials") {
        // Handle manual credentials flow
        setCredentialFields(authResponse.required_fields);
        setCredentialInstructions(authResponse.instructions || "");
        // Initialize credential values
        const initialValues: Record<string, string> = {};
        authResponse.required_fields.forEach(field => {
          initialValues[field.name] = "";
        });
        setCredentialValues(initialValues);
        setShowCredentialsModal(true);
      }
      // For OAuth flows, the function already handles redirection
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to initiate connection.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleSubmitCredentials = async () => {
    setError(null);
    setIsSubmitting(true);
    try {
      // Validate required fields
      const missingFields = credentialFields
        .filter(field => field.required && !credentialValues[field.name]?.trim())
        .map(field => field.label);

      if (missingFields.length > 0) {
        throw new Error(`Please fill in the following required fields: ${missingFields.join(", ")}`);
      }

      await submitManualCredentials(apiBaseUrl, connectorType, credentialValues, authToken);
      setShowCredentialsModal(false);
      await fetchStatus(); // Refresh status to show connected state
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit credentials.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDisconnect = async () => {
    setError(null);
    setIsSubmitting(true);
    try {
      await disconnectConnector(apiBaseUrl, connectorType, authToken);
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

  const handleRepositoryIngest = async (repoPath: string, ingestedConnectorType: string) => {
    if (ingestedConnectorType !== connectorType) return;
    
    setIsSubmitting(true);
    setError(null);
    
    try {
      const response = await fetch(`${apiBaseUrl}/ee/connectors/${ingestedConnectorType}/ingest-repository`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`,
        },
        body: JSON.stringify({
          connector_type: ingestedConnectorType,
          repo_path: repoPath,
          folder_name: 'github-repos', // Or use a state-managed folder name
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to ingest repository');
      }

      const result = await response.json();
      
      console.log('Repository ingested successfully:', result);
      
      // Show success message with details
      const docCount = result.documents?.length || 0;
      const successMessage = `Successfully ingested repository "${repoPath}"! Created ${docCount} document${docCount !== 1 ? 's' : ''}.`;
      alert(successMessage);
      
      // Close the file browser modal on success
      setShowFileBrowserModal(false);
      
    } catch (error) {
      console.error('Error ingesting repository:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error during repository ingestion';
      setError(errorMessage);
      alert(`Failed to ingest repository "${repoPath}": ${errorMessage}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleConfirmFileIngest = async () => {
    if (!ingestionTargetFileId || !ingestionTargetFileName) return;

    setIsSubmitting(true);
    setError(null);
    try {
      // Pass metadata and rules to ingestConnectorFile
      const result = await ingestConnectorFile(apiBaseUrl, connectorType, authToken, ingestionTargetFileId, {
        metadata: JSON.parse(ingestionMetadata),
        rules: JSON.parse(ingestionRules),
        // morphikFolderName and morphikEndUserId can be added here if there are UI elements to collect them
        // For now, they will be undefined and thus not sent if not explicitly set.
      });
      console.log("Ingestion successfully queued:", result);
      
      // Show success message
      alert(`Successfully started ingestion for "${ingestionTargetFileName}"! Document ID: ${result.morphik_document_id || result.document_id || 'N/A'}`);
      
      // Close modal on success
      setShowIngestionModal(false);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to ingest file.";
      setError(errorMessage);
      console.error("Ingestion error:", errorMessage);
      alert(`Failed to ingest "${ingestionTargetFileName}": ${errorMessage}`);
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
        ) : !isLoading && !error && authStatus && !authStatus.is_authenticated ? (
          <div className="flex min-h-[60px] items-center justify-center rounded-lg border bg-gray-50 p-4 dark:bg-gray-800/30">
            <Button onClick={handleConnect} disabled={isSubmitting} size="lg">
              {isSubmitting ? <Loader2 className="mr-2 h-5 w-5 animate-spin" /> : <PlugZap className="mr-2 h-5 w-5" />}
              Connect to {displayName}
            </Button>
          </div>
        ) : (
          <div className="rounded-lg border bg-gray-50 p-4 dark:bg-gray-800/30">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                {isLoading && (
                  <div className="flex items-center space-x-2 text-sm">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span>Checking connection status...</span>
                  </div>
                )}

                {!isLoading && authStatus?.is_authenticated && (
                  <div className="flex items-center space-x-2 text-sm text-green-700 dark:text-green-400">
                    <div className="h-2 w-2 rounded-full bg-green-500"></div>
                    <span>{authStatus.message || `Connected to ${displayName}`}</span>
                  </div>
                )}

                {!isLoading && authStatus && !authStatus.is_authenticated && (
                  <div className="flex items-center space-x-2 text-sm text-gray-500">
                    <div className="h-2 w-2 rounded-full bg-gray-400"></div>
                    <span>{authStatus.message || `Not connected to ${displayName}`}</span>
                  </div>
                )}

                {error && (
                  <div className="flex items-center space-x-2 text-sm text-red-600 dark:text-red-400">
                    <AlertCircle className="h-4 w-4" />
                    <span>{error}</span>
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
                    <Button onClick={handleConnect} disabled={isSubmitting}>
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

      {/* Manual Credentials Modal */}
      <Dialog open={showCredentialsModal} onOpenChange={setShowCredentialsModal}>
        <DialogContent className="sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle>Connect to {displayName}</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-4">
            {credentialInstructions && (
              <div className="rounded-md bg-blue-50 p-3 text-sm text-blue-700 dark:bg-blue-950/30 dark:text-blue-300">
                {credentialInstructions}
              </div>
            )}

            {credentialFields.map(field => (
              <div key={field.name} className="space-y-2">
                <Label htmlFor={field.name}>
                  {field.label}
                  {field.required && <span className="text-red-500">*</span>}
                </Label>
                {field.description && <p className="text-sm text-gray-600 dark:text-gray-400">{field.description}</p>}

                {field.type === "select" ? (
                  <Select
                    value={credentialValues[field.name] || ""}
                    onValueChange={value => setCredentialValues(prev => ({ ...prev, [field.name]: value }))}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder={`Select ${field.label}`} />
                    </SelectTrigger>
                    <SelectContent>
                      {field.options?.map(option => (
                        <SelectItem key={option.value} value={option.value}>
                          {option.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                ) : (
                  <Input
                    id={field.name}
                    type={field.type}
                    value={credentialValues[field.name] || ""}
                    onChange={e => setCredentialValues(prev => ({ ...prev, [field.name]: e.target.value }))}
                    placeholder={field.description}
                    required={field.required}
                  />
                )}
              </div>
            ))}

            {error && (
              <div className="rounded-md border border-red-200 bg-red-50 p-2 text-sm text-red-700 dark:border-red-700 dark:bg-red-900/30 dark:text-red-300">
                <AlertCircle className="mr-1 inline h-4 w-4" /> {error}
              </div>
            )}
          </div>
          <DialogFooter>
            <DialogClose asChild>
              <Button variant="outline">Cancel</Button>
            </DialogClose>
            <Button onClick={handleSubmitCredentials} disabled={isSubmitting}>
              {isSubmitting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
              Connect
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* File Browser Modal */}
      <Dialog open={showFileBrowserModal} onOpenChange={setShowFileBrowserModal}>
        <DialogContent className="max-w-4xl h-[80vh]">
          <DialogHeader>
            <DialogTitle>Browse {displayName}</DialogTitle>
          </DialogHeader>
          <div className="h-full overflow-y-auto">
            <FileBrowser
              connectorType={connectorType}
              apiBaseUrl={apiBaseUrl}
              authToken={authToken}
              onFileIngest={handleFileIngest}
              onRepositoryIngest={handleRepositoryIngest}
            />
          </div>
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
