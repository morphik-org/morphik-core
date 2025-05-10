'use client';

import { useState, useEffect, useCallback } from 'react';
import {
  getConnectorAuthStatus,
  initiateConnectorAuth,
  disconnectConnector,
  ingestConnectorFile,
  type ConnectorAuthStatus,
} from '@/lib/connectorsApi';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { PlugZap, Unplug, AlertCircle, Loader2, FileText } from 'lucide-react';
import { FileBrowser } from './FileBrowser';

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

  const fetchStatus = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const status = await getConnectorAuthStatus(apiBaseUrl, connectorType);
      setAuthStatus(status);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred while fetching status.');
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
      initiateConnectorAuth(apiBaseUrl, connectorType);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to initiate connection.');
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
      setError(err instanceof Error ? err.message : 'Failed to disconnect.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleFileIngest = async (fileId: string, fileName: string, ingestedConnectorType: string) => {
    if (ingestedConnectorType !== connectorType) return;

    setIsSubmitting(true);
    setError(null);
    try {
      const result = await ingestConnectorFile(apiBaseUrl, connectorType, fileId, displayName);
      console.log('Ingestion successfully queued:', result);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to ingest file.';
      setError(errorMessage);
      console.error('Ingestion error:', errorMessage);
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
        <CardDescription>
          Manage your connection and browse files from the {displayName} service.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className={`p-4 rounded-lg border min-h-[60px] ${authStatus?.is_authenticated ? 'bg-green-50 dark:bg-green-900/30 border-green-200 dark:border-green-700' : 'bg-gray-50 dark:bg-gray-800/30'}`}>
          <div className="flex justify-between items-center">
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
                    {authStatus.is_authenticated ? 'Connected' : 'Not Connected'}
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
                    {isSubmitting && !error ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Unplug className="mr-2 h-4 w-4" />}
                    Disconnect
                  </Button>
                ) : (
                  <Button onClick={handleConnect} disabled={isSubmitting || !authStatus.auth_url}>
                    {isSubmitting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <PlugZap className="mr-2 h-4 w-4" />}
                    Connect
                  </Button>
                )}
              </div>
            )}
          </div>
        </div>

        {!isLoading && authStatus?.is_authenticated && (
          <div className="mt-4">
            <h3 className="text-lg font-semibold mb-2">Browse Files</h3>
            <FileBrowser
              connectorType={connectorType}
              apiBaseUrl={apiBaseUrl}
              onFileIngest={handleFileIngest}
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
}
