"use client";

import { ConnectorCard } from "./ConnectorCard";
import { BookLock } from "lucide-react"; // Example icon for Google Drive

// In the future, this could come from a configuration or an API call
const availableConnectors = [
  {
    connectorType: "google_drive",
    displayName: "Google Drive",
    icon: BookLock, // Using an appropriate icon from lucide-react
    description: "Access files and folders from your Google Drive.",
  },
  // Add other connectors here as they are implemented
  // {
  //   connectorType: 's3',
  //   displayName: 'Amazon S3',
  //   icon: SomeOtherIcon,
  //   description: 'Connect to your Amazon S3 buckets.'
  // },
];

interface ConnectorListProps {
  apiBaseUrl: string; // Added apiBaseUrl prop
}

export function ConnectorList({ apiBaseUrl }: ConnectorListProps) {
  if (availableConnectors.length === 0) {
    return (
      <div className="text-center text-muted-foreground">
        <p>No data connectors are currently available.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {availableConnectors.map(connector => (
        <ConnectorCard
          key={connector.connectorType}
          connectorType={connector.connectorType}
          displayName={connector.displayName}
          icon={connector.icon}
          apiBaseUrl={apiBaseUrl} // Pass apiBaseUrl down
        />
      ))}
    </div>
  );
}
