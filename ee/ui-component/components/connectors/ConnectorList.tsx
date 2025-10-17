"use client";

import { ConnectorCard } from "./ConnectorCard";
import { BookLock, BookOpen } from "lucide-react";
import { GitHub } from "../chat/icons"; // Import our custom GitHub icon
// import { useHeader } from "@/contexts/header-context"; // Removed - MorphikUI handles breadcrumbs
// import { useEffect } from "react"; // Removed - not needed

// In the future, this could come from a configuration or an API call
const availableConnectors = [
  {
    connectorType: "google_drive",
    displayName: "Google Drive",
    icon: BookLock, // Using an appropriate icon from lucide-react
    description: "Access files and folders from your Google Drive.",
  },
  {
    connectorType: "github",
    displayName: "GitHub",
    icon: GitHub,
    description: "Access repositories and files from GitHub.",
  },
  {
    connectorType: "zotero",
    displayName: "Zotero",
    icon: BookOpen, // Using an appropriate icon for Zotero
    description: "Access your Zotero library and research papers.",
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
  authToken: string | null;
}

export function ConnectorList({ apiBaseUrl, authToken }: ConnectorListProps) {
  // const { setCustomBreadcrumbs } = useHeader();

  // Removed - MorphikUI handles breadcrumbs centrally
  // useEffect(() => {
  //   setCustomBreadcrumbs([{ label: "Home", href: "/" }, { label: "Connectors" }]);
  //   return () => setCustomBreadcrumbs(null);
  // }, [setCustomBreadcrumbs]);

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
          authToken={authToken} // Pass authToken down
        />
      ))}
    </div>
  );
}
