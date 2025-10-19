"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { Trash2 } from "lucide-react";
import { DynamicSiteHeader } from "../dynamic-site-header";

interface DocumentsHeaderProps {
  selectedFolder: string | null;
  onNavigateHome: () => void;
  selectedDocuments?: string[];
  onDeleteMultiple?: () => void;
  refreshAction?: () => void;
  uploadDialogComponent?: React.ReactNode;
}

export function DocumentsHeader({
  selectedFolder,
  selectedDocuments = [],
  onDeleteMultiple,
  refreshAction,
  uploadDialogComponent,
}: DocumentsHeaderProps) {
  // Generate breadcrumbs based on selected folder
  const breadcrumbs = selectedFolder
    ? [
        { label: "Documents", href: "/dashboard/documents" },
        { label: selectedFolder === "all" ? "All Documents" : selectedFolder },
      ]
    : [{ label: "Documents" }];

  // Right side content
  const rightContent = (
    <>
      {selectedDocuments.length > 0 && onDeleteMultiple && (
        <Button
          variant="outline"
          size="icon"
          onClick={onDeleteMultiple}
          className="h-8 w-8 border-red-200 text-red-500 hover:border-red-300 hover:bg-red-50"
          title={`Delete ${selectedDocuments.length} selected document${selectedDocuments.length > 1 ? "s" : ""}`}
        >
          <Trash2 className="h-4 w-4" />
        </Button>
      )}

      {refreshAction && (
        <Button variant="outline" size="sm" onClick={refreshAction} title="Refresh documents">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="mr-1"
          >
            <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"></path>
            <path d="M21 3v5h-5"></path>
            <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"></path>
            <path d="M8 16H3v5"></path>
          </svg>
          Refresh
        </Button>
      )}

      {uploadDialogComponent}
    </>
  );

  return <DynamicSiteHeader customBreadcrumbs={breadcrumbs} rightContent={rightContent} />;
}
