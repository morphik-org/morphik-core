"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { Trash2 } from "lucide-react";

interface BreadcrumbNavigationProps {
  selectedFolder: string | null;
  onNavigateHome: () => void;
  selectedDocuments?: string[];
  onDeleteMultiple?: () => void;
  refreshAction?: () => void;
  uploadDialogComponent?: React.ReactNode;
}

const BreadcrumbNavigation: React.FC<BreadcrumbNavigationProps> = ({
  selectedFolder,
  onNavigateHome,
  selectedDocuments = [],
  onDeleteMultiple,
  refreshAction,
  uploadDialogComponent,
}) => {
  if (selectedFolder === null) return null;

  return (
    <div className="border-border bg-background p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <button
            onClick={onNavigateHome}
            className="flex items-center space-x-1 text-muted-foreground transition-colors hover:text-foreground"
          >
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
              className="h-4 w-4"
            >
              <path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
              <polyline points="9,22 9,12 15,12 15,22" />
            </svg>
            <span>Morphik</span>
          </button>
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
            className="h-4 w-4 text-muted-foreground"
          >
            <path d="m9 18 6-6-6-6" />
          </svg>
          <span className="font-medium text-foreground">
            {selectedFolder === "all" ? "All Documents" : selectedFolder}
          </span>
        </div>
        <div className="flex items-center space-x-2">
          {selectedDocuments.length > 0 && onDeleteMultiple && (
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="icon"
                onClick={onDeleteMultiple}
                className="h-8 w-8 border-red-200 text-red-500 hover:border-red-300 hover:bg-red-50"
                title={`Delete ${selectedDocuments.length} selected document${selectedDocuments.length > 1 ? "s" : ""}`}
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
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
        </div>
      </div>
    </div>
  );
};

export default BreadcrumbNavigation;
