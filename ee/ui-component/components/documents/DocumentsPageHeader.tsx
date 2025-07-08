"use client";

import React, { useEffect } from "react";
import { useHeader } from "@/contexts/header-context";

interface DocumentsPageHeaderProps {
  selectedFolder: string | null;
  onNavigateHome: () => void;
}

export function useDocumentsPageHeader({ selectedFolder, onNavigateHome }: DocumentsPageHeaderProps) {
  const { setCustomBreadcrumbs } = useHeader();

  useEffect(() => {
    // Set breadcrumbs
    const breadcrumbs = selectedFolder
      ? [
          {
            label: "Documents",
            onClick: (e: React.MouseEvent) => {
              e.preventDefault();
              onNavigateHome();
            },
          },
          { label: selectedFolder === "all" ? "All Documents" : selectedFolder },
        ]
      : [{ label: "Documents" }];

    setCustomBreadcrumbs(breadcrumbs);

    // Cleanup on unmount
    return () => {
      setCustomBreadcrumbs(null);
    };
  }, [selectedFolder, onNavigateHome, setCustomBreadcrumbs]);
}
