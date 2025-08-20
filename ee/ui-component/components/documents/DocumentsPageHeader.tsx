"use client";

// import React, { useEffect } from "react"; // Removed - not needed
// import { useHeader } from "@/contexts/header-context"; // Removed - MorphikUI handles breadcrumbs

interface DocumentsPageHeaderProps {
  selectedFolder: string | null;
  onNavigateHome: () => void;
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
export function useDocumentsPageHeader({ selectedFolder, onNavigateHome }: DocumentsPageHeaderProps) {
  // Removed - MorphikUI handles breadcrumbs centrally
  // const { setCustomBreadcrumbs } = useHeader();
  // useEffect(() => {
  //   // Set breadcrumbs
  //   const breadcrumbs = selectedFolder
  //     ? [
  //         {
  //           label: "Documents",
  //           onClick: (e: React.MouseEvent) => {
  //             e.preventDefault();
  //             onNavigateHome();
  //           },
  //         },
  //         { label: selectedFolder === "all" ? "All Documents" : selectedFolder },
  //       ]
  //     : [{ label: "Documents" }];
  //   setCustomBreadcrumbs(breadcrumbs);
  //   // Cleanup on unmount
  //   return () => {
  //     setCustomBreadcrumbs(null);
  //   };
  // }, [selectedFolder, onNavigateHome, setCustomBreadcrumbs]);
}
