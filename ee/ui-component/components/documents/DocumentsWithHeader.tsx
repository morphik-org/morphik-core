"use client";

import React, { useEffect, useState, useCallback, useRef } from "react";
import { useHeader } from "@/contexts/header-context";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Layers, Trash2, Upload, RefreshCw, PlusCircle, ChevronsDown, ChevronsUp } from "lucide-react";
import DocumentsSection from "./DocumentsSection";
import { useWorkflowManagement } from "./shared/CommonHooks";
import WorkflowDialogs from "./shared/WorkflowDialogs";

interface DocumentsWithHeaderProps {
  apiBaseUrl: string;
  authToken: string | null;
  initialFolder?: string | null;
  onDocumentUpload?: (fileName: string, fileSize: number) => void;
  onDocumentDelete?: (fileName: string) => void;
  onDocumentClick?: (fileName: string) => void;
  onFolderClick?: (folderName: string | null) => void;
  onFolderCreate?: (folderName: string) => void;
  onRefresh?: () => void;
  onViewInPDFViewer?: (documentId: string) => void;
}

export default function DocumentsWithHeader(props: DocumentsWithHeaderProps) {
  const { setCustomBreadcrumbs, setRightContent } = useHeader();
  const [selectedFolder, setSelectedFolder] = useState<string | null>(props.initialFolder || null);
  const [showUploadDialog, setShowUploadDialog] = useState(false);
  const [showNewFolderDialog, setShowNewFolderDialog] = useState(false);
  const [allFoldersExpanded, setAllFoldersExpanded] = useState(false);
  const [folders, setFolders] = useState<Array<{ id: string; name: string }>>([]);

  // Create a ref to access DocumentsSection methods
  const documentsSectionRef = useRef<{
    handleRefresh: () => void;
    handleDeleteMultipleDocuments: () => void;
    selectedDocuments: string[];
  } | null>(null);

  // Use workflow management hook
  const {
    folderWorkflows,
    loadingWorkflows,
    showWorkflowDialog,
    setShowWorkflowDialog,
    availableWorkflows,
    showAddWorkflowDialog,
    setShowAddWorkflowDialog,
    selectedWorkflowToAdd,
    setSelectedWorkflowToAdd,
    fetchFolderWorkflows,
    fetchAvailableWorkflows,
    addWorkflow,
    removeWorkflow,
  } = useWorkflowManagement(props.apiBaseUrl, props.authToken, selectedFolder, folders);

  // Handle folder changes from DocumentsSection
  const handleFolderClick = useCallback(
    (folderName: string | null) => {
      setSelectedFolder(folderName);
      props.onFolderClick?.(folderName);
    },
    [props]
  );

  // Handle refresh
  const handleRefresh = useCallback(() => {
    if (documentsSectionRef.current?.handleRefresh) {
      documentsSectionRef.current.handleRefresh();
    }
    props.onRefresh?.();
  }, [props]);

  // Handle delete multiple
  const handleDeleteMultiple = useCallback(() => {
    if (documentsSectionRef.current?.handleDeleteMultipleDocuments) {
      documentsSectionRef.current.handleDeleteMultipleDocuments();
    }
  }, []);

  // Update header when folder changes
  useEffect(() => {
    // Set breadcrumbs based on current folder
    const breadcrumbs = selectedFolder
      ? [
          {
            label: "Documents",
            onClick: () => {
              setSelectedFolder(null);
              handleFolderClick(null);
            },
          },
          { label: selectedFolder === "all" ? "All Documents" : selectedFolder, current: true },
        ]
      : [{ label: "Documents", current: true }];

    setCustomBreadcrumbs(breadcrumbs);

    // Set right content based on current view
    const rightContent = selectedFolder ? (
      // Folder view controls
      <>
        {selectedFolder !== "all" && (
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              setShowWorkflowDialog(true);
              if (selectedFolder && folders.length > 0) {
                const folder = folders.find(f => f.name === selectedFolder);
                if (folder) {
                  fetchFolderWorkflows(folder.id);
                }
              }
            }}
            className="flex items-center gap-2"
          >
            <Layers className="h-4 w-4" />
            <span>Workflows</span>
            {folderWorkflows.length > 0 && (
              <Badge variant="secondary" className="ml-1 px-1.5 py-0.5 text-xs">
                {folderWorkflows.length}
              </Badge>
            )}
          </Button>
        )}

        {documentsSectionRef.current && documentsSectionRef.current.selectedDocuments.length > 0 && (
          <Button
            variant="outline"
            size="icon"
            onClick={handleDeleteMultiple}
            className="h-8 w-8 border-red-200 text-red-500 hover:border-red-300 hover:bg-red-50"
            title={`Delete ${documentsSectionRef.current.selectedDocuments.length} selected document${
              documentsSectionRef.current.selectedDocuments.length > 1 ? "s" : ""
            }`}
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        )}

        <Button variant="outline" size="sm" onClick={handleRefresh} title="Refresh documents">
          <RefreshCw className="h-4 w-4" />
          <span className="ml-1">Refresh</span>
        </Button>

        <Button variant="default" size="sm" onClick={() => setShowUploadDialog(true)}>
          <Upload className="mr-2 h-4 w-4" />
          Upload
        </Button>
      </>
    ) : (
      // Root level controls
      <>
        <Button variant="outline" size="sm" onClick={() => setShowNewFolderDialog(true)}>
          <PlusCircle className="mr-2 h-4 w-4" />
          New Folder
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => setAllFoldersExpanded(prev => !prev)}
          className="flex items-center gap-1.5"
          title="Expand or collapse all folders"
        >
          {allFoldersExpanded ? <ChevronsUp className="h-4 w-4" /> : <ChevronsDown className="h-4 w-4" />}
          <span>{allFoldersExpanded ? "Collapse All" : "Expand All"}</span>
        </Button>
        <Button variant="outline" size="sm" onClick={handleRefresh} title="Refresh documents">
          <RefreshCw className="h-4 w-4" />
          <span className="ml-1">Refresh</span>
        </Button>
        <Button variant="default" size="sm" onClick={() => setShowUploadDialog(true)}>
          <Upload className="mr-2 h-4 w-4" />
          Upload
        </Button>
      </>
    );

    setRightContent(rightContent);

    // Cleanup on unmount
    return () => {
      setCustomBreadcrumbs(null);
      setRightContent(null);
    };
  }, [
    selectedFolder,
    folderWorkflows.length,
    folders,
    allFoldersExpanded,
    handleFolderClick,
    handleRefresh,
    handleDeleteMultiple,
    setCustomBreadcrumbs,
    setRightContent,
    setShowWorkflowDialog,
    fetchFolderWorkflows,
  ]);

  // Callback to receive folders from DocumentsSection
  const handleFoldersUpdate = useCallback((newFolders: Array<{ id: string; name: string }>) => {
    setFolders(newFolders);
  }, []);

  return (
    <>
      <DocumentsSection
        {...props}
        ref={documentsSectionRef}
        onFolderClick={handleFolderClick}
        showUploadDialog={showUploadDialog}
        setShowUploadDialog={setShowUploadDialog}
        showNewFolderDialog={showNewFolderDialog}
        setShowNewFolderDialog={setShowNewFolderDialog}
        onFoldersUpdate={handleFoldersUpdate}
        allFoldersExpanded={allFoldersExpanded}
      />

      {/* Workflow dialogs */}
      <WorkflowDialogs
        showWorkflowDialog={showWorkflowDialog}
        setShowWorkflowDialog={setShowWorkflowDialog}
        showAddWorkflowDialog={showAddWorkflowDialog}
        setShowAddWorkflowDialog={setShowAddWorkflowDialog}
        folderWorkflows={folderWorkflows}
        loadingWorkflows={loadingWorkflows}
        availableWorkflows={availableWorkflows}
        selectedWorkflowToAdd={selectedWorkflowToAdd}
        setSelectedWorkflowToAdd={setSelectedWorkflowToAdd}
        selectedFolder={selectedFolder}
        apiBaseUrl={props.apiBaseUrl}
        authToken={props.authToken}
        onFetchFolderWorkflows={fetchFolderWorkflows}
        onFetchAvailableWorkflows={fetchAvailableWorkflows}
        onAddWorkflow={addWorkflow}
        onRemoveWorkflow={removeWorkflow}
        folders={folders}
      />
    </>
  );
}
