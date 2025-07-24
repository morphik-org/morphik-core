"use client";

import React, { useState, useMemo, useCallback } from "react";
import { Checkbox } from "@/components/ui/checkbox";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Eye,
  Download,
  Trash2,
  Copy,
  Check,
  Search,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  Folder as FolderIcon,
  FileText,
  ChevronRight,
  ChevronDown,
} from "lucide-react";
import { showAlert } from "@/components/ui/alert-system";

import { Document, Folder, FolderSummary } from "@/components/types";
import { EmptyDocuments, NoMatchingDocuments, LoadingDocuments } from "./shared/EmptyStates";

type ColumnType = "string" | "int" | "float" | "bool" | "Date" | "json";

interface DocumentListProps {
  documents: Document[];
  selectedDocument: Document | null;
  selectedDocuments: string[];
  handleDocumentClick: (document: Document) => void;
  handleCheckboxChange: (checked: boolean | "indeterminate", docId: string) => void;
  getSelectAllState: () => boolean | "indeterminate";
  setSelectedDocuments: (docIds: string[]) => void;
  setDocuments: (docs: Document[]) => void;
  loading: boolean;
  apiBaseUrl: string;
  authToken: string | null;
  selectedFolder?: string | null;
  onViewInPDFViewer?: (documentId: string) => void; // Add PDF viewer navigation
  onDownloadDocument?: (documentId: string) => void; // Add download functionality
  onDeleteDocument?: (documentId: string) => void; // Add delete functionality
  onDeleteMultipleDocuments?: () => void; // Add bulk delete functionality
  folders?: FolderSummary[]; // Optional since it's fetched internally
  showBorder?: boolean; // Control whether to show the outer border and rounded corners
  hideSearchBar?: boolean; // Control whether to hide the search bar
  externalSearchQuery?: string; // External search query when search bar is hidden
  onSearchChange?: (query: string) => void; // Callback for search changes when search bar is hidden
  allFoldersExpanded?: boolean; // Control whether all folders should be expanded
}

const DocumentList: React.FC<DocumentListProps> = React.memo(function DocumentList({
  documents,
  selectedDocument,
  selectedDocuments,
  handleDocumentClick,
  handleCheckboxChange,
  getSelectAllState,
  setSelectedDocuments,
  loading,
  apiBaseUrl,
  authToken,
  onViewInPDFViewer,
  onDownloadDocument,
  onDeleteDocument,
  onDeleteMultipleDocuments,
  showBorder = true,
  hideSearchBar = false,
  externalSearchQuery = "",
  onSearchChange,
  allFoldersExpanded = false,
}) {
  const [mounted, setMounted] = useState(false);

  React.useEffect(() => {
    setMounted(true);
  }, []);
  const [copiedDocumentId, setCopiedDocumentId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");

  // Use external search query when search bar is hidden
  const effectiveSearchQuery = hideSearchBar ? externalSearchQuery : searchQuery;
  const [sortColumn, setSortColumn] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");

  // State for expanded folders and their documents
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set());
  const [folderDocuments, setFolderDocuments] = useState<Record<string, Document[]>>({});

  // Get unique metadata fields from all documents, excluding external_id
  const existingMetadataFields = useMemo(() => {
    const fields = new Set<string>();
    documents.forEach(doc => {
      if (doc.metadata) {
        Object.keys(doc.metadata).forEach(key => {
          // Filter out external_id since we have a dedicated Document ID column
          if (key !== "external_id") {
            fields.add(key);
          }
        });
      }
    });
    return Array.from(fields);
  }, [documents]);

  // Apply search and sort logic with memoization, including expanded folder documents
  const filteredDocuments = useMemo(() => {
    let result: (Document & {
      itemType?: "document" | "folder" | "all";
      folderData?: Folder;
      isChildDocument?: boolean;
      parentFolderName?: string;
    })[] = [];

    // Add all main documents
    documents.forEach(doc => {
      result.push(doc);

      // If this is a folder and it's expanded, add its documents as children
      if ((doc as Document & { itemType?: string }).itemType === "folder") {
        const folderName = doc.filename || "";
        if (expandedFolders.has(folderName) && folderDocuments[folderName]) {
          folderDocuments[folderName].forEach(childDoc => {
            result.push({
              ...childDoc,
              isChildDocument: true,
              parentFolderName: folderName,
              itemType: "document",
            });
          });
        }
      }

      // Note: "All Documents" expansion now works by expanding all folders,
      // so we don't add separate children for it
    });

    // Apply search filter
    if (effectiveSearchQuery.trim()) {
      const query = effectiveSearchQuery.toLowerCase();
      result = result.filter(doc => {
        // Search in filename
        if (doc.filename?.toLowerCase().includes(query)) return true;

        // Search in document ID
        if (doc.external_id.toLowerCase().includes(query)) return true;

        // Search in metadata values
        if (doc.metadata) {
          for (const value of Object.values(doc.metadata)) {
            if (String(value).toLowerCase().includes(query)) return true;
          }
        }

        return false;
      });
    }

    // Apply sorting
    if (sortColumn) {
      result.sort((a, b) => {
        let aValue: string;
        let bValue: string;

        // Get values based on column
        if (sortColumn === "filename") {
          aValue = a.filename || "";
          bValue = b.filename || "";
        } else if (sortColumn === "external_id") {
          aValue = a.external_id;
          bValue = b.external_id;
        } else {
          // Metadata column
          const aMetaValue = a.metadata?.[sortColumn];
          const bMetaValue = b.metadata?.[sortColumn];

          // Handle different types of metadata values
          if (typeof aMetaValue === "object" && aMetaValue !== null) {
            aValue = JSON.stringify(aMetaValue);
          } else {
            aValue = String(aMetaValue ?? "");
          }

          if (typeof bMetaValue === "object" && bMetaValue !== null) {
            bValue = JSON.stringify(bMetaValue);
          } else {
            bValue = String(bMetaValue ?? "");
          }
        }

        // Convert to strings for comparison
        aValue = String(aValue).toLowerCase();
        bValue = String(bValue).toLowerCase();

        // Compare values
        if (aValue < bValue) return sortDirection === "asc" ? -1 : 1;
        if (aValue > bValue) return sortDirection === "asc" ? 1 : -1;
        return 0;
      });
    }

    return result;
  }, [documents, effectiveSearchQuery, sortColumn, sortDirection, expandedFolders, folderDocuments]);

  // Copy document ID to clipboard
  const copyDocumentId = async (documentId: string) => {
    try {
      await navigator.clipboard.writeText(documentId);
      setCopiedDocumentId(documentId);
      setTimeout(() => setCopiedDocumentId(null), 2000); // Reset after 2 seconds
    } catch (err) {
      console.error("Failed to copy document ID:", err);
      showAlert("Failed to copy document ID", { type: "error", duration: 3000 });
    }
  };

  // Fetch documents for a specific folder
  const fetchFolderDocuments = useCallback(
    async (folderName: string) => {
      if (folderDocuments[folderName]) {
        return; // Already fetched
      }

      try {
        // First get folder details to get the folder ID
        const foldersResponse = await fetch(`${apiBaseUrl}/folders/summary`, {
          headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
        });

        if (!foldersResponse.ok) {
          throw new Error(`Failed to fetch folders: ${foldersResponse.statusText}`);
        }

        const foldersData = await foldersResponse.json();
        const folder = foldersData.find((f: FolderSummary) => f.name === folderName);

        if (!folder) {
          console.warn(`Folder "${folderName}" not found`);
          return;
        }

        // Get document IDs for this folder
        let documentIds: string[] = [];
        if (Array.isArray(folder.document_ids)) {
          documentIds = folder.document_ids;
        } else {
          // Fetch detailed folder info if document_ids not in summary
          const folderDetailResponse = await fetch(`${apiBaseUrl}/folders/${folder.id}`, {
            headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
          });
          if (folderDetailResponse.ok) {
            const folderDetail = await folderDetailResponse.json();
            documentIds = Array.isArray(folderDetail.document_ids) ? folderDetail.document_ids : [];
          }
        }

        if (documentIds.length > 0) {
          // Fetch document details
          const docsResponse = await fetch(`${apiBaseUrl}/batch/documents`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
            },
            body: JSON.stringify({ document_ids: documentIds }),
          });

          if (docsResponse.ok) {
            const docs = await docsResponse.json();
            setFolderDocuments(prev => ({ ...prev, [folderName]: docs }));
          }
        } else {
          // Empty folder
          setFolderDocuments(prev => ({ ...prev, [folderName]: [] }));
        }
      } catch (error) {
        console.error(`Error fetching documents for folder "${folderName}":`, error);
        setFolderDocuments(prev => ({ ...prev, [folderName]: [] }));
      }
    },
    [apiBaseUrl, authToken, folderDocuments]
  );

  // Effect to handle allFoldersExpanded prop changes
  React.useEffect(() => {
    if (allFoldersExpanded) {
      // Expand all folders
      const allFolderNames = documents
        .filter((doc: Document & { itemType?: string }) => doc.itemType === "folder")
        .map((doc: Document & { itemType?: string }) => doc.filename || "");
      setExpandedFolders(new Set(allFolderNames));

      // Fetch documents for all folders that aren't already fetched
      allFolderNames.forEach(folderName => {
        if (!folderDocuments[folderName]) {
          fetchFolderDocuments(folderName);
        }
      });
    } else {
      // Collapse all folders
      setExpandedFolders(new Set());
    }
  }, [allFoldersExpanded, documents, folderDocuments, fetchFolderDocuments]);

  // Handle folder expansion toggle
  const toggleFolderExpansion = useCallback(
    (folderName: string, event: React.MouseEvent) => {
      event.stopPropagation(); // Prevent folder navigation

      setExpandedFolders(prev => {
        const newSet = new Set(prev);
        if (newSet.has(folderName)) {
          newSet.delete(folderName);
        } else {
          newSet.add(folderName);
          // Fetch documents when expanding
          fetchFolderDocuments(folderName);
        }
        return newSet;
      });
    },
    [fetchFolderDocuments]
  );

  // Use existing metadata fields as columns
  const allColumns = useMemo(() => {
    return existingMetadataFields.map(field => ({
      name: field,
      description: `Extracted ${field}`,
      _type: "string" as ColumnType,
    }));
  }, [existingMetadataFields]);

  // Handle column sorting
  const handleSort = useCallback(
    (column: string) => {
      if (sortColumn === column) {
        // If clicking the same column, toggle direction
        setSortDirection(prev => (prev === "asc" ? "desc" : "asc"));
      } else {
        // If clicking a different column, set it as the sort column with asc direction
        setSortColumn(column);
        setSortDirection("asc");
      }
    },
    [sortColumn]
  );

  // Base grid template for the scrollable part – exclude the Actions column.
  const gridTemplateColumns = useMemo(
    () => `48px minmax(200px, 350px) 160px ${allColumns.map(() => "140px").join(" ")}`,
    [allColumns]
  );

  const DocumentListHeader = () => {
    return (
      <div className="sticky top-0 z-20 border-b bg-muted font-medium">
        <div className="flex min-w-fit">
          {/* Main scrollable content */}
          <div className="grid flex-1 items-center" style={{ gridTemplateColumns }}>
            <div className="flex items-center justify-center px-3 py-2">
              <Checkbox
                id="select-all-documents"
                checked={getSelectAllState()}
                onCheckedChange={checked => {
                  if (checked === true) {
                    // Select all items (including folders)
                    setSelectedDocuments(documents.map(doc => doc.external_id));
                  } else {
                    setSelectedDocuments([]);
                  }
                }}
                aria-label="Select all documents"
              />
            </div>
            <div
              className="flex cursor-pointer items-center gap-1 px-3 py-2 text-sm font-semibold hover:bg-muted/50"
              onClick={() => handleSort("filename")}
            >
              Filename
              {sortColumn === "filename" &&
                (sortDirection === "asc" ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />)}
              {sortColumn !== "filename" && <ArrowUpDown className="h-3 w-3 opacity-30" />}
            </div>
            <div
              className="flex cursor-pointer items-center gap-1 px-3 py-2 text-sm font-semibold hover:bg-muted/50"
              onClick={() => handleSort("external_id")}
            >
              Document ID
              {sortColumn === "external_id" &&
                (sortDirection === "asc" ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />)}
              {sortColumn !== "external_id" && <ArrowUpDown className="h-3 w-3 opacity-30" />}
            </div>
            {allColumns.map(column => (
              <div
                key={column.name}
                className="flex max-w-[160px] cursor-pointer items-center gap-1 px-3 py-2 text-sm font-semibold hover:bg-muted/50"
                onClick={() => handleSort(column.name)}
              >
                <span className="truncate" title={column.name}>
                  {column.name}
                </span>
                {sortColumn === column.name ? (
                  sortDirection === "asc" ? (
                    <ArrowUp className="h-3 w-3 flex-shrink-0" />
                  ) : (
                    <ArrowDown className="h-3 w-3 flex-shrink-0" />
                  )
                ) : (
                  <ArrowUpDown className="h-3 w-3 flex-shrink-0 opacity-30" />
                )}
              </div>
            ))}
          </div>
          {/* Sticky Actions column */}
          <div className="sticky right-0 top-0 z-30 w-[120px] border-l bg-muted px-3 py-2 text-center text-sm font-semibold">
            Actions
          </div>
        </div>
      </div>
    );
  };

  if (loading && !documents.length) {
    return (
      <div
        className={`flex h-full w-full flex-col overflow-hidden ${showBorder ? "rounded-t-md border-l border-r border-t shadow-sm" : ""}`}
      >
        {/* Search Bar */}{" "}
        {!hideSearchBar && (
          <div className="border-b border-border bg-background p-3">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Search documents..."
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                className="pl-9"
              />
            </div>
          </div>
        )}
        <div className="flex-1 overflow-auto">
          {DocumentListHeader()}
          <LoadingDocuments />
          {/* Spacer to ensure container fills available height */}
          <div className="min-h-16 flex-1"></div>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`flex h-full w-full flex-col overflow-hidden ${showBorder ? "rounded-t-md border-l border-r border-t shadow-sm" : ""}`}
    >
      {/* Search Bar - Fixed at top */}
      {!hideSearchBar && (
        <div className="border-b border-border bg-background p-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search documents..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              className="pl-9"
            />
          </div>
        </div>
      )}

      {/* Bulk actions bar */}
      {mounted && selectedDocuments.length > 0 && (
        <div className="flex items-center justify-between border-b bg-muted/50 px-4 py-2">
          <span className="text-sm text-muted-foreground">
            {selectedDocuments.length} item{selectedDocuments.length > 1 ? "s" : ""} selected
          </span>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={() => setSelectedDocuments([])}>
              Clear selection
            </Button>
            {onDeleteMultipleDocuments && (
              <Button
                variant="destructive"
                size="sm"
                onClick={onDeleteMultipleDocuments}
                className="flex items-center gap-2"
              >
                <Trash2 className="h-4 w-4" />
                Delete selected
              </Button>
            )}
          </div>
        </div>
      )}

      {/* Main content area with horizontal scroll */}
      <div className="min-h-0 flex-1 overflow-x-auto overflow-y-auto">
        {/* Header */}
        {DocumentListHeader()}

        {/* Content rows */}
        {filteredDocuments.map(doc => (
          <div
            key={`${doc.external_id}${(doc as Document & { isChildDocument?: boolean; parentFolderName?: string }).isChildDocument ? `-child-${(doc as Document & { isChildDocument?: boolean; parentFolderName?: string }).parentFolderName}` : ""}`}
            onClick={() => {
              // Handle different item types
              if ((doc as Document & { itemType?: string }).itemType === "folder") {
                // Navigate to folder when clicking on folder row (but not on chevron)
                handleDocumentClick(doc);
              } else {
                // Handle document clicks for actual documents
                handleDocumentClick(doc);
              }
            }}
            className={`relative flex min-w-fit border-b border-border ${
              (doc as Document & { itemType?: string }).itemType === "folder"
                ? "cursor-pointer hover:bg-muted/50"
                : doc.external_id === selectedDocument?.external_id
                  ? "cursor-pointer bg-primary/10 hover:bg-primary/15"
                  : "cursor-pointer hover:bg-muted/70"
            } ${(doc as Document & { isChildDocument?: boolean }).isChildDocument ? "bg-gray-50 dark:bg-gray-900" : ""}`}
            style={
              {
                // no-op for flex container
              }
            }
          >
            {/* Main scrollable content */}
            <div className="grid flex-1 items-center" style={{ gridTemplateColumns }}>
              <div className="flex items-center justify-center px-3 py-2">
                {/* Show checkbox for all items except child documents */}
                {!(doc as Document & { isChildDocument?: boolean }).isChildDocument ? (
                  <Checkbox
                    id={`doc-${doc.external_id}`}
                    checked={selectedDocuments.includes(doc.external_id)}
                    onCheckedChange={checked => handleCheckboxChange(checked, doc.external_id)}
                    onClick={e => e.stopPropagation()}
                    aria-label={`Select ${doc.filename || "document"}`}
                  />
                ) : (
                  <div className="h-4 w-4" /> // Empty space for alignment
                )}
              </div>
              <div
                className={`flex flex-1 items-center gap-2 px-3 py-2 ${(doc as Document & { isChildDocument?: boolean }).isChildDocument ? "pl-8" : ""}`}
              >
                {/* Chevron for folders and "All Documents" or status dot for documents */}
                {(doc as Document & { itemType?: string }).itemType === "folder" ? (
                  <button
                    onClick={e => toggleFolderExpansion(doc.filename || "", e)}
                    className="group relative flex-shrink-0 rounded p-0.5 transition-colors hover:bg-gray-100 dark:hover:bg-gray-800"
                  >
                    {expandedFolders.has(doc.filename || "") ? (
                      <ChevronDown className="h-3 w-3 text-gray-600" />
                    ) : (
                      <ChevronRight className="h-3 w-3 text-gray-600" />
                    )}
                  </button>
                ) : (doc as Document & { itemType?: string }).itemType === "document" ||
                  !(doc as Document & { itemType?: string }).itemType ? (
                  <div className="group relative flex-shrink-0">
                    {doc.system_metadata?.status === "completed" ? (
                      <div className="h-2 w-2 rounded-full bg-green-500" />
                    ) : doc.system_metadata?.status === "failed" ? (
                      <div className="h-2 w-2 rounded-full bg-red-500" />
                    ) : doc.system_metadata?.status === "uploading" ? (
                      <div className="h-2 w-2 animate-spin rounded-full border-2 border-blue-500 border-t-transparent" />
                    ) : (
                      <div className="h-2 w-2 animate-pulse rounded-full bg-amber-500" />
                    )}
                    <div className="absolute -top-8 left-1/2 z-10 hidden -translate-x-1/2 whitespace-nowrap rounded-md border bg-popover px-2 py-1 text-xs text-foreground shadow-md group-hover:block">
                      {doc.system_metadata?.status === "completed"
                        ? "Completed"
                        : doc.system_metadata?.status === "failed"
                          ? "Failed"
                          : doc.system_metadata?.status === "uploading"
                            ? "Uploading"
                            : doc.system_metadata?.status === "processing" && doc.system_metadata?.progress
                              ? `${doc.system_metadata.progress.step_name} (${doc.system_metadata.progress.current_step}/${doc.system_metadata.progress.total_steps})`
                              : "Processing"}
                    </div>
                  </div>
                ) : (
                  <div className="h-2 w-2 flex-shrink-0" /> // Empty space to maintain alignment
                )}

                {/* Icon to show file/folder type */}
                <div className="flex-shrink-0">
                  {(doc as Document & { itemType?: string }).itemType === "folder" ? (
                    <FolderIcon className="h-4 w-4 text-blue-600" />
                  ) : (
                    <FileText className="h-4 w-4 text-gray-600" />
                  )}
                </div>

                <span className="truncate font-medium">{doc.filename || "N/A"}</span>
                {/* Progress bar for processing documents */}
                {doc.system_metadata?.status === "processing" && doc.system_metadata?.progress && (
                  <div className="mt-1 w-full">
                    <div className="h-1 w-full overflow-hidden rounded-full bg-gray-200">
                      <div
                        className="h-full bg-blue-500 transition-all duration-300 ease-out"
                        style={{ width: `${doc.system_metadata.progress.percentage || 0}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>
              <div className="px-3 py-2">
                <button
                  onClick={e => {
                    e.stopPropagation();
                    copyDocumentId(doc.external_id);
                  }}
                  className="group flex items-center gap-2 font-mono text-xs text-muted-foreground transition-colors hover:text-foreground"
                  title="Click to copy Document ID"
                >
                  <span className="max-w-[120px] truncate">{doc.external_id}</span>
                  {copiedDocumentId === doc.external_id ? (
                    <Check className="h-3 w-3 text-green-500" />
                  ) : (
                    <Copy className="h-3 w-3 opacity-0 transition-opacity group-hover:opacity-100" />
                  )}
                </button>
              </div>
              {/* Render metadata values for each column */}
              {allColumns.map(column => (
                <div key={column.name} className="truncate px-3 py-2" title={String(doc.metadata?.[column.name] ?? "")}>
                  {String(doc.metadata?.[column.name] ?? "-")}
                </div>
              ))}
            </div>
            {/* Sticky Actions column */}
            <div
              className={`sticky right-0 z-20 flex w-[120px] items-center justify-end gap-1 border-l border-border px-3 py-2 ${
                doc.external_id === selectedDocument?.external_id ? "bg-accent" : "bg-background"
              } ${(doc as Document & { isChildDocument?: boolean }).isChildDocument ? "bg-gray-50 dark:bg-gray-900" : ""}`}
            >
              {/* Only show actions for actual documents, not folders or special items */}
              {((doc as Document & { itemType?: string }).itemType === "document" ||
                !(doc as Document & { itemType?: string }).itemType) && (
                <>
                  {doc.content_type === "application/pdf" && onViewInPDFViewer && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={e => {
                        e.stopPropagation();
                        onViewInPDFViewer(doc.external_id);
                      }}
                      className="h-8 w-8 p-0"
                      title="View in PDF Viewer"
                    >
                      <Eye className="h-4 w-4" />
                    </Button>
                  )}
                  {onDownloadDocument && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={e => {
                        e.stopPropagation();
                        onDownloadDocument(doc.external_id);
                      }}
                      className="h-8 w-8 p-0"
                      title="Download Document"
                    >
                      <Download className="h-4 w-4" />
                    </Button>
                  )}
                  {onDeleteDocument && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={e => {
                        e.stopPropagation();
                        onDeleteDocument(doc.external_id);
                      }}
                      className="h-8 w-8 p-0 text-destructive hover:text-destructive"
                      title="Delete Document"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  )}
                </>
              )}
            </div>
          </div>
        ))}

        {filteredDocuments.length === 0 && documents.length > 0 && (
          <NoMatchingDocuments
            searchQuery={effectiveSearchQuery}
            hasFilters={false}
            onClearFilters={() => {
              if (hideSearchBar && onSearchChange) {
                onSearchChange("");
              } else {
                setSearchQuery("");
              }
            }}
          />
        )}

        {documents.length === 0 && <EmptyDocuments />}

        {/* Spacer to ensure container fills available height */}
        <div className="min-h-16 flex-1"></div>
      </div>
    </div>
  );
});

export default DocumentList;
