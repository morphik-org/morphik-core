"use client";

import React, { useState, useMemo, useCallback } from "react";
import { Checkbox } from "@/components/ui/checkbox";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
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
  Loader2,
} from "lucide-react";
import { showAlert } from "@/components/ui/alert-system";

import { Document, Folder, FolderSummary, ProcessingProgress } from "../types";
import { EmptyDocuments, NoMatchingDocuments, LoadingDocuments } from "./shared/EmptyStates";

type ColumnType = "string" | "int" | "float" | "bool" | "Date" | "json";

interface DocumentListPaginationConfig {
  skip: number;
  limit: number;
  returnedCount: number;
  totalCount: number | null;
  hasMore: boolean;
  nextSkip: number | null;
  onPageChange: (nextSkip: number) => void;
  onPageSizeChange?: (limit: number) => void;
  pageSizeOptions?: number[];
  loading?: boolean;
}

interface DocumentListProps {
  documents: Document[];
  selectedDocument: Document | null;
  selectedDocuments: string[];
  handleDocumentClick: (document: Document) => void;
  handleCheckboxChange: (checked: boolean | "indeterminate", docId: string) => void;
  setSelectedDocuments: (docIds: string[]) => void;
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
  pagination?: DocumentListPaginationConfig;
}

interface FolderDocumentsState {
  documents: Document[];
  hasMore: boolean;
  nextSkip: number | null;
  loading: boolean;
}

const FOLDER_PAGE_SIZE = 50;

const DocumentList: React.FC<DocumentListProps> = React.memo(function DocumentList({
  documents,
  selectedDocument,
  selectedDocuments,
  handleDocumentClick,
  handleCheckboxChange,
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
  pagination,
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
  const [folderDocuments, setFolderDocuments] = useState<Record<string, FolderDocumentsState>>({});

  const paginationConfig = pagination;

  const paginationSummary = useMemo(() => {
    if (!paginationConfig) {
      return "";
    }

    const { skip, returnedCount, totalCount } = paginationConfig;
    const hasResults = returnedCount > 0;
    const start = hasResults ? skip + 1 : 0;
    const end = hasResults ? skip + returnedCount : 0;
    const boundedEnd = typeof totalCount === "number" ? Math.min(end, totalCount) : end;

    if (!hasResults) {
      if (totalCount === 0) {
        return "No documents to display";
      }
      if (totalCount === null) {
        return "No documents on this page";
      }
    }

    if (typeof totalCount === "number") {
      return `Showing ${start}-${boundedEnd} of ${totalCount}`;
    }

    return hasResults ? `Showing ${start}-${end}` : "No documents to display";
  }, [paginationConfig]);

  const hasNextPage = useMemo(() => {
    if (!paginationConfig) {
      return false;
    }

    if (paginationConfig.hasMore) {
      return true;
    }

    if (typeof paginationConfig.totalCount === "number") {
      return paginationConfig.skip + paginationConfig.returnedCount < paginationConfig.totalCount;
    }

    return false;
  }, [paginationConfig]);

  const disablePrev = !paginationConfig || paginationConfig.loading || paginationConfig.skip <= 0;
  const disableNext = !paginationConfig || paginationConfig.loading || !hasNextPage;

  const nextSkipRaw = paginationConfig
    ? typeof paginationConfig.nextSkip === "number"
      ? paginationConfig.nextSkip
      : paginationConfig.skip + paginationConfig.limit
    : 0;

  const nextSkipClamped =
    paginationConfig && typeof paginationConfig.totalCount === "number"
      ? Math.min(nextSkipRaw, Math.max(0, paginationConfig.totalCount - paginationConfig.limit))
      : nextSkipRaw;

  const pageSizeOptions = useMemo(() => {
    if (!paginationConfig) {
      return [];
    }

    const baseOptions = paginationConfig.pageSizeOptions ?? [25, 50, 100];
    const values = new Set<number>(baseOptions);
    values.add(paginationConfig.limit);

    return Array.from(values).sort((a, b) => a - b);
  }, [paginationConfig]);

  const handlePrevPage = () => {
    if (!paginationConfig || disablePrev) {
      return;
    }

    const prevSkip = Math.max(0, paginationConfig.skip - paginationConfig.limit);
    paginationConfig.onPageChange(prevSkip);
  };

  const handleNextPage = () => {
    if (!paginationConfig || disableNext) {
      return;
    }

    paginationConfig.onPageChange(Math.max(0, nextSkipClamped));
  };

  const handlePageSizeChange = (value: string) => {
    if (!paginationConfig || !paginationConfig.onPageSizeChange) {
      return;
    }

    const parsed = Number(value);
    if (!Number.isNaN(parsed) && parsed > 0) {
      paginationConfig.onPageSizeChange(parsed);
    }
  };

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
      itemType?: "document" | "folder" | "all" | "folder-load-more";
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
          const folderState = folderDocuments[folderName];
          folderState.documents.forEach(childDoc => {
            result.push({
              ...childDoc,
              isChildDocument: true,
              parentFolderName: folderName,
              itemType: "document",
            });
          });
          if (folderState.hasMore) {
            result.push({
              external_id: `folder-load-more-${folderName}`,
              filename: "Load more…",
              content_type: "application/vnd.morphik.load-more",
              metadata: {},
              additional_metadata: {},
              system_metadata: {},
              parentFolderName: folderName,
              itemType: "folder-load-more" as const,
            });
          }
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

  const selectableItems = useMemo(
    () =>
      filteredDocuments.filter(doc => {
        const itemType = (doc as Document & { itemType?: string }).itemType;
        const isChildDocument = (doc as Document & { isChildDocument?: boolean }).isChildDocument;
        if (itemType === "folder-load-more") {
          return false;
        }
        if (isChildDocument) {
          return false;
        }
        return true;
      }),
    [filteredDocuments]
  );

  const visibleSelectableIds = useMemo(() => selectableItems.map(item => item.external_id), [selectableItems]);

  const visibleSelectableIdSet = useMemo(() => new Set(visibleSelectableIds), [visibleSelectableIds]);

  const selectedCountOnPage = useMemo(
    () => visibleSelectableIds.reduce((count, id) => (selectedDocuments.includes(id) ? count + 1 : count), 0),
    [selectedDocuments, visibleSelectableIds]
  );

  const selectAllState: boolean | "indeterminate" = useMemo(() => {
    if (visibleSelectableIds.length === 0) {
      return false;
    }
    if (selectedCountOnPage === 0) {
      return false;
    }
    if (selectedCountOnPage === visibleSelectableIds.length) {
      return true;
    }
    return "indeterminate";
  }, [selectedCountOnPage, visibleSelectableIds]);

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

  const normalizeFolderDocument = useCallback((doc: Partial<Document>): Document => {
    const systemMetadata = {
      ...(doc.system_metadata as Record<string, unknown> | undefined),
    };

    if (!("status" in systemMetadata) || !systemMetadata?.status) {
      (systemMetadata as Record<string, unknown>).status = "processing";
    }

    return {
      external_id: doc.external_id ?? "",
      filename: doc.filename,
      content_type: doc.content_type ?? "application/octet-stream",
      metadata: doc.metadata ? (doc.metadata as Record<string, unknown>) : {},
      additional_metadata: doc.additional_metadata ? (doc.additional_metadata as Record<string, unknown>) : {},
      system_metadata: systemMetadata,
      folder_name: doc.folder_name,
      app_id: doc.app_id,
      end_user_id: doc.end_user_id,
    };
  }, []);

  const fetchFolderDocuments = useCallback(
    async (folderName: string, skip = 0, force = false) => {
      if (!force && folderDocuments[folderName] && skip === 0) {
        return;
      }

      setFolderDocuments(prev => {
        const existing = prev[folderName];
        return {
          ...prev,
          [folderName]: {
            documents: skip > 0 && existing ? existing.documents : (existing?.documents ?? []),
            hasMore: existing?.hasMore ?? false,
            nextSkip: existing?.nextSkip ?? null,
            loading: true,
          },
        };
      });

      try {
        const requestBody = {
          identifiers: [folderName],
          include_document_count: false,
          include_status_counts: false,
          include_documents: true,
          document_skip: skip,
          document_limit: FOLDER_PAGE_SIZE,
          document_fields: [
            "external_id",
            "filename",
            "content_type",
            "metadata",
            "additional_metadata",
            "system_metadata",
            "folder_name",
          ],
          sort_by: "updated_at",
          sort_direction: "desc",
        };

        const response = await fetch(`${apiBaseUrl}/folders/details`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
          },
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch folder documents: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        const entry = Array.isArray(data?.folders) ? data.folders[0] : null;
        const documentInfo = entry?.document_info ?? {};
        const docs = Array.isArray(documentInfo?.documents) ? documentInfo.documents : [];

        const normalizedDocs = docs
          .map((doc: unknown): Document | null => {
            const partial = doc as Partial<Document> | null | undefined;
            if (!partial) {
              return null;
            }
            return normalizeFolderDocument(partial);
          })
          .filter((doc: Document | null): doc is Document => Boolean(doc?.external_id));

        setFolderDocuments(prev => {
          const existing = prev[folderName];
          const baseDocs = skip > 0 && existing ? existing.documents : [];
          const mergedDocs =
            skip > 0
              ? [
                  ...baseDocs,
                  ...normalizedDocs.filter((newDoc: Document) => {
                    return !baseDocs.some(existingDoc => existingDoc.external_id === newDoc.external_id);
                  }),
                ]
              : normalizedDocs;

          const hasMore = Boolean(documentInfo?.has_more);
          const nextSkip =
            typeof documentInfo?.next_skip === "number"
              ? documentInfo.next_skip
              : hasMore
                ? skip + normalizedDocs.length
                : null;

          return {
            ...prev,
            [folderName]: {
              documents: mergedDocs,
              hasMore,
              nextSkip,
              loading: false,
            },
          };
        });
      } catch (error) {
        console.error(`Error fetching documents for folder "${folderName}":`, error);
        setFolderDocuments(prev => ({
          ...prev,
          [folderName]: {
            documents: prev[folderName]?.documents ?? [],
            hasMore: prev[folderName]?.hasMore ?? false,
            nextSkip: prev[folderName]?.nextSkip ?? null,
            loading: false,
          },
        }));
        showAlert(`Failed to load documents for folder "${folderName}"`, {
          type: "error",
          duration: 4000,
        });
      }
    },
    [apiBaseUrl, authToken, folderDocuments, normalizeFolderDocument]
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
        if (!folderDocuments[folderName] || folderDocuments[folderName].documents.length === 0) {
          fetchFolderDocuments(folderName, 0, true);
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
                checked={selectAllState}
                onCheckedChange={checked => {
                  if (checked === true || checked === "indeterminate") {
                    const updated = new Set(selectedDocuments);
                    visibleSelectableIds.forEach(id => updated.add(id));
                    setSelectedDocuments(Array.from(updated));
                  } else {
                    const remaining = selectedDocuments.filter(id => !visibleSelectableIdSet.has(id));
                    setSelectedDocuments(remaining);
                  }
                }}
                aria-label="Select visible rows"
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
        className={`flex h-full w-full flex-col overflow-hidden${
          showBorder ? "rounded-t-md border-l border-r border-t shadow-sm" : ""
        }`}
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
        <div className="flex min-h-0 flex-1 flex-col">
          <div className="min-h-0 flex-1 overflow-auto">
            {DocumentListHeader()}
            <LoadingDocuments />
          </div>
        </div>
      </div>
    );
  }

  const totalItemsCount = documents.length;
  const itemsLabel = totalItemsCount === 1 ? "item" : "items";
  const selectedCount = selectedDocuments.length;

  return (
    <div
      className={`flex h-full w-full flex-col overflow-hidden${
        showBorder ? "rounded-t-md border-l border-r border-t shadow-sm" : ""
      }`}
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
      {mounted && (
        <div className="flex items-center justify-between border-b bg-muted/50 px-4 py-2">
          <span className="text-sm text-muted-foreground">
            {selectedCount} of {totalItemsCount} {itemsLabel} selected
          </span>
          <div className="flex items-center gap-2">
            {selectedDocuments.length > 0 && (
              <>
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
              </>
            )}
          </div>
        </div>
      )}

      <div className="flex min-h-0 flex-1 flex-col">
        {/* Main content area with horizontal scroll */}
        <div className="min-h-0 flex-1 overflow-x-auto overflow-y-auto">
          {/* Header */}
          {DocumentListHeader()}

          {/* Content rows */}
          {filteredDocuments.map(doc => {
            const itemType = (doc as Document & { itemType?: string }).itemType;

            if (itemType === "folder-load-more") {
              const folderName = (doc as Document & { parentFolderName: string }).parentFolderName;
              const folderState = folderDocuments[folderName];
              const isLoading = folderState?.loading ?? false;
              const nextSkip =
                folderState?.nextSkip ??
                (folderState && folderState.documents.length > 0 ? folderState.documents.length : FOLDER_PAGE_SIZE);

              return (
                <div
                  key={doc.external_id}
                  className="flex items-center justify-center border-b border-border bg-muted/40 px-3 py-2 text-sm text-muted-foreground"
                >
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={e => {
                      e.stopPropagation();
                      fetchFolderDocuments(folderName, nextSkip ?? 0, true);
                    }}
                    disabled={isLoading}
                    className="flex items-center gap-2"
                  >
                    {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <ChevronDown className="h-4 w-4" />}
                    {isLoading ? "Loading…" : "Load more documents"}
                  </Button>
                </div>
              );
            }

            return (
              <div
                key={`${doc.external_id}${
                  (doc as Document & { isChildDocument?: boolean; parentFolderName?: string }).isChildDocument
                    ? `-child-${
                        (doc as Document & { isChildDocument?: boolean; parentFolderName?: string }).parentFolderName
                      }`
                    : ""
                }`}
                onClick={() => {
                  // Handle different item types
                  if (itemType === "folder") {
                    // Navigate to folder when clicking on folder row (but not on chevron)
                    handleDocumentClick(doc);
                  } else {
                    // Handle document clicks for actual documents
                    handleDocumentClick(doc);
                  }
                }}
                className={`relative flex min-w-fit border-b border-border ${
                  itemType === "folder"
                    ? "cursor-pointer hover:bg-muted/50"
                    : doc.external_id === selectedDocument?.external_id
                      ? "cursor-pointer bg-primary/10 hover:bg-primary/15"
                      : "cursor-pointer hover:bg-muted/70"
                } ${
                  (doc as Document & { isChildDocument?: boolean }).isChildDocument ? "bg-gray-50 dark:bg-gray-900" : ""
                }`}
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
                    className={`flex flex-1 items-center gap-2 px-3 py-2 ${
                      (doc as Document & { isChildDocument?: boolean }).isChildDocument ? "pl-8" : ""
                    }`}
                  >
                    {/* Chevron for folders and "All Documents" or status dot for documents */}
                    {itemType === "folder" ? (
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
                    ) : itemType === "document" || !itemType ? (
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
                                  ? `${(doc.system_metadata.progress as ProcessingProgress).step_name} (${(doc.system_metadata.progress as ProcessingProgress).current_step}/${(doc.system_metadata.progress as ProcessingProgress).total_steps})`
                                  : "Processing"}
                        </div>
                      </div>
                    ) : (
                      <div className="h-2 w-2 flex-shrink-0" /> // Empty space to maintain alignment
                    )}

                    {/* Icon to show file/folder type */}
                    <div className="flex-shrink-0">
                      {itemType === "folder" ? (
                        <FolderIcon className="h-4 w-4 text-blue-600" />
                      ) : (
                        <FileText className="h-4 w-4 text-gray-600" />
                      )}
                    </div>

                    <span className="truncate font-medium">{doc.filename || "N/A"}</span>
                    {/* Progress bar for processing documents */}
                    {doc.system_metadata?.status === "processing" &&
                      (doc.system_metadata?.progress as ProcessingProgress | undefined) && (
                        <div className="mt-1 w-full">
                          <div className="h-1 w-full overflow-hidden rounded-full bg-gray-200">
                            <div
                              className="h-full bg-blue-500 transition-all duration-300 ease-out"
                              style={{
                                width: `${(doc.system_metadata.progress as ProcessingProgress).percentage || 0}%`,
                              }}
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
                    <div
                      key={column.name}
                      className="truncate px-3 py-2"
                      title={String(doc.metadata?.[column.name] ?? "")}
                    >
                      {String(doc.metadata?.[column.name] ?? "-")}
                    </div>
                  ))}
                </div>
                {/* Sticky Actions column */}
                <div
                  className={`sticky right-0 z-20 flex w-[120px] items-center justify-end gap-1 border-l border-border px-3 py-2 ${
                    doc.external_id === selectedDocument?.external_id ? "bg-accent" : "bg-background"
                  } ${
                    (doc as Document & { isChildDocument?: boolean }).isChildDocument
                      ? "bg-gray-50 dark:bg-gray-900"
                      : ""
                  }`}
                >
                  {/* Only show actions for actual documents, not folders or special items */}
                  {(!itemType || itemType === "document") && (
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
            );
          })}

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
        </div>

        {paginationConfig && (
          <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 text-xs text-muted-foreground">
            <div className="flex flex-wrap items-center gap-3">
              {paginationConfig.onPageSizeChange && pageSizeOptions.length > 0 && (
                <div className="flex items-center gap-2">
                  <span className="font-medium text-foreground/80">Rows per page</span>
                  <Select value={String(paginationConfig.limit)} onValueChange={handlePageSizeChange}>
                    <SelectTrigger className="h-7 w-20 text-xs">
                      <SelectValue placeholder={String(paginationConfig.limit)} />
                    </SelectTrigger>
                    <SelectContent>
                      {pageSizeOptions.map(option => (
                        <SelectItem key={option} value={String(option)}>
                          {option}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}
              <div className="flex items-center gap-2">
                {paginationConfig.loading && <Loader2 className="h-3 w-3 animate-spin" />}
                <span>{paginationSummary || "No documents to display"}</span>
                <span className="text-muted-foreground">(Selections apply to visible rows)</span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" onClick={handlePrevPage} disabled={disablePrev}>
                Previous
              </Button>
              <Button variant="outline" size="sm" onClick={handleNextPage} disabled={disableNext}>
                Next
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
});

export default DocumentList;
