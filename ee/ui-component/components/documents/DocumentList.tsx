"use client";

import React, { useState, useMemo, useCallback } from "react";
import { Checkbox } from "@/components/ui/checkbox";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Plus,
  Wand2,
  Upload,
  Filter,
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
  Files,
  ChevronRight,
  ChevronDown,
} from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { showAlert } from "@/components/ui/alert-system";

import { Document, Folder, FolderSummary } from "@/components/types";
import { EmptyDocuments, NoMatchingDocuments, LoadingDocuments } from "./shared/EmptyStates";

type ColumnType = "string" | "int" | "float" | "bool" | "Date" | "json";

interface CustomColumn {
  name: string;
  description: string;
  _type: ColumnType;
  schema?: string;
}

interface MetadataExtractionRule {
  type: "metadata_extraction";
  schema: Record<string, unknown>;
}

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
  folders?: FolderSummary[]; // Optional since it's fetched internally
  showBorder?: boolean; // Control whether to show the outer border and rounded corners
  hideSearchBar?: boolean; // Control whether to hide the search bar
  externalSearchQuery?: string; // External search query when search bar is hidden
  onSearchChange?: (query: string) => void; // Callback for search changes when search bar is hidden
}

// Filter Dialog Component
const FilterDialog = ({
  isOpen,
  onClose,
  columns,
  filterValues,
  setFilterValues,
}: {
  isOpen: boolean;
  onClose: () => void;
  columns: CustomColumn[];
  filterValues: Record<string, string>;
  setFilterValues: React.Dispatch<React.SetStateAction<Record<string, string>>>;
}) => {
  const [localFilters, setLocalFilters] = useState<Record<string, string>>(filterValues);

  const handleApplyFilters = () => {
    setFilterValues(localFilters);
    onClose();
  };

  const handleClearFilters = () => {
    setLocalFilters({});
    setFilterValues({});
    onClose();
  };

  const handleFilterChange = (column: string, value: string) => {
    setLocalFilters(prev => ({
      ...prev,
      [column]: value,
    }));
  };

  return (
    <Dialog open={isOpen} onOpenChange={open => !open && onClose()}>
      <DialogContent onPointerDownOutside={e => e.preventDefault()}>
        <DialogHeader>
          <DialogTitle>Filter Documents</DialogTitle>
          <DialogDescription>Filter documents by their metadata values</DialogDescription>
        </DialogHeader>
        <div className="max-h-96 space-y-4 overflow-y-auto py-4">
          {columns.map(column => (
            <div key={column.name} className="space-y-2">
              <label htmlFor={`filter-${column.name}`} className="text-sm font-medium">
                {column.name}
              </label>
              <Input
                id={`filter-${column.name}`}
                placeholder={`Filter by ${column.name}...`}
                value={localFilters[column.name] || ""}
                onChange={e => handleFilterChange(column.name, e.target.value)}
              />
            </div>
          ))}
        </div>
        <DialogFooter className="flex justify-between">
          <Button variant="outline" onClick={handleClearFilters}>
            Clear Filters
          </Button>
          <div className="flex gap-2">
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button onClick={handleApplyFilters}>Apply Filters</Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

// Create a separate Column Dialog component to isolate its state
const AddColumnDialog = ({
  isOpen,
  onClose,
  onAddColumn,
}: {
  isOpen: boolean;
  onClose: () => void;
  onAddColumn: (column: CustomColumn) => void;
}) => {
  const [localColumnName, setLocalColumnName] = useState("");
  const [localColumnDescription, setLocalColumnDescription] = useState("");
  const [localColumnType, setLocalColumnType] = useState<ColumnType>("string");
  const [localColumnSchema, setLocalColumnSchema] = useState<string>("");

  const handleLocalSchemaFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = event => {
        setLocalColumnSchema(event.target?.result as string);
      };
      reader.readAsText(file);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (localColumnName.trim()) {
      const column: CustomColumn = {
        name: localColumnName.trim(),
        description: localColumnDescription.trim(),
        _type: localColumnType,
      };

      if (localColumnType === "json" && localColumnSchema) {
        column.schema = localColumnSchema;
      }

      onAddColumn(column);

      // Reset form values
      setLocalColumnName("");
      setLocalColumnDescription("");
      setLocalColumnType("string");
      setLocalColumnSchema("");

      // Close the dialog
      onClose();
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={open => !open && onClose()}>
      <DialogContent onPointerDownOutside={e => e.preventDefault()}>
        <form onSubmit={handleSubmit}>
          <DialogHeader>
            <DialogTitle>Add Custom Column</DialogTitle>
            <DialogDescription>Add a new column and specify its type and description.</DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <label htmlFor="column-name" className="text-sm font-medium">
                Column Name
              </label>
              <Input
                id="column-name"
                placeholder="e.g. Author, Category, etc."
                value={localColumnName}
                onChange={e => setLocalColumnName(e.target.value)}
                autoFocus
              />
            </div>
            <div className="space-y-2">
              <label htmlFor="column-type" className="text-sm font-medium">
                Type
              </label>
              <Select value={localColumnType} onValueChange={value => setLocalColumnType(value as ColumnType)}>
                <SelectTrigger id="column-type">
                  <SelectValue placeholder="Select data type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="string">String</SelectItem>
                  <SelectItem value="int">Integer</SelectItem>
                  <SelectItem value="float">Float</SelectItem>
                  <SelectItem value="bool">Boolean</SelectItem>
                  <SelectItem value="Date">Date</SelectItem>
                  <SelectItem value="json">JSON</SelectItem>
                </SelectContent>
              </Select>
            </div>
            {localColumnType === "json" && (
              <div className="space-y-2">
                <label htmlFor="column-schema" className="text-sm font-medium">
                  JSON Schema
                </label>
                <div className="flex items-center space-x-2">
                  <Input
                    id="column-schema-file"
                    type="file"
                    accept=".json"
                    className="hidden"
                    onChange={handleLocalSchemaFileChange}
                  />
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => document.getElementById("column-schema-file")?.click()}
                    className="flex items-center gap-2"
                  >
                    <Upload className="h-4 w-4" />
                    Upload Schema
                  </Button>
                  <span className="text-sm text-muted-foreground">
                    {localColumnSchema ? "Schema loaded" : "No schema uploaded"}
                  </span>
                </div>
              </div>
            )}
            <div className="space-y-2">
              <label htmlFor="column-description" className="text-sm font-medium">
                Description
              </label>
              <Textarea
                id="column-description"
                placeholder="Describe in natural language what information this column should contain..."
                value={localColumnDescription}
                onChange={e => setLocalColumnDescription(e.target.value)}
              />
            </div>
          </div>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit">Add Column</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};

const DocumentList: React.FC<DocumentListProps> = React.memo(function DocumentList({
  documents,
  selectedDocument,
  selectedDocuments,
  handleDocumentClick,
  handleCheckboxChange,
  getSelectAllState,
  setSelectedDocuments,
  setDocuments,
  loading,
  apiBaseUrl,
  authToken,
  selectedFolder,
  onViewInPDFViewer,
  onDownloadDocument,
  onDeleteDocument,
  showBorder = true,
  hideSearchBar = false,
  externalSearchQuery = "",
  onSearchChange,
}) {
  const [customColumns, setCustomColumns] = useState<CustomColumn[]>([]);
  const [showAddColumnDialog, setShowAddColumnDialog] = useState(false);
  const [isExtracting, setIsExtracting] = useState(false);
  const [showFilterDialog, setShowFilterDialog] = useState(false);
  const [filterValues, setFilterValues] = useState<Record<string, string>>({});
  const [copiedDocumentId, setCopiedDocumentId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");

  // Use external search query when search bar is hidden
  const effectiveSearchQuery = hideSearchBar ? externalSearchQuery : searchQuery;
  const [sortColumn, setSortColumn] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");

  // State for expanded folders and their documents
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set());
  const [folderDocuments, setFolderDocuments] = useState<Record<string, Document[]>>({});
  const [isAllDocumentsExpanded, setIsAllDocumentsExpanded] = useState(false);

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

  // Apply filter, search, and sort logic with memoization, including expanded folder documents
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

    // Apply column filters
    if (Object.keys(filterValues).length > 0) {
      result = result.filter(doc => {
        // Check if document matches all filter criteria
        return Object.entries(filterValues).every(([key, value]) => {
          if (!value || value.trim() === "") return true; // Skip empty filters

          const docValue = doc.metadata?.[key];
          if (docValue === undefined) return false;

          // String comparison (case-insensitive)
          return String(docValue).toLowerCase().includes(value.toLowerCase());
        });
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
  }, [documents, filterValues, effectiveSearchQuery, sortColumn, sortDirection, expandedFolders, folderDocuments]);

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

  // Handle "All Documents" expansion toggle
  const toggleAllDocumentsExpansion = useCallback(
    (event: React.MouseEvent) => {
      event.stopPropagation();

      setIsAllDocumentsExpanded(prev => {
        const newExpanded = !prev;

        if (newExpanded) {
          // When expanding "All Documents", expand all folders
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
          // When collapsing "All Documents", collapse all folders
          setExpandedFolders(new Set());
        }

        return newExpanded;
      });
    },
    [documents, folderDocuments, fetchFolderDocuments]
  );

  // Combine existing metadata fields with custom columns
  const allColumns = useMemo(() => {
    const metadataColumns: CustomColumn[] = existingMetadataFields.map(field => ({
      name: field,
      description: `Extracted ${field}`,
      _type: "string", // Default to string type for existing metadata
    }));

    // Merge with custom columns, preferring custom column definitions if they exist
    const mergedColumns = [...metadataColumns];
    customColumns.forEach(customCol => {
      const existingIndex = mergedColumns.findIndex(col => col.name === customCol.name);
      if (existingIndex >= 0) {
        mergedColumns[existingIndex] = customCol;
      } else {
        mergedColumns.push(customCol);
      }
    });

    return mergedColumns;
  }, [existingMetadataFields, customColumns]);

  const handleAddColumn = useCallback((column: CustomColumn) => {
    setCustomColumns(prev => [...prev, column]);
  }, []);

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

  // Handle data extraction
  const handleExtract = useCallback(async () => {
    // First, find the folder object to get its ID
    if (!selectedFolder || customColumns.length === 0) {
      console.error("Cannot extract: No folder selected or no columns defined");
      return;
    }

    // We need to get the folder ID for the API call
    try {
      setIsExtracting(true);

      // First, get folders to find the current folder ID
      const foldersResponse = await fetch(`${apiBaseUrl}/folders/summary`, {
        headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
      });

      if (!foldersResponse.ok) {
        throw new Error(`Failed to fetch folders: ${foldersResponse.statusText}`);
      }

      const folders = await foldersResponse.json();
      const currentFolder = folders.find((folder: FolderSummary) => folder.name === selectedFolder);

      if (!currentFolder) {
        throw new Error(`Folder "${selectedFolder}" not found`);
      }

      // Ensure we have document_ids – fetch folder detail if missing
      let docIds: string[] = Array.isArray(currentFolder.document_ids) ? currentFolder.document_ids : [];
      if (docIds.length === 0) {
        const detailRes = await fetch(`${apiBaseUrl}/folders/${currentFolder.id}`, {
          headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
        });
        if (detailRes.ok) {
          const detail: Folder = await detailRes.json();
          docIds = Array.isArray(detail.document_ids) ? detail.document_ids : [];
        }
      }

      // Convert columns to metadata extraction rule
      const rule: MetadataExtractionRule = {
        type: "metadata_extraction",
        schema: Object.fromEntries(
          customColumns.map(col => [
            col.name,
            {
              type: col._type,
              description: col.description,
              ...(col.schema ? { schema: JSON.parse(col.schema) } : {}),
            },
          ])
        ),
      };

      // Set the rule
      const setRuleResponse = await fetch(`${apiBaseUrl}/folders/${currentFolder.id}/set_rule`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
        body: JSON.stringify({
          rules: [rule],
        }),
      });

      if (!setRuleResponse.ok) {
        throw new Error(`Failed to set rule: ${setRuleResponse.statusText}`);
      }

      const result = await setRuleResponse.json();
      console.log("Rule set successfully:", result);

      // Show success message
      showAlert("Extraction rule set successfully!", {
        type: "success",
        duration: 3000,
      });

      // Force a fresh refresh after setting the rule
      // This is a special function to ensure we get truly fresh data
      const refreshAfterRule = async () => {
        try {
          console.log("Performing fresh refresh after setting extraction rule");
          // Clear folder data to force a clean refresh
          const folderResponse = await fetch(`${apiBaseUrl}/folders/summary`, {
            method: "GET",
            headers: {
              ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
            },
          });

          if (!folderResponse.ok) {
            throw new Error(`Failed to fetch folders: ${folderResponse.statusText}`);
          }

          const freshFolders = await folderResponse.json();
          console.log(`Rule: Fetched ${freshFolders.length} folders with fresh data`);

          // Now fetch documents based on the current folder
          if (selectedFolder && selectedFolder !== "all") {
            // Find the folder by name
            const targetFolder = freshFolders.find((folder: FolderSummary) => folder.name === selectedFolder);

            if (targetFolder) {
              console.log(`Rule: Found folder ${targetFolder.name} in fresh data`);

              // Ensure we have document IDs (may be missing in summary response)
              let documentIds = Array.isArray(targetFolder.document_ids) ? targetFolder.document_ids : [];
              if (documentIds.length === 0 && targetFolder.id) {
                const detResp = await fetch(`${apiBaseUrl}/folders/${targetFolder.id}`, {
                  headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
                });
                if (detResp.ok) {
                  const det: Folder = await detResp.json();
                  documentIds = Array.isArray(det.document_ids) ? det.document_ids : [];
                }
              }

              if (documentIds.length > 0) {
                // Fetch document details for the IDs
                const docResponse = await fetch(`${apiBaseUrl}/batch/documents`, {
                  method: "POST",
                  headers: {
                    "Content-Type": "application/json",
                    ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
                  },
                  body: JSON.stringify({
                    document_ids: [...documentIds],
                  }),
                });

                if (!docResponse.ok) {
                  throw new Error(`Failed to fetch documents: ${docResponse.statusText}`);
                }

                const freshDocs = await docResponse.json();
                console.log(`Rule: Fetched ${freshDocs.length} document details`);

                // Update documents state
                setDocuments(freshDocs);
              } else {
                // Empty folder
                setDocuments([]);
              }
            } else {
              console.log(`Rule: Selected folder ${selectedFolder} not found in fresh data`);
              setDocuments([]);
            }
          } else {
            // For "all" documents view, fetch all documents
            const allDocsResponse = await fetch(`${apiBaseUrl}/documents`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
              },
              body: JSON.stringify({}),
            });

            if (!allDocsResponse.ok) {
              throw new Error(`Failed to fetch all documents: ${allDocsResponse.statusText}`);
            }

            const allDocs = await allDocsResponse.json();
            console.log(`Rule: Fetched ${allDocs.length} documents for "all" view`);
            setDocuments(allDocs);
          }
        } catch (err) {
          console.error("Error refreshing after setting rule:", err);
          showAlert("Error refreshing data after setting rule", {
            type: "error",
            duration: 3000,
          });
        }
      };

      // Execute the refresh
      await refreshAfterRule();
    } catch (error) {
      console.error("Error setting extraction rule:", error);
      showAlert(`Failed to set extraction rule: ${error instanceof Error ? error.message : String(error)}`, {
        type: "error",
        title: "Error",
        duration: 5000,
      });
    } finally {
      setIsExtracting(false);
    }
  }, [selectedFolder, customColumns, apiBaseUrl, authToken, setDocuments]);

  // Calculate how many filters are currently active
  const activeFilterCount = useMemo(
    () => Object.values(filterValues).filter(v => v && v.trim() !== "").length,
    [filterValues]
  );

  // Base grid template for the scrollable part – exclude the Actions column.
  const gridTemplateColumns = useMemo(
    () => `48px minmax(200px, 350px) 160px ${allColumns.map(() => "140px").join(" ")}`,
    [allColumns]
  );

  const DocumentListHeader = () => {
    return (
      <div className="relative sticky top-0 z-20 border-b bg-muted font-medium">
        <div className="flex w-max min-w-full">
          {/* Main scrollable content */}
          <div className="grid flex-1 items-center" style={{ gridTemplateColumns }}>
            <div className="flex items-center justify-center px-3 py-2">
              <Checkbox
                id="select-all-documents"
                checked={getSelectAllState()}
                onCheckedChange={checked => {
                  if (checked) {
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
      <div className={`w-full overflow-hidden ${showBorder ? "rounded-md border shadow-sm" : ""}`}>
        {/* Search Bar */}
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
        <div className="h-[calc(100vh-280px)] overflow-auto">
          {DocumentListHeader()}
          <LoadingDocuments />
        </div>
      </div>
    );
  }

  return (
    <div className={`w-full overflow-hidden ${showBorder ? "rounded-md border shadow-sm" : ""}`}>
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

      {/* Main content area with horizontal scroll */}
      <div className="h-[calc(100vh-280px)] overflow-auto">
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
              } else if ((doc as Document & { itemType?: string }).itemType !== "all") {
                // Handle document clicks for actual documents
                handleDocumentClick(doc);
              } else {
                // Handle "All Documents" click
                handleDocumentClick(doc);
              }
            }}
            className={`relative flex w-max min-w-full border-b border-border ${
              (doc as Document & { itemType?: string }).itemType === "folder"
                ? "cursor-pointer hover:bg-muted/50"
                : doc.external_id === selectedDocument?.external_id
                  ? "bg-primary/10 hover:bg-primary/15"
                  : "hover:bg-muted/70"
            } ${(doc as Document & { isChildDocument?: boolean }).isChildDocument ? "bg-gray-50" : ""}`}
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
                className={`flex items-center gap-2 px-3 py-2 ${(doc as Document & { isChildDocument?: boolean }).isChildDocument ? "pl-8" : ""}`}
              >
                {/* Chevron for folders and "All Documents" or status dot for documents */}
                {(doc as Document & { itemType?: string }).itemType === "folder" ? (
                  <button
                    onClick={e => toggleFolderExpansion(doc.filename || "", e)}
                    className="group relative flex-shrink-0 rounded p-0.5 transition-colors hover:bg-gray-100"
                  >
                    {expandedFolders.has(doc.filename || "") ? (
                      <ChevronDown className="h-3 w-3 text-gray-600" />
                    ) : (
                      <ChevronRight className="h-3 w-3 text-gray-600" />
                    )}
                  </button>
                ) : (doc as Document & { itemType?: string }).itemType === "all" ? (
                  <button
                    onClick={toggleAllDocumentsExpansion}
                    className="group relative flex-shrink-0 rounded p-0.5 transition-colors hover:bg-gray-100"
                  >
                    {isAllDocumentsExpanded ? (
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
                  ) : (doc as Document & { itemType?: string }).itemType === "all" ? (
                    <Files className="h-4 w-4 text-purple-600" />
                  ) : (
                    <FileText className="h-4 w-4 text-gray-600" />
                  )}
                </div>

                <span className="truncate font-medium">{doc.filename || "N/A"}</span>
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
              } ${(doc as Document & { isChildDocument?: boolean }).isChildDocument ? "bg-gray-50" : ""}`}
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
            hasFilters={Object.keys(filterValues).length > 0}
            onClearFilters={() => {
              setFilterValues({});
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

      <div className="flex justify-between border-t p-3">
        {/* Filter stats */}
        <div className="flex items-center text-sm text-muted-foreground">
          {Object.keys(filterValues).length > 0 || effectiveSearchQuery.trim() ? (
            <div className="flex items-center gap-1">
              {effectiveSearchQuery.trim() && <Search className="h-4 w-4" />}
              {Object.keys(filterValues).length > 0 && <Filter className="h-4 w-4" />}
              <span>
                {filteredDocuments.length} of {documents.length} documents
                {(Object.keys(filterValues).length > 0 || effectiveSearchQuery.trim()) && (
                  <Button
                    variant="link"
                    className="ml-1 h-auto p-0 text-sm"
                    onClick={() => {
                      setFilterValues({});
                      if (hideSearchBar && onSearchChange) {
                        onSearchChange("");
                      } else {
                        setSearchQuery("");
                      }
                    }}
                  >
                    Clear all
                  </Button>
                )}
              </span>
            </div>
          ) : null}
        </div>

        {/* Action buttons */}
        <div className="flex gap-2">
          {/* Filter button */}
          <Button
            variant={activeFilterCount > 0 ? "default" : "outline"}
            size="sm"
            className="h-8 text-xs font-medium"
            onClick={() => setShowFilterDialog(true)}
          >
            <Filter className="mr-0.5 h-3.5 w-3.5" />
            Filter
            {activeFilterCount > 0 && (
              <span className="ml-1 flex h-4 w-4 items-center justify-center rounded-full bg-primary/20 text-[10px] text-primary">
                {activeFilterCount}
              </span>
            )}
          </Button>

          {/* Add column button */}
          <Button
            variant="outline"
            size="sm"
            className="h-8 text-xs font-medium"
            title="Add column"
            onClick={() => setShowAddColumnDialog(true)}
          >
            <Plus className="mr-0.5 h-3.5 w-3.5" />
            Column
          </Button>

          {customColumns.length > 0 && selectedFolder && (
            <Button className="gap-2" onClick={handleExtract} disabled={isExtracting || !selectedFolder}>
              <Wand2 className="h-4 w-4" />
              {isExtracting ? "Processing..." : "Extract"}
            </Button>
          )}
        </div>
      </div>

      {/* Render dialogs */}
      <AddColumnDialog
        isOpen={showAddColumnDialog}
        onClose={() => setShowAddColumnDialog(false)}
        onAddColumn={handleAddColumn}
      />

      <FilterDialog
        isOpen={showFilterDialog}
        onClose={() => setShowFilterDialog(false)}
        columns={allColumns}
        filterValues={filterValues}
        setFilterValues={setFilterValues}
      />
    </div>
  );
});

export default DocumentList;
