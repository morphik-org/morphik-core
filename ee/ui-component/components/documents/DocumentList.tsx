"use client";

import React, { useState, useEffect } from "react";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
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
import { Plus, Wand2, Upload, Filter } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { showAlert } from "@/components/ui/alert-system";

import { Document, Folder } from "@/components/types";

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

const DocumentList: React.FC<DocumentListProps> = ({
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
}) => {
  const [customColumns, setCustomColumns] = useState<CustomColumn[]>([]);
  const [showAddColumnDialog, setShowAddColumnDialog] = useState(false);
  const [isExtracting, setIsExtracting] = useState(false);
  const [showFilterDialog, setShowFilterDialog] = useState(false);
  const [filterValues, setFilterValues] = useState<Record<string, string>>({});
  const [filteredDocuments, setFilteredDocuments] = useState<Document[]>([]);

  // Get unique metadata fields from all documents
  const existingMetadataFields = React.useMemo(() => {
    const fields = new Set<string>();
    documents.forEach(doc => {
      if (doc.metadata) {
        Object.keys(doc.metadata).forEach(key => fields.add(key));
      }
    });
    return Array.from(fields);
  }, [documents]);

  // Apply filter logic
  useEffect(() => {
    if (Object.keys(filterValues).length === 0) {
      setFilteredDocuments(documents);
      return;
    }

    const filtered = documents.filter(doc => {
      // Check if document matches all filter criteria
      return Object.entries(filterValues).every(([key, value]) => {
        if (!value || value.trim() === "") return true; // Skip empty filters

        const docValue = doc.metadata?.[key];
        if (docValue === undefined) return false;

        // String comparison (case-insensitive)
        return String(docValue).toLowerCase().includes(value.toLowerCase());
      });
    });

    setFilteredDocuments(filtered);
  }, [documents, filterValues]);

  // Combine existing metadata fields with custom columns
  const allColumns = React.useMemo(() => {
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

  const handleAddColumn = (column: CustomColumn) => {
    setCustomColumns([...customColumns, column]);
  };

  // Handle data extraction

  const handleExtract = async () => {
    // First, find the folder object to get its ID
    if (!selectedFolder || customColumns.length === 0) {
      console.error("Cannot extract: No folder selected or no columns defined");
      return;
    }

    // We need to get the folder ID for the API call
    try {
      setIsExtracting(true);

      // First, get folders to find the current folder ID
      const foldersResponse = await fetch(`${apiBaseUrl}/folders`, {
        headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
      });

      if (!foldersResponse.ok) {
        throw new Error(`Failed to fetch folders: ${foldersResponse.statusText}`);
      }

      const folders = await foldersResponse.json();
      const currentFolder = folders.find((folder: Folder) => folder.name === selectedFolder);

      if (!currentFolder) {
        throw new Error(`Folder "${selectedFolder}" not found`);
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
          const folderResponse = await fetch(`${apiBaseUrl}/folders`, {
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
            const targetFolder = freshFolders.find((folder: Folder) => folder.name === selectedFolder);

            if (targetFolder) {
              console.log(`Rule: Found folder ${targetFolder.name} in fresh data`);

              // Get the document IDs from the folder
              const documentIds = Array.isArray(targetFolder.document_ids) ? targetFolder.document_ids : [];
              console.log(`Rule: Folder has ${documentIds.length} documents`);

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
  };

  // Calculate how many filters are currently active
  const activeFilterCount = Object.values(filterValues).filter(v => v && v.trim() !== "").length;

  const DocumentListHeader = () => {
    return (
      <div className="relative sticky top-0 z-10 border-b bg-muted font-medium">
        <div
          className="grid w-full items-center"
          style={{
            gridTemplateColumns: `48px minmax(200px, 350px) 100px 120px ${allColumns.map(() => "140px").join(" ")}`,
          }}
        >
          <div className="flex items-center justify-center p-3">
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
          <div className="p-3 text-sm font-semibold">Filename</div>
          <div className="p-3 text-sm font-semibold">Type</div>
          <div className="p-3 text-sm font-semibold">
            <div className="group relative inline-flex items-center">
              Status
              <span className="ml-1 cursor-help text-muted-foreground">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <circle cx="12" cy="12" r="10"></circle>
                  <line x1="12" y1="16" x2="12" y2="12"></line>
                  <line x1="12" y1="8" x2="12.01" y2="8"></line>
                </svg>
              </span>
              <div className="absolute left-0 top-6 z-[100] hidden w-64 rounded-md border bg-background p-3 text-xs text-foreground shadow-lg group-hover:block">
                Documents with &quot;Processing&quot; status are queryable, but visual features like direct visual
                context will only be available after processing completes.
              </div>
            </div>
          </div>
          {allColumns.map(column => (
            <div key={column.name} className="p-3 text-sm font-semibold">
              <div className="group relative inline-flex items-center">
                {column.name}
                <span className="ml-1 cursor-help text-muted-foreground">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="16" x2="12" y2="12"></line>
                    <line x1="12" y1="8" x2="12.01" y2="8"></line>
                  </svg>
                </span>
                <div className="absolute left-0 top-6 z-[100] hidden w-64 rounded-md border bg-background p-3 text-xs text-foreground shadow-lg group-hover:block">
                  <p>{column.description}</p>
                  <p className="mt-1 font-medium">Type: {column._type}</p>
                  {column.schema && <p className="mt-1 text-xs">Schema provided</p>}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Render dialogs separately */}
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
  };

  if (loading && !documents.length) {
    return (
      <div className="w-full overflow-hidden rounded-md border shadow-sm">
        <DocumentListHeader />
        <div className="p-8">
          <div className="flex flex-col items-center justify-center">
            <div className="mb-4 h-8 w-8 animate-spin rounded-full border-b-2 border-primary"></div>
            <p className="text-muted-foreground">Loading documents...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full overflow-hidden rounded-md border shadow-sm">
      <DocumentListHeader />
      <ScrollArea className="h-[calc(100vh-220px)]">
        {filteredDocuments.map(doc => (
          <div
            key={doc.external_id}
            onClick={() => handleDocumentClick(doc)}
            className={`grid w-full items-center border-b ${
              doc.external_id === selectedDocument?.external_id
                ? "bg-primary/10 hover:bg-primary/15"
                : "hover:bg-muted/70"
            }`}
            style={{
              gridTemplateColumns: `48px minmax(200px, 350px) 100px 120px ${allColumns.map(() => "140px").join(" ")}`,
            }}
          >
            <div className="flex items-center justify-center p-3">
              <Checkbox
                id={`doc-${doc.external_id}`}
                checked={selectedDocuments.includes(doc.external_id)}
                onCheckedChange={checked => handleCheckboxChange(checked, doc.external_id)}
                onClick={e => e.stopPropagation()}
                aria-label={`Select ${doc.filename || "document"}`}
              />
            </div>
            <div className="flex items-center p-3">
              <span className="truncate font-medium">{doc.filename || "N/A"}</span>
            </div>
            <div className="p-3">
              <Badge variant="secondary" className="text-xs capitalize">
                {doc.content_type.split("/")[0]}
              </Badge>
            </div>
            <div className="p-3">
              {doc.system_metadata?.status === "completed" ? (
                <Badge
                  variant="outline"
                  className="flex items-center gap-1 border-green-200 bg-green-50 text-xs font-normal text-green-700"
                >
                  <span className="h-1.5 w-1.5 rounded-full bg-green-500"></span>
                  Completed
                </Badge>
              ) : doc.system_metadata?.status === "failed" ? (
                <Badge
                  variant="outline"
                  className="flex items-center gap-1 border-red-200 bg-red-50 text-xs font-normal text-red-700"
                >
                  <span className="h-1.5 w-1.5 rounded-full bg-red-500"></span>
                  Failed
                </Badge>
              ) : (
                <div className="group relative flex items-center">
                  <Badge
                    variant="outline"
                    className="flex items-center gap-1 border-amber-200 bg-amber-50 text-xs font-normal text-amber-700"
                  >
                    <div className="h-1.5 w-1.5 animate-pulse rounded-full bg-amber-500"></div>
                    Processing
                  </Badge>
                  <div className="absolute -bottom-14 left-0 z-10 hidden whitespace-nowrap rounded-md border bg-popover p-2 text-xs text-foreground shadow-md group-hover:block">
                    Document is being processed. Partial search available.
                  </div>
                </div>
              )}
            </div>
            {/* Render metadata values for each column */}
            {allColumns.map(column => (
              <div key={column.name} className="truncate p-3" title={String(doc.metadata?.[column.name] ?? "")}>
                {String(doc.metadata?.[column.name] ?? "-")}
              </div>
            ))}
          </div>
        ))}

        {filteredDocuments.length === 0 && documents.length > 0 && (
          <div className="flex flex-col items-center justify-center p-12 text-center">
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-muted">
              <Filter className="text-muted-foreground" />
            </div>
            <p className="text-muted-foreground">No documents match the current filters.</p>
            <Button variant="link" className="mt-2" onClick={() => setFilterValues({})}>
              Clear all filters
            </Button>
          </div>
        )}

        {documents.length === 0 && (
          <div className="flex flex-col items-center justify-center p-12 text-center">
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-muted">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="text-muted-foreground"
              >
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="9" y1="15" x2="15" y2="15"></line>
              </svg>
            </div>
            <p className="text-muted-foreground">No documents found in this view.</p>
            <p className="mt-1 text-xs text-muted-foreground">
              Try uploading a document or selecting a different folder.
            </p>
          </div>
        )}
      </ScrollArea>

      <div className="flex justify-between border-t p-3">
        {/* Filter stats */}
        <div className="flex items-center text-sm text-muted-foreground">
          {Object.keys(filterValues).length > 0 ? (
            <div className="flex items-center gap-1">
              <Filter className="h-4 w-4" />
              <span>
                {filteredDocuments.length} of {documents.length} documents
                {Object.keys(filterValues).length > 0 && (
                  <Button variant="link" className="ml-1 h-auto p-0 text-sm" onClick={() => setFilterValues({})}>
                    Clear filters
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
    </div>
  );
};

export default DocumentList;
