"use client";

import React, { useState } from 'react';
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import { Dialog, DialogClose, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Plus, Wand2, Upload } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

import { Document } from '@/components/types';

type ColumnType = 'string' | 'int' | 'float' | 'bool' | 'Date' | 'json';

interface CustomColumn {
  name: string;
  description: string;
  _type: ColumnType;
  schema?: string;
}

interface MetadataExtractionRule {
  type: "metadata_extraction";
  schema: Record<string, any>;
}

interface DocumentListProps {
  documents: Document[];
  selectedDocument: Document | null;
  selectedDocuments: string[];
  handleDocumentClick: (document: Document) => void;
  handleCheckboxChange: (checked: boolean | "indeterminate", docId: string) => void;
  getSelectAllState: () => boolean | "indeterminate";
  setSelectedDocuments: (docIds: string[]) => void;
  loading: boolean;
  apiBaseUrl: string;
  authToken: string | null;
  selectedFolder?: string | null;
}

// Create a separate Column Dialog component to isolate its state
const AddColumnDialog = ({ 
  isOpen, 
  onClose,
  onAddColumn
}: { 
  isOpen: boolean; 
  onClose: () => void;
  onAddColumn: (column: CustomColumn) => void;
}) => {
  const [localColumnName, setLocalColumnName] = useState('');
  const [localColumnDescription, setLocalColumnDescription] = useState('');
  const [localColumnType, setLocalColumnType] = useState<ColumnType>('string');
  const [localColumnSchema, setLocalColumnSchema] = useState<string>('');

  const handleLocalSchemaFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
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
        _type: localColumnType
      };
      
      if (localColumnType === 'json' && localColumnSchema) {
        column.schema = localColumnSchema;
      }
      
      onAddColumn(column);
      
      // Reset form values
      setLocalColumnName('');
      setLocalColumnDescription('');
      setLocalColumnType('string');
      setLocalColumnSchema('');
      
      // Close the dialog
      onClose();
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent onPointerDownOutside={(e) => e.preventDefault()}>
        <form onSubmit={handleSubmit}>
          <DialogHeader>
            <DialogTitle>Add Custom Column</DialogTitle>
            <DialogDescription>
              Add a new column and specify its type and description.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <label htmlFor="column-name" className="text-sm font-medium">Column Name</label>
              <Input
                id="column-name"
                placeholder="e.g. Author, Category, etc."
                value={localColumnName}
                onChange={(e) => setLocalColumnName(e.target.value)}
                autoFocus
              />
            </div>
            <div className="space-y-2">
              <label htmlFor="column-type" className="text-sm font-medium">Type</label>
              <Select 
                value={localColumnType} 
                onValueChange={(value) => setLocalColumnType(value as ColumnType)}
              >
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
            {localColumnType === 'json' && (
              <div className="space-y-2">
                <label htmlFor="column-schema" className="text-sm font-medium">JSON Schema</label>
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
                    onClick={() => document.getElementById('column-schema-file')?.click()}
                    className="flex items-center gap-2"
                  >
                    <Upload className="h-4 w-4" />
                    Upload Schema
                  </Button>
                  <span className="text-sm text-muted-foreground">
                    {localColumnSchema ? 'Schema loaded' : 'No schema uploaded'}
                  </span>
                </div>
              </div>
            )}
            <div className="space-y-2">
              <label htmlFor="column-description" className="text-sm font-medium">Description</label>
              <Textarea
                id="column-description"
                placeholder="Describe in natural language what information this column should contain..."
                value={localColumnDescription}
                onChange={(e) => setLocalColumnDescription(e.target.value)}
              />
            </div>
          </div>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={onClose}>Cancel</Button>
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
  loading,
  apiBaseUrl,
  authToken,
  selectedFolder
}) => {
  const [customColumns, setCustomColumns] = useState<CustomColumn[]>([]);
  const [showAddColumnDialog, setShowAddColumnDialog] = useState(false);
  const [isExtracting, setIsExtracting] = useState(false);

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
        headers: authToken ? { 'Authorization': `Bearer ${authToken}` } : {}
      });
      
      if (!foldersResponse.ok) {
        throw new Error(`Failed to fetch folders: ${foldersResponse.statusText}`);
      }
      
      const folders = await foldersResponse.json();
      const currentFolder = folders.find((folder: any) => folder.name === selectedFolder);
      
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
              ...(col.schema ? { schema: JSON.parse(col.schema) } : {})
            }
          ])
        )
      };
      
      // Set the rule
      const setRuleResponse = await fetch(`${apiBaseUrl}/folders/${currentFolder.id}/set_rule`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(authToken ? { 'Authorization': `Bearer ${authToken}` } : {})
        },
        body: JSON.stringify({
          rules: [rule]
        })
      });
      
      if (!setRuleResponse.ok) {
        throw new Error(`Failed to set rule: ${setRuleResponse.statusText}`);
      }
      
      const result = await setRuleResponse.json();
      console.log("Rule set successfully:", result);
      
      // Show success message
      alert("Extraction rule set successfully!");
    } catch (error) {
      console.error("Error setting extraction rule:", error);
      alert(`Failed to set extraction rule: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setIsExtracting(false);
    }
  };

  // Create a component for the header to reuse across all return statements
  const DocumentListHeader = () => (
    <div className="bg-muted border-b font-medium sticky top-0 z-10 relative">
      <div className="grid items-center w-full" style={{ 
        gridTemplateColumns: `48px minmax(200px, 350px) 100px 120px 140px ${customColumns.map(() => '140px').join(' ')}` 
      }}>
        <div className="flex items-center justify-center p-3">
          <Checkbox
            id="select-all-documents"
            checked={getSelectAllState()}
            onCheckedChange={(checked) => {
              if (checked) {
                setSelectedDocuments(documents.map(doc => doc.external_id));
              } else {
                setSelectedDocuments([]);
              }
            }}
            aria-label="Select all documents"
          />
        </div>
        <div className="text-sm font-semibold p-3">Filename</div>
        <div className="text-sm font-semibold p-3">Type</div>
        <div className="text-sm font-semibold p-3">
          <div className="group relative inline-flex items-center">
            Status
            <span className="ml-1 text-muted-foreground cursor-help">
              <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="16" x2="12" y2="12"></line>
                <line x1="12" y1="8" x2="12.01" y2="8"></line>
              </svg>
            </span>
            <div className="absolute left-0 top-6 hidden group-hover:block bg-background border text-foreground text-xs p-3 rounded-md w-64 z-[100] shadow-lg">
              Documents with "Processing" status are queryable, but visual features like direct visual context will only be available after processing completes.
            </div>
          </div>
        </div>
        <div className="text-sm font-semibold p-3">ID</div>
        {customColumns.map((column) => (
          <div key={column.name} className="text-sm font-semibold p-3">
            <div className="group relative inline-flex items-center">
              {column.name}
              {/* <span className="ml-1 text-xs text-muted-foreground">({column._type})</span> */}
              <span className="ml-1 text-muted-foreground cursor-help">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10"></circle>
                  <line x1="12" y1="16" x2="12" y2="12"></line>
                  <line x1="12" y1="8" x2="12.01" y2="8"></line>
                </svg>
              </span>
              <div className="absolute left-0 top-6 hidden group-hover:block bg-background border text-foreground text-xs p-3 rounded-md w-64 z-[100] shadow-lg">
                <p>{column.description}</p>
                <p className="mt-1 font-medium">Type: {column._type}</p>
                {column.schema && (
                  <p className="mt-1 text-xs">Schema provided</p>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
      
      <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
        <Button 
          variant="ghost" 
          size="icon" 
          className="h-6 w-6 rounded-full" 
          title="Add column"
          onClick={() => setShowAddColumnDialog(true)}
        >
          <Plus className="h-4 w-4" />
          <span className="sr-only">Add column</span>
        </Button>
        
        {/* Render the dialog separately */}
        <AddColumnDialog 
          isOpen={showAddColumnDialog}
          onClose={() => setShowAddColumnDialog(false)}
          onAddColumn={handleAddColumn}
        />
      </div>
    </div>
  );

  if (loading && !documents.length) {
    return (
      <div className="border rounded-md overflow-hidden shadow-sm w-full">
        <DocumentListHeader />
        <div className="p-8">
          <div className="flex flex-col items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mb-4"></div>
            <p className="text-muted-foreground">Loading documents...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="border rounded-md overflow-hidden shadow-sm w-full">
      <DocumentListHeader />

      <ScrollArea className="h-[calc(100vh-220px)]">
        {documents.map((doc) => (
          <div 
            key={doc.external_id}
            onClick={() => handleDocumentClick(doc)}
            className={`grid items-center w-full border-b ${
              doc.external_id === selectedDocument?.external_id 
                ? 'bg-primary/10 hover:bg-primary/15' 
                : 'hover:bg-muted/70'
            }`}
            style={{ 
              gridTemplateColumns: `48px minmax(200px, 350px) 100px 120px 140px ${customColumns.map(() => '140px').join(' ')}` 
            }}
          >
            <div className="flex items-center justify-center p-3">
              <Checkbox 
                id={`doc-${doc.external_id}`}
                checked={selectedDocuments.includes(doc.external_id)}
                onCheckedChange={(checked) => handleCheckboxChange(checked, doc.external_id)}
                onClick={(e) => e.stopPropagation()}
                aria-label={`Select ${doc.filename || 'document'}`}
              />
            </div>
            <div className="flex items-center p-3">
              <span className="truncate font-medium">{doc.filename || 'N/A'}</span>
            </div>
            <div className="p-3">
              <Badge variant="secondary" className="capitalize text-xs">
                {doc.content_type.split('/')[0]}
              </Badge>
            </div>
            <div className="p-3">
              {doc.system_metadata?.status === "completed" ? (
                <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200 flex items-center gap-1 font-normal text-xs">
                  <span className="h-1.5 w-1.5 rounded-full bg-green-500"></span>
                  Completed
                </Badge>
              ) : doc.system_metadata?.status === "failed" ? (
                <Badge variant="outline" className="bg-red-50 text-red-700 border-red-200 flex items-center gap-1 font-normal text-xs">
                  <span className="h-1.5 w-1.5 rounded-full bg-red-500"></span>
                  Failed
                </Badge>
              ) : (
                <div className="group relative flex items-center">
                  <Badge variant="outline" className="bg-amber-50 text-amber-700 border-amber-200 flex items-center gap-1 font-normal text-xs">
                    <div className="h-1.5 w-1.5 rounded-full bg-amber-500 animate-pulse"></div>
                    Processing
                  </Badge>
                  <div className="absolute left-0 -bottom-14 hidden group-hover:block bg-popover border text-foreground text-xs p-2 rounded-md whitespace-nowrap z-10 shadow-md">
                    Document is being processed. Partial search available.
                  </div>
                </div>
              )}
            </div>
            <div className="font-mono text-xs opacity-80 p-3">
              {doc.external_id.substring(0, 10)}...
            </div>
            {customColumns.map((column) => (
              <div key={column.name} className="p-3">
                {/* Empty cell for custom column */}
              </div>
            ))}
          </div>
        ))}
        
        {documents.length === 0 && (
          <div className="p-12 text-center flex flex-col items-center justify-center">
            <div className="h-12 w-12 rounded-full bg-muted flex items-center justify-center mb-4">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-muted-foreground">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="9" y1="15" x2="15" y2="15"></line>
              </svg>
            </div>
            <p className="text-muted-foreground">
              No documents found in this view.
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Try uploading a document or selecting a different folder.
            </p>
          </div>
        )}
      </ScrollArea>
      
      {customColumns.length > 0 && (
        <div className="border-t p-3 flex justify-end">
          <Button 
            className="gap-2" 
            onClick={handleExtract}
            disabled={isExtracting || !selectedFolder}
          >
            <Wand2 className="h-4 w-4" />
            {isExtracting ? 'Processing...' : 'Extract'}
          </Button>
        </div>
      )}
    </div>
  );
};

export default DocumentList;