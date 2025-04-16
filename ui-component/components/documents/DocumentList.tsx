"use client";

import React from 'react';
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';

import { Document } from '@/components/types';

interface DocumentListProps {
  documents: Document[];
  selectedDocument: Document | null;
  selectedDocuments: string[];
  handleDocumentClick: (document: Document) => void;
  handleCheckboxChange: (checked: boolean | "indeterminate", docId: string) => void;
  getSelectAllState: () => boolean | "indeterminate";
  setSelectedDocuments: (docIds: string[]) => void;
  loading: boolean;
}

const DocumentList: React.FC<DocumentListProps> = ({
  documents,
  selectedDocument,
  selectedDocuments,
  handleDocumentClick,
  handleCheckboxChange,
  getSelectAllState,
  setSelectedDocuments,
  loading
}) => {
  if (loading && !documents.length) {
    return <div className="text-center py-8 flex-1">Loading documents...</div>;
  }

  // Status badge helper component (used in the document list items)
  // Status rendering is handled inline in the component instead

  return (
    <div className="border rounded-md">
      <div className="bg-muted border-b p-3 font-medium sticky top-0">
        <div className="grid grid-cols-12">
          <div className="col-span-1 flex items-center justify-center">
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
          <div className="col-span-4">Filename</div>
          <div className="col-span-2">Type</div>
          <div className="col-span-2">
            <div className="group relative inline-flex items-center">
              Status
              <span className="ml-1 text-muted-foreground cursor-help">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10"></circle>
                  <line x1="12" y1="16" x2="12" y2="12"></line>
                  <line x1="12" y1="8" x2="12.01" y2="8"></line>
                </svg>
              </span>
              <div className="absolute left-0 -top-24 hidden group-hover:block bg-gray-800 text-white text-xs p-2 rounded w-60 z-50 shadow-lg">
                Documents with &quot;Processing&quot; status are queryable, but visual features like direct visual context will only be available after processing completes.
              </div>
            </div>
          </div>
          <div className="col-span-3">ID</div>
        </div>
      </div>

      <ScrollArea className="h-[calc(100vh-200px)]">
        {documents.map((doc) => (
          <div 
            key={doc.external_id}
            onClick={() => handleDocumentClick(doc)}
            className={`grid grid-cols-12 p-3 cursor-pointer hover:bg-muted/50 border-b ${
              doc.external_id === selectedDocument?.external_id ? 'bg-muted' : ''
            }`}
          >
            <div className="col-span-1 flex items-center justify-center">
              <Checkbox 
                id={`doc-${doc.external_id}`}
                checked={selectedDocuments.includes(doc.external_id)}
                onCheckedChange={(checked) => handleCheckboxChange(checked, doc.external_id)}
                onClick={(e) => e.stopPropagation()}
                aria-label={`Select ${doc.filename || 'document'}`}
              />
            </div>
            <div className="col-span-4 flex items-center">
              <span className="truncate">{doc.filename || 'N/A'}</span>
            </div>
            <div className="col-span-2">
              <Badge variant="secondary">
                {doc.content_type.split('/')[0]}
              </Badge>
            </div>
            <div className="col-span-2">
              {doc.system_metadata?.status === "completed" ? (
                <Badge variant="outline" className="bg-green-50 text-green-800 border-green-200">
                  Completed
                </Badge>
              ) : doc.system_metadata?.status === "failed" ? (
                <Badge variant="outline" className="bg-red-50 text-red-700 border-red-200">
                  Failed
                </Badge>
              ) : (
                <div className="group relative flex items-center">
                  <Badge variant="outline" className="bg-amber-50 text-amber-700 border-amber-200 px-2 py-1">
                    Processing
                  </Badge>
                  <div className="absolute left-0 -bottom-10 hidden group-hover:block bg-gray-800 text-white text-xs p-2 rounded whitespace-nowrap z-10">
                    Document is being processed. Partial search available.
                  </div>
                </div>
              )}
            </div>
            <div className="col-span-3 font-mono text-xs">
              {doc.external_id.substring(0, 8)}...
            </div>
          </div>
        ))}
        
        {documents.length === 0 && (
          <div className="p-8 text-center text-muted-foreground">
            No documents found in this view. Try uploading a document or selecting a different folder.
          </div>
        )}
      </ScrollArea>
    </div>
  );
};

export default DocumentList;