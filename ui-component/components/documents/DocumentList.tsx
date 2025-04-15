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

  return (
    <div className="border rounded-md">
      <div className="bg-gray-100 border-b p-3 font-medium sticky top-0">
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
          <div className="col-span-3">Type</div>
          <div className="col-span-4">ID</div>
        </div>
      </div>

      <ScrollArea className="h-[calc(100vh-200px)]">
        {documents.map((doc) => (
          <div 
            key={doc.external_id}
            onClick={() => handleDocumentClick(doc)}
            className="grid grid-cols-12 p-3 cursor-pointer hover:bg-gray-50 border-b"
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
              {doc.filename || 'N/A'}
              {doc.external_id === selectedDocument?.external_id && (
                <Badge variant="outline" className="ml-2">Selected</Badge>
              )}
            </div>
            <div className="col-span-3">
              <Badge variant="secondary">
                {doc.content_type.split('/')[0]}
              </Badge>
            </div>
            <div className="col-span-4 font-mono text-xs">
              {doc.external_id.substring(0, 8)}...
            </div>
          </div>
        ))}
      </ScrollArea>
    </div>
  );
};

export default DocumentList;