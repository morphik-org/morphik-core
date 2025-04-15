"use client";

import React from 'react';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Info } from 'lucide-react';

import { Document } from '@/components/types';

interface DocumentDetailProps {
  selectedDocument: Document | null;
  handleDeleteDocument: (documentId: string) => Promise<void>;
  loading: boolean;
}

const DocumentDetail: React.FC<DocumentDetailProps> = ({
  selectedDocument,
  handleDeleteDocument,
  loading
}) => {
  if (!selectedDocument) {
    return (
      <div className="h-[calc(100vh-200px)] flex items-center justify-center p-8 border border-dashed rounded-lg">
        <div className="text-center text-muted-foreground">
          <Info className="mx-auto h-12 w-12 mb-2" />
          <p>Select a document to view details</p>
        </div>
      </div>
    );
  }

  return (
    <div className="border rounded-lg">
      <div className="bg-muted px-4 py-3 border-b sticky top-0">
        <h3 className="text-lg font-semibold">Document Details</h3>
      </div>
      
      <ScrollArea className="h-[calc(100vh-200px)]">
        <div className="p-4 space-y-4">
          <div>
            <h3 className="font-medium mb-1">Filename</h3>
            <p>{selectedDocument.filename || 'N/A'}</p>
          </div>
          
          <div>
            <h3 className="font-medium mb-1">Content Type</h3>
            <Badge>{selectedDocument.content_type}</Badge>
          </div>
          
          <div>
            <h3 className="font-medium mb-1">Document ID</h3>
            <p className="font-mono text-xs">{selectedDocument.external_id}</p>
          </div>
          
          <Accordion type="single" collapsible>
            <AccordionItem value="metadata">
              <AccordionTrigger>Metadata</AccordionTrigger>
              <AccordionContent>
                <pre className="bg-muted p-2 rounded text-xs overflow-x-auto whitespace-pre-wrap">
                  {JSON.stringify(selectedDocument.metadata, null, 2)}
                </pre>
              </AccordionContent>
            </AccordionItem>
            
            <AccordionItem value="system-metadata">
              <AccordionTrigger>System Metadata</AccordionTrigger>
              <AccordionContent>
                <pre className="bg-muted p-2 rounded text-xs overflow-x-auto whitespace-pre-wrap">
                  {JSON.stringify(selectedDocument.system_metadata, null, 2)}
                </pre>
              </AccordionContent>
            </AccordionItem>
            
            <AccordionItem value="additional-metadata">
              <AccordionTrigger>Additional Metadata</AccordionTrigger>
              <AccordionContent>
                <pre className="bg-muted p-2 rounded text-xs overflow-x-auto whitespace-pre-wrap">
                  {JSON.stringify(selectedDocument.additional_metadata, null, 2)}
                </pre>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
          
          <div className="pt-4 border-t mt-4">
            <Dialog>
              <DialogTrigger asChild>
                <Button variant="outline" size="sm" className="w-full border-red-500 text-red-500 hover:bg-red-100 dark:hover:bg-red-950">
                  Delete Document
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Delete Document</DialogTitle>
                  <DialogDescription>
                    Are you sure you want to delete this document? This action cannot be undone.
                  </DialogDescription>
                </DialogHeader>
                <div className="py-3">
                  <p className="font-medium">Document: {selectedDocument.filename || selectedDocument.external_id}</p>
                  <p className="text-sm text-muted-foreground mt-1">ID: {selectedDocument.external_id}</p>
                </div>
                <DialogFooter>
                  <Button variant="outline" onClick={() => (document.querySelector('[data-state="open"] button[data-state="closed"]') as HTMLElement)?.click()}>Cancel</Button>
                  <Button 
                    variant="outline" 
                    className="border-red-500 text-red-500 hover:bg-red-100 dark:hover:bg-red-950"
                    onClick={() => handleDeleteDocument(selectedDocument.external_id)}
                    disabled={loading}
                  >
                    {loading ? 'Deleting...' : 'Delete'}
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </div>
        </div>
      </ScrollArea>
    </div>
  );
};

export default DocumentDetail;