"use client";

import React, { useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Info } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import Image from 'next/image';

import { Document, Folder } from '@/components/types';

interface DocumentDetailProps {
  selectedDocument: Document | null;
  handleDeleteDocument: (documentId: string) => Promise<void>;
  folders: Folder[];
  apiBaseUrl: string;
  authToken: string | null;
  refreshDocuments: () => void;
  refreshFolders: () => void;
  loading: boolean;
  onClose: () => void;
}

const DocumentDetail: React.FC<DocumentDetailProps> = ({
  selectedDocument,
  handleDeleteDocument,
  folders,
  apiBaseUrl,
  authToken,
  refreshDocuments,
  refreshFolders,
  loading,
  onClose
}) => {
  const [isMovingToFolder, setIsMovingToFolder] = useState(false);

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

  const currentFolder = selectedDocument.system_metadata?.folder_name as string | undefined;

  const handleMoveToFolder = async (folderName: string | null) => {
    if (isMovingToFolder || !selectedDocument) return;

    const documentId = selectedDocument.external_id;
    setIsMovingToFolder(true);

    try {
      // First, get the folder ID from the name if a name is provided
      if (folderName) {
        // Find the target folder by name
        const targetFolder = folders.find(folder => folder.name === folderName);
        if (targetFolder && targetFolder.id) {
          console.log(`Found folder with ID: ${targetFolder.id} for name: ${folderName}`);

          // Add to folder using folder ID
          await fetch(`${apiBaseUrl}/folders/${targetFolder.id}/documents/${documentId}`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              ...(authToken ? { 'Authorization': `Bearer ${authToken}` } : {})
            }
          });
        } else {
          console.error(`Could not find folder with name: ${folderName}`);
        }
      }

      // If there's a current folder and we're either moving to a new folder or removing from folder
      if (currentFolder) {
        // Find the current folder ID
        const currentFolderObj = folders.find(folder => folder.name === currentFolder);
        if (currentFolderObj && currentFolderObj.id) {
          // Remove from current folder using folder ID
          await fetch(`${apiBaseUrl}/folders/${currentFolderObj.id}/documents/${documentId}`, {
            method: 'DELETE',
            headers: {
              'Content-Type': 'application/json',
              ...(authToken ? { 'Authorization': `Bearer ${authToken}` } : {})
            }
          });
        }
      }

      // Refresh folders first to get updated document_ids
      await refreshFolders();
      // Then refresh documents with the updated folder information
      await refreshDocuments();
    } catch (error) {
      console.error('Error updating folder:', error);
    } finally {
      setIsMovingToFolder(false);
    }
  };

  return (
    <div className="border rounded-lg">
      <div className="bg-muted px-4 py-3 border-b sticky top-0 flex justify-between items-center">
        <h3 className="text-lg font-semibold">Document Details</h3>
        <Button
          variant="ghost"
          size="icon"
          onClick={onClose}
          className="rounded-full hover:bg-background/80"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
          <span className="sr-only">Close panel</span>
        </Button>
      </div>

      <ScrollArea className="h-[calc(100vh-200px)]">
        <div className="p-4 space-y-4">
          <div>
            <h3 className="font-medium mb-1">Filename</h3>
            <p>{selectedDocument.filename || 'N/A'}</p>
          </div>

          <div>
            <h3 className="font-medium mb-1">Content Type</h3>
            <Badge variant="secondary">{selectedDocument.content_type}</Badge>
          </div>

          <div>
            <h3 className="font-medium mb-1">Folder</h3>
            <div className="flex items-center gap-2">
              <Image src="/icons/folder-icon.png" alt="Folder" width={16} height={16} />
              <Select
                value={currentFolder || "_none"}
                onValueChange={(value) => handleMoveToFolder(value === "_none" ? null : value)}
                disabled={isMovingToFolder}
              >
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Not in a folder" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="_none">Not in a folder</SelectItem>
                  {folders.map((folder) => (
                    <SelectItem key={folder.name} value={folder.name}>
                      {folder.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
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
