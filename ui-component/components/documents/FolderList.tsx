"use client";

import React from 'react';
import { Button } from '@/components/ui/button';
import { PlusCircle, ArrowLeft, MessageSquare } from 'lucide-react';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Folder } from '@/components/types';
import { useRouter, usePathname } from 'next/navigation';
import Image from 'next/image';

interface FolderListProps {
  folders: Folder[];
  selectedFolder: string | null;
  setSelectedFolder: (folderName: string | null) => void;
  apiBaseUrl: string;
  authToken: string | null;
  refreshFolders: () => void;
  loading: boolean;
  refreshAction?: () => void;
  selectedDocuments?: string[];
  handleDeleteMultipleDocuments?: () => void;
  showUploadDialog?: boolean;
  setShowUploadDialog?: (show: boolean) => void;
  uploadDialogComponent?: React.ReactNode;
}

const FolderList: React.FC<FolderListProps> = ({
  folders,
  selectedFolder,
  setSelectedFolder,
  apiBaseUrl,
  authToken,
  refreshFolders,
  loading,
  refreshAction,
  selectedDocuments = [],
  handleDeleteMultipleDocuments,
  uploadDialogComponent
}) => {
  const router = useRouter();
  const pathname = usePathname();
  const [showNewFolderDialog, setShowNewFolderDialog] = React.useState(false);
  const [newFolderName, setNewFolderName] = React.useState('');
  const [newFolderDescription, setNewFolderDescription] = React.useState('');
  const [isCreatingFolder, setIsCreatingFolder] = React.useState(false);
  
  // Function to update both state and URL
  const updateSelectedFolder = (folderName: string | null) => {
    setSelectedFolder(folderName);
    
    // Update URL to reflect the selected folder
    if (folderName) {
      router.push(`${pathname}?folder=${encodeURIComponent(folderName)}`);
    } else {
      router.push(pathname);
    }
  };

  const handleCreateFolder = async () => {
    if (!newFolderName.trim()) return;
    
    setIsCreatingFolder(true);
    
    try {
      console.log(`Creating folder: ${newFolderName}`);
      
      const response = await fetch(`${apiBaseUrl}/folders`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(authToken ? { 'Authorization': `Bearer ${authToken}` } : {})
        },
        body: JSON.stringify({
          name: newFolderName.trim(),
          description: newFolderDescription.trim() || undefined
        })
      });
      
      if (!response.ok) {
        throw new Error(`Failed to create folder: ${response.statusText}`);
      }
      
      // Get the created folder data
      const folderData = await response.json();
      console.log(`Created folder with ID: ${folderData.id} and name: ${folderData.name}`);
      
      // Close dialog and reset form
      setShowNewFolderDialog(false);
      setNewFolderName('');
      setNewFolderDescription('');
      
      // Refresh folder list - use a fresh fetch
      await refreshFolders();
      
      // Auto-select this newly created folder so user can immediately add files to it
      // This ensures we start with a clean empty folder view
      updateSelectedFolder(folderData.name);
      
    } catch (error) {
      console.error('Error creating folder:', error);
    } finally {
      setIsCreatingFolder(false);
    }
  };
  
  // If we're viewing a specific folder or all documents, show back button and folder title
  if (selectedFolder !== null) {
    return (
      <div className="mb-4">
        <div className="flex justify-between items-center py-2">
          <div className="flex items-center gap-4">
            <Button 
              variant="ghost" 
              size="icon"
              className="rounded-full hover:bg-muted/50" 
              onClick={() => updateSelectedFolder(null)}
            >
              <ArrowLeft size={18} />
            </Button>
            <div className="flex items-center">
              {selectedFolder === "all" ? (
                <span className="text-3xl mr-3" aria-hidden="true">📄</span>
              ) : (
                <Image src="/icons/folder-icon.png" alt="Folder" width={32} height={32} className="mr-3" />
              )}
              <h2 className="font-medium text-xl">
                {selectedFolder === "all" ? "All Documents" : selectedFolder}
              </h2>
            </div>
            
            {/* Show action buttons if documents are selected */}
            {selectedDocuments && selectedDocuments.length > 0 && (
              <div className="flex gap-2 ml-4">
                {/* Chat with selected button */}
                <Button 
                  variant="outline" 
                  onClick={() => {
                    // Build proper URL path
                    let path = '/';
                    if (typeof window !== 'undefined') {
                      const currentPath = window.location.pathname;
                      if (currentPath !== '/') {
                        path = currentPath;
                      }
                    }
                    
                    // Create filter with external_id which is the correct field name
                    const filter = JSON.stringify({ external_id: selectedDocuments });
                    const filtersParam = encodeURIComponent(filter);
                    
                    // Navigate to chat with selected documents
                    // Use window.location to force a full page reload
                    if (typeof window !== 'undefined') {
                      window.location.href = `${path}?section=chat&filters=${filtersParam}`;
                    } else {
                      router.push(`${path}?section=chat&filters=${filtersParam}`);
                    }
                  }}
                  className="border-primary text-primary hover:bg-primary/10 flex items-center gap-1"
                >
                  <MessageSquare size={16} />
                  Chat with {selectedDocuments.length} selected
                </Button>
                
                {/* Delete button */}
                {handleDeleteMultipleDocuments && (
                  <Button 
                    variant="outline" 
                    onClick={handleDeleteMultipleDocuments}
                    className="border-red-500 text-red-500 hover:bg-red-50"
                  >
                    Delete {selectedDocuments.length} selected
                  </Button>
                )}
              </div>
            )}
          </div>
          
          {/* Action buttons */}
          <div className="flex items-center gap-2">
            {refreshAction && (
              <Button
                variant="outline"
                onClick={refreshAction}
                className="flex items-center"
                title="Refresh documents"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mr-1">
                  <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"></path>
                  <path d="M21 3v5h-5"></path>
                  <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"></path>
                  <path d="M8 16H3v5"></path>
                </svg>
                Refresh
              </Button>
            )}
            
            {/* Upload dialog component */}
            {uploadDialogComponent}
          </div>
        </div>
      </div>
    );
  }
  
  return (
    <div className="mb-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="font-medium text-lg">Folders</h2>
        <Dialog open={showNewFolderDialog} onOpenChange={setShowNewFolderDialog}>
          <DialogTrigger asChild>
            <Button variant="outline" size="sm">
              <PlusCircle className="h-4 w-4 mr-2" /> New Folder
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create New Folder</DialogTitle>
              <DialogDescription>
                Create a new folder to organize your documents.
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div>
                <Label htmlFor="folderName">Folder Name</Label>
                <Input
                  id="folderName"
                  value={newFolderName}
                  onChange={(e) => setNewFolderName(e.target.value)}
                  placeholder="My Folder"
                  className="mt-1"
                />
              </div>
              <div>
                <Label htmlFor="folderDescription">Description (Optional)</Label>
                <Textarea
                  id="folderDescription"
                  value={newFolderDescription}
                  onChange={(e) => setNewFolderDescription(e.target.value)}
                  placeholder="Enter a description for this folder"
                  className="mt-1"
                  rows={3}
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setShowNewFolderDialog(false)}>
                Cancel
              </Button>
              <Button onClick={handleCreateFolder} disabled={isCreatingFolder || !newFolderName.trim()}>
                {isCreatingFolder ? 'Creating...' : 'Create Folder'}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
      
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-6 py-2">
        <div 
          className="cursor-pointer group flex flex-col items-center"
          onClick={() => updateSelectedFolder("all")}
        >
          <div className="mb-2 group-hover:scale-110 transition-transform">
            <span className="text-4xl" aria-hidden="true">📄</span>
          </div>
          <span className="text-sm font-medium text-center group-hover:text-primary transition-colors">All Documents</span>
        </div>
        
        {folders.map((folder) => (
          <div
            key={folder.name}
            className="cursor-pointer group flex flex-col items-center"
            onClick={() => updateSelectedFolder(folder.name)}
          >
            <div className="mb-2 group-hover:scale-110 transition-transform">
              <Image src="/icons/folder-icon.png" alt="Folder" width={64} height={64} />
            </div>
            <span className="text-sm font-medium truncate text-center w-full max-w-[100px] group-hover:text-primary transition-colors">{folder.name}</span>
          </div>
        ))}
      </div>
      
      {folders.length === 0 && !loading && (
        <div className="flex flex-col items-center justify-center p-8 mt-4">
          <Image src="/icons/folder-icon.png" alt="Folder" width={80} height={80} className="opacity-50 mb-3" />
          <p className="text-sm text-muted-foreground">No folders yet. Create one to organize your documents.</p>
        </div>
      )}
      
      {loading && folders.length === 0 && (
        <div className="flex items-center justify-center p-8 mt-4">
          <div className="flex items-center space-x-2">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-primary"></div>
            <p className="text-sm text-muted-foreground">Loading folders...</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default FolderList;