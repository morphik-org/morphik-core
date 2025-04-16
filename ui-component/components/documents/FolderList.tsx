"use client";

import React from 'react';
import { Button } from '@/components/ui/button';
import { PlusCircle, Folder as FolderIcon, File, ArrowLeft } from 'lucide-react';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { cn } from '@/lib/utils';
import { Card, CardContent } from '@/components/ui/card';
import { Folder } from '@/components/types';

interface FolderListProps {
  folders: Folder[];
  selectedFolder: string | null;
  setSelectedFolder: (folderName: string | null) => void;
  apiBaseUrl: string;
  authToken: string | null;
  refreshFolders: () => void;
  loading: boolean;
}

const FolderList: React.FC<FolderListProps> = ({
  folders,
  selectedFolder,
  setSelectedFolder,
  apiBaseUrl,
  authToken,
  refreshFolders,
  loading
}) => {
  const [showNewFolderDialog, setShowNewFolderDialog] = React.useState(false);
  const [newFolderName, setNewFolderName] = React.useState('');
  const [newFolderDescription, setNewFolderDescription] = React.useState('');
  const [isCreatingFolder, setIsCreatingFolder] = React.useState(false);

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
      setSelectedFolder(folderData.name);
      
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
        <div className="flex items-center gap-2 mb-4">
          <Button 
            variant="ghost" 
            size="sm" 
            className="p-1 h-8 w-8" 
            onClick={() => setSelectedFolder(null)}
          >
            <ArrowLeft size={16} />
          </Button>
          <h2 className="font-medium text-lg flex items-center">
            {selectedFolder === "all" ? (
              <>
                <File className="h-5 w-5 mr-2" />
                All Documents
              </>
            ) : (
              <>
                <FolderIcon className="h-5 w-5 mr-2" />
                {selectedFolder}
              </>
            )}
          </h2>
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
      
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
        <Card
          className={cn(
            "cursor-pointer hover:border-primary transition-colors",
            "flex flex-col items-center justify-center h-24"
          )}
          onClick={() => setSelectedFolder("all")}
        >
          <CardContent className="p-4 flex flex-col items-center justify-center">
            <File className="h-10 w-10 mb-1" />
            <span className="text-sm font-medium text-center">All Documents</span>
          </CardContent>
        </Card>
        
        {folders.map((folder) => (
          <Card
            key={folder.name}
            className={cn(
              "cursor-pointer hover:border-primary transition-colors",
              "flex flex-col items-center justify-center h-24"
            )}
            onClick={() => setSelectedFolder(folder.name)}
          >
            <CardContent className="p-4 flex flex-col items-center justify-center">
              <FolderIcon className="h-10 w-10 mb-1" />
              <span className="text-sm font-medium truncate text-center w-full">{folder.name}</span>
            </CardContent>
          </Card>
        ))}
      </div>
      
      {folders.length === 0 && !loading && (
        <div className="text-center p-8 text-sm text-muted-foreground">
          No folders yet. Create one to organize your documents.
        </div>
      )}
      
      {loading && folders.length === 0 && (
        <div className="text-center p-8 text-sm text-muted-foreground">
          Loading folders...
        </div>
      )}
    </div>
  );
};

export default FolderList;