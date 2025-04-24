"use client";

import React from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Settings } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

import { SearchOptions, Folder } from '@/components/types';

interface SearchOptionsDialogProps {
  showSearchAdvanced: boolean;
  setShowSearchAdvanced: (show: boolean) => void;
  searchOptions: SearchOptions;
  updateSearchOption: <K extends keyof SearchOptions>(key: K, value: SearchOptions[K]) => void;
  folders: Folder[];
}

const SearchOptionsDialog: React.FC<SearchOptionsDialogProps> = ({
  showSearchAdvanced,
  setShowSearchAdvanced,
  searchOptions,
  updateSearchOption,
  folders
}) => {
  return (
    <Dialog open={showSearchAdvanced} onOpenChange={setShowSearchAdvanced}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" className="flex items-center">
          <Settings className="mr-2 h-4 w-4" />
          Advanced Options
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Search Options</DialogTitle>
          <DialogDescription>
            Configure advanced search parameters
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-4 py-4">
          <div>
            <Label htmlFor="search-filters" className="block mb-2">Filters (JSON)</Label>
            <Textarea
              id="search-filters"
              value={searchOptions.filters}
              onChange={(e) => updateSearchOption('filters', e.target.value)}
              placeholder='{"key": "value"}'
              rows={3}
            />
          </div>

          <div>
            <Label htmlFor="search-k" className="block mb-2">
              Number of Results (k): {searchOptions.k}
            </Label>
            <Input
              id="search-k"
              type="number"
              min={1}
              value={searchOptions.k}
              onChange={(e) => updateSearchOption('k', parseInt(e.target.value) || 1)}
            />
          </div>

          <div>
            <Label htmlFor="search-min-score" className="block mb-2">
              Minimum Score: {searchOptions.min_score.toFixed(2)}
            </Label>
            <Input
              id="search-min-score"
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={searchOptions.min_score}
              onChange={(e) => updateSearchOption('min_score', parseFloat(e.target.value) || 0)}
            />
          </div>

          <div className="flex items-center justify-between">
            <Label htmlFor="search-reranking">Use Reranking</Label>
            <Switch
              id="search-reranking"
              checked={searchOptions.use_reranking}
              onCheckedChange={(checked) => updateSearchOption('use_reranking', checked)}
            />
          </div>

          <div className="flex items-center justify-between">
            <Label htmlFor="search-colpali">Use Colpali</Label>
            <Switch
              id="search-colpali"
              checked={searchOptions.use_colpali}
              onCheckedChange={(checked) => updateSearchOption('use_colpali', checked)}
            />
          </div>

          <div>
            <Label htmlFor="folderName" className="block mb-2">Scope to Folder</Label>
            <Select
              value={searchOptions.folder_name || "__none__"}
              onValueChange={(value) => updateSearchOption('folder_name', value === "__none__" ? undefined : value)}
            >
              <SelectTrigger className="w-full" id="folderName">
                <SelectValue placeholder="Select a folder" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__none__">All Folders</SelectItem>
                {folders.map(folder => (
                  <SelectItem key={folder.name} value={folder.name}>
                    {folder.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-sm text-muted-foreground mt-1">
              Limit search results to documents within a specific folder
            </p>
          </div>
        </div>

        <DialogFooter>
          <Button onClick={() => setShowSearchAdvanced(false)}>Apply</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default SearchOptionsDialog;
