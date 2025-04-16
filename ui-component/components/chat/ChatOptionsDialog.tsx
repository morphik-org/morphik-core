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

import { QueryOptions, Folder } from '@/components/types';

interface ChatOptionsDialogProps {
  showChatAdvanced: boolean;
  setShowChatAdvanced: (show: boolean) => void;
  queryOptions: QueryOptions;
  updateQueryOption: <K extends keyof QueryOptions>(key: K, value: QueryOptions[K]) => void;
  availableGraphs: string[];
  folders: Folder[];
}

const ChatOptionsDialog: React.FC<ChatOptionsDialogProps> = ({
  showChatAdvanced,
  setShowChatAdvanced,
  queryOptions,
  updateQueryOption,
  availableGraphs,
  folders
}) => {
  return (
    <Dialog open={showChatAdvanced} onOpenChange={setShowChatAdvanced}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" className="flex items-center">
          <Settings className="mr-2 h-4 w-4" />
          Advanced Options
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Chat Options</DialogTitle>
          <DialogDescription>
            Configure advanced chat parameters
          </DialogDescription>
        </DialogHeader>
        
        <div className="grid gap-4 py-4">
          <div>
            <Label htmlFor="query-filters" className="block mb-2">Filters (JSON)</Label>
            <Textarea 
              id="query-filters" 
              value={queryOptions.filters} 
              onChange={(e) => updateQueryOption('filters', e.target.value)}
              placeholder='{"key": "value"}'
              rows={3}
            />
          </div>
          
          <div>
            <Label htmlFor="query-k" className="block mb-2">
              Number of Results (k): {queryOptions.k}
            </Label>
            <Input 
              id="query-k" 
              type="number" 
              min={1} 
              value={queryOptions.k}
              onChange={(e) => updateQueryOption('k', parseInt(e.target.value) || 1)}
            />
          </div>
          
          <div>
            <Label htmlFor="query-min-score" className="block mb-2">
              Minimum Score: {queryOptions.min_score.toFixed(2)}
            </Label>
            <Input 
              id="query-min-score" 
              type="number" 
              min={0} 
              max={1} 
              step={0.01}
              value={queryOptions.min_score}
              onChange={(e) => updateQueryOption('min_score', parseFloat(e.target.value) || 0)}
            />
          </div>
          
          <div className="flex items-center justify-between">
            <Label htmlFor="query-reranking">Use Reranking</Label>
            <Switch 
              id="query-reranking"
              checked={queryOptions.use_reranking}
              onCheckedChange={(checked) => updateQueryOption('use_reranking', checked)}
            />
          </div>
          
          <div className="flex items-center justify-between">
            <Label htmlFor="query-colpali">Use Colpali</Label>
            <Switch 
              id="query-colpali"
              checked={queryOptions.use_colpali}
              onCheckedChange={(checked) => updateQueryOption('use_colpali', checked)}
            />
          </div>
          
          <div>
            <Label htmlFor="query-max-tokens" className="block mb-2">
              Max Tokens: {queryOptions.max_tokens}
            </Label>
            <Input 
              id="query-max-tokens" 
              type="number" 
              min={1} 
              max={2048}
              value={queryOptions.max_tokens}
              onChange={(e) => updateQueryOption('max_tokens', parseInt(e.target.value) || 1)}
            />
          </div>
          
          <div>
            <Label htmlFor="query-temperature" className="block mb-2">
              Temperature: {queryOptions.temperature.toFixed(2)}
            </Label>
            <Input 
              id="query-temperature" 
              type="number" 
              min={0} 
              max={2} 
              step={0.01}
              value={queryOptions.temperature}
              onChange={(e) => updateQueryOption('temperature', parseFloat(e.target.value) || 0)}
            />
          </div>

          <div>
            <Label htmlFor="graphName" className="block mb-2">Knowledge Graph</Label>
            <Select 
              value={queryOptions.graph_name || "__none__"} 
              onValueChange={(value) => updateQueryOption('graph_name', value === "__none__" ? undefined : value)}
            >
              <SelectTrigger className="w-full" id="graphName">
                <SelectValue placeholder="Select a knowledge graph" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__none__">None (Standard RAG)</SelectItem>
                {availableGraphs.map(graphName => (
                  <SelectItem key={graphName} value={graphName}>
                    {graphName}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-sm text-muted-foreground mt-1">
              Select a knowledge graph to enhance your query with structured relationships
            </p>
          </div>

          <div>
            <Label htmlFor="folderName" className="block mb-2">Scope to Folder</Label>
            <Select 
              value={queryOptions.folder_name || "__none__"} 
              onValueChange={(value) => updateQueryOption('folder_name', value === "__none__" ? undefined : value)}
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
              Limit chat results to documents within a specific folder
            </p>
          </div>
        </div>
        
        <DialogFooter>
          <Button onClick={() => setShowChatAdvanced(false)}>Apply</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default ChatOptionsDialog;