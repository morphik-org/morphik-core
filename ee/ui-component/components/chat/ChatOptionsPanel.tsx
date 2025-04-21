"use client";

import React from 'react';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Settings, ChevronUp, ChevronDown } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

import { QueryOptions } from '@/components/types';

interface ChatOptionsPanelProps {
  showChatAdvanced: boolean;
  setShowChatAdvanced: (show: boolean) => void;
  queryOptions: QueryOptions;
  updateQueryOption: <K extends keyof QueryOptions>(key: K, value: QueryOptions[K]) => void;
  availableGraphs: string[];
}

const ChatOptionsPanel: React.FC<ChatOptionsPanelProps> = ({
  showChatAdvanced,
  setShowChatAdvanced,
  queryOptions,
  updateQueryOption,
  availableGraphs
}) => {
  return (
    <div>
      <button
        type="button"
        className="flex items-center text-sm text-muted-foreground hover:text-foreground"
        onClick={() => setShowChatAdvanced(!showChatAdvanced)}
      >
        <Settings className="mr-1 h-4 w-4" />
        Advanced Options
        {showChatAdvanced ? <ChevronUp className="ml-1 h-4 w-4" /> : <ChevronDown className="ml-1 h-4 w-4" />}
      </button>

      {showChatAdvanced && (
        <div className="mt-3 p-4 border rounded-md bg-muted">
          <div className="space-y-4">
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
              <p className="text-sm text-muted-foreground">
                Select a knowledge graph to enhance your query with structured relationships
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatOptionsPanel;
