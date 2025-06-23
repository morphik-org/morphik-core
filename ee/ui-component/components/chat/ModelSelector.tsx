"use client";

import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import { Sparkles, Settings, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { useModels } from "@/hooks/useModels";

interface Model {
  id: string;
  name: string;
  provider: string;
  description?: string;
}

interface ModelSelectorProps {
  apiBaseUrl: string;
  authToken: string | null;
  selectedModel?: string;
  onModelChange?: (modelId: string) => void;
  onOpenSettings?: () => void;
}

export function ModelSelector({
  apiBaseUrl,
  authToken,
  selectedModel,
  onModelChange,
  onOpenSettings,
}: ModelSelectorProps) {
  const { models, loading } = useModels(apiBaseUrl, authToken);
  const [currentModel, setCurrentModel] = useState<string>(selectedModel || "");

  useEffect(() => {
    // If no model is selected, select the first one
    if (!currentModel && models.length > 0) {
      const defaultModel = models[0].id;
      setCurrentModel(defaultModel);
      onModelChange?.(defaultModel);
    }
  }, [models, currentModel, onModelChange]);

  const selectedModelData = models.find(m => m.id === currentModel);

  const handleModelSelect = (modelId: string) => {
    setCurrentModel(modelId);
    onModelChange?.(modelId);
  };

  // Group models by provider
  const modelsByProvider = models.reduce(
    (acc, model) => {
      if (!acc[model.provider]) {
        acc[model.provider] = [];
      }
      acc[model.provider].push(model);
      return acc;
    },
    {} as Record<string, Model[]>
  );

  const providerIcons: Record<string, string> = {
    openai: "🟢",
    anthropic: "🔶",
    google: "🔵",
    groq: "⚡",
    deepseek: "🌊",
    configured: "⚙️",
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm" className="h-8 px-3 text-xs font-medium" disabled={loading}>
          <Sparkles className="mr-1.5 h-3.5 w-3.5" />
          <span className="max-w-[120px] truncate">
            {loading ? "Loading..." : selectedModelData?.name || "Select Model"}
          </span>
          <ChevronDown className="ml-1.5 h-3 w-3" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-[320px]">
        {Object.entries(modelsByProvider).map(([provider, providerModels], index) => (
          <React.Fragment key={provider}>
            {index > 0 && <DropdownMenuSeparator />}
            <div className="px-2 py-1.5 text-xs font-medium text-muted-foreground">
              {providerIcons[provider] || "●"} {provider.charAt(0).toUpperCase() + provider.slice(1)}
            </div>
            {providerModels.map(model => (
              <DropdownMenuItem
                key={model.id}
                onClick={() => handleModelSelect(model.id)}
                className={cn("flex flex-col items-start py-2", currentModel === model.id && "bg-accent")}
              >
                <div className="text-sm font-medium">{model.name}</div>
                {model.description && <div className="text-xs text-muted-foreground">{model.description}</div>}
              </DropdownMenuItem>
            ))}
          </React.Fragment>
        ))}
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={onOpenSettings} className="py-2">
          <Settings className="mr-2 h-4 w-4" />
          <span>Manage API Keys</span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
