"use client";

import React, { useState, useEffect, useRef } from "react";
import { ChevronDown, Lock } from "lucide-react";
import { cn } from "@/lib/utils";
import { ModelConfigAPI } from "@/lib/modelConfigApi";

interface Model {
  id: string;
  name: string;
  provider: string;
  description?: string;
}

interface ModelSelector2Props {
  apiBaseUrl: string;
  authToken: string | null;
  selectedModel?: string;
  onModelChange?: (modelId: string) => void;
  onRequestApiKey?: (provider: string) => void;
}

export function ModelSelector2({
  apiBaseUrl,
  authToken,
  selectedModel,
  onModelChange,
  onRequestApiKey,
}: ModelSelector2Props) {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [currentModel, setCurrentModel] = useState<string>(selectedModel || "");
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Get saved API keys to determine which models are available
  const [availableProviders, setAvailableProviders] = useState<Set<string>>(new Set());
  const [customModels, setCustomModels] = useState<Model[]>([]);

  useEffect(() => {
    const loadAvailableProviders = async () => {
      const providers = new Set<string>();
      const api = new ModelConfigAPI(authToken);

      try {
        // Get config from backend and localStorage
        const mergedConfig = await api.getMergedConfig();

        // Check which providers have API keys
        if (mergedConfig.openai?.apiKey) providers.add("openai");
        if (mergedConfig.anthropic?.apiKey) providers.add("anthropic");
        if (mergedConfig.google?.apiKey) providers.add("google");
        if (mergedConfig.groq?.apiKey) providers.add("groq");
        if (mergedConfig.deepseek?.apiKey) providers.add("deepseek");

        // Load custom models
        const customModelsList = await api.listCustomModels();
        const transformedCustomModels = customModelsList.map(
          (model: { id: string; name: string; provider: string }) => ({
            id: `custom_${model.id}`,
            name: model.name,
            provider: model.provider,
            description: `Custom ${model.provider} model`,
          })
        );
        setCustomModels(transformedCustomModels);

        // Add custom model providers
        customModelsList.forEach((model: { provider: string; config: { api_key?: string } }) => {
          if (mergedConfig[model.provider]?.apiKey || model.config.api_key) {
            providers.add(model.provider);
          }
        });
      } catch (err) {
        console.error("Failed to load configurations:", err);

        // Fallback to localStorage
        const savedConfig = localStorage.getItem("morphik_api_keys");
        if (savedConfig) {
          try {
            const config = JSON.parse(savedConfig);

            if (config.openai?.apiKey) providers.add("openai");
            if (config.anthropic?.apiKey) providers.add("anthropic");
            if (config.google?.apiKey) providers.add("google");
            if (config.groq?.apiKey) providers.add("groq");
            if (config.deepseek?.apiKey) providers.add("deepseek");
          } catch (parseErr) {
            console.error("Failed to parse API keys:", parseErr);
          }
        }

        // Load custom models from localStorage
        const savedModels = localStorage.getItem("morphik_custom_models");
        if (savedModels) {
          try {
            const parsedModels = JSON.parse(savedModels);
            const transformedCustomModels = parsedModels.map(
              (model: { id: string; name: string; provider: string }) => ({
                id: `custom_${model.id}`,
                name: model.name,
                provider: model.provider,
                description: `Custom ${model.provider} model`,
              })
            );
            setCustomModels(transformedCustomModels);

            // Add custom model providers to available providers
            const config = JSON.parse(localStorage.getItem("morphik_api_keys") || "{}");
            parsedModels.forEach((model: { provider: string; config: { api_key?: string } }) => {
              if (config[model.provider]?.apiKey || model.config.api_key) {
                providers.add(model.provider);
              }
            });
          } catch (parseErr) {
            console.error("Failed to parse custom models:", parseErr);
          }
        }
      }

      // Check for OpenAI key in environment (from .env file)
      if (process.env.OPENAI_API_KEY || process.env.NEXT_PUBLIC_OPENAI_API_KEY) {
        providers.add("openai");
      }

      // Always add configured provider since it uses server-side keys
      providers.add("configured");

      setAvailableProviders(providers);
    };

    if (isOpen) {
      loadAvailableProviders();
    }
  }, [isOpen, authToken]); // Re-check when dropdown opens

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch(`${apiBaseUrl}/models`, {
          headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
        });
        if (response.ok) {
          const data = await response.json();

          // Transform the chat_models to our expected format
          const transformedModels = (data.chat_models || []).map(
            (model: { config: { model_name?: string }; model: string; id: string; provider: string }) => ({
              id: model.config.model_name || model.model,
              name: model.id.replace(/_/g, " ").replace(/\b\w/g, (l: string) => l.toUpperCase()),
              provider: model.provider,
              description: `Model: ${model.model}`,
            })
          );

          // Combine server models with custom models
          const allModels = [...transformedModels, ...customModels];
          setModels(allModels);

          // If no model is selected, try to select the first available one
          const allModelsForSelection = [...transformedModels, ...customModels];
          if (!currentModel && allModelsForSelection.length > 0) {
            const firstAvailable = allModelsForSelection.find(
              (m: Model) => availableProviders.has(m.provider) || m.provider === "configured"
            );
            if (firstAvailable) {
              const modelId = firstAvailable.id;
              setCurrentModel(modelId);
              onModelChange?.(modelId);
            }
          }
        }
      } catch (err) {
        console.error("Failed to fetch models:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchModels();
  }, [apiBaseUrl, authToken, availableProviders, customModels, currentModel, onModelChange]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const selectedModelData = models.find(m => m.id === currentModel);
  const isModelAvailable = (model: Model) => {
    // Custom models are available if they have an API key in their config or in saved API keys
    if (model.id.startsWith("custom_")) {
      return availableProviders.has(model.provider);
    }
    return availableProviders.has(model.provider) || model.provider === "configured";
  };

  const handleModelSelect = (model: Model) => {
    if (isModelAvailable(model)) {
      setCurrentModel(model.id);
      onModelChange?.(model.id);
      setIsOpen(false);
    }
  };

  const providerIcons: Record<string, string> = {
    openai: "🟢",
    anthropic: "🔶",
    google: "🔵",
    groq: "⚡",
    deepseek: "🌊",
    configured: "⚙️",
    ollama: "🦙",
    together: "🤝",
    azure: "☁️",
  };

  return (
    <div className="relative" ref={dropdownRef}>
      {/* Current Model Display */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
        disabled={loading}
      >
        <span>{loading ? "Loading..." : selectedModelData?.name || "Select model"}</span>
        <ChevronDown className={cn("h-3 w-3 transition-transform", isOpen && "rotate-180")} />
      </button>

      {/* Dropdown */}
      {isOpen && !loading && (
        <div className="absolute bottom-full left-0 mb-2 w-72 rounded-lg border bg-popover p-1 shadow-lg">
          <div className="max-h-80 overflow-y-auto">
            {models.map(model => {
              const isAvailable = isModelAvailable(model);

              return (
                <div
                  key={model.id}
                  className={cn(
                    "group relative flex items-start gap-2 rounded-md px-2 py-2 text-sm",
                    isAvailable ? "cursor-pointer hover:bg-accent" : "cursor-not-allowed opacity-50",
                    currentModel === model.id && "bg-accent"
                  )}
                  onClick={() => handleModelSelect(model)}
                  onMouseEnter={e => {
                    if (!isAvailable) {
                      const tooltip = e.currentTarget.querySelector(".tooltip") as HTMLElement;
                      if (tooltip) tooltip.style.display = "block";
                    }
                  }}
                  onMouseLeave={e => {
                    if (!isAvailable) {
                      const tooltip = e.currentTarget.querySelector(".tooltip") as HTMLElement;
                      if (tooltip) tooltip.style.display = "none";
                    }
                  }}
                >
                  <span className="text-base">{providerIcons[model.provider] || "●"}</span>
                  <div className="flex-1">
                    <div className="flex items-center gap-1.5">
                      <span className="font-medium">{model.name}</span>
                      {!isAvailable && <Lock className="h-3 w-3" />}
                    </div>
                    {model.description && <div className="text-xs text-muted-foreground">{model.description}</div>}
                  </div>

                  {/* Hover tooltip for locked models */}
                  {!isAvailable && (
                    <div
                      className="tooltip absolute bottom-full left-1/2 mb-1 hidden -translate-x-1/2 whitespace-nowrap rounded bg-popover-foreground px-2 py-1 text-xs text-popover"
                      style={{ display: "none" }}
                      onClick={e => {
                        e.stopPropagation();
                        onRequestApiKey?.(model.provider);
                        setIsOpen(false);
                      }}
                    >
                      Add API key →
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
