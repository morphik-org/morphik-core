"use client";

import React, { useState, useEffect, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Plus, Trash2, Edit2, X, Copy, ExternalLink } from "lucide-react";
import { showAlert } from "@/components/ui/alert-system";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { ModelConfigAPI } from "@/lib/modelConfigApi";
import { CustomModel } from "@/components/types";
import { useTheme } from "next-themes";
import { useMorphik } from "@/contexts/morphik-context";

interface ModelManagerProps {
  apiKeys: Record<string, { apiKey?: string; baseUrl?: string; [key: string]: unknown }>;
  authToken?: string | null;
}

const PROVIDER_INFO = {
  openai: {
    name: "OpenAI",
    logo: {
      light: "/provider-logos/OpenAI-black-monoblossom.png",
      dark: "/provider-logos/OpenAI-white-monoblossom.png",
    },
    exampleConfig: {
      model: "gpt-4-turbo-preview",
      temperature: 0.7,
      max_tokens: 4096,
    },
    docsUrl: "https://docs.litellm.ai/docs/providers/openai",
  },
  anthropic: {
    name: "Anthropic",
    logo: { light: "/provider-logos/Anthropic-black.png", dark: "/provider-logos/Anthropic-white.png" },
    exampleConfig: {
      model: "claude-3-opus-20240229",
      temperature: 0.7,
      max_tokens: 4096,
    },
    docsUrl: "https://docs.litellm.ai/docs/providers/anthropic",
  },
  google: {
    name: "Google",
    logo: { light: "/provider-logos/gemini.svg", dark: "/provider-logos/gemini.svg" },
    exampleConfig: {
      model: "gemini/gemini-1.5-pro-latest",
      temperature: 0.7,
      max_tokens: 4096,
    },
    docsUrl: "https://docs.litellm.ai/docs/providers/vertex",
  },
  groq: {
    name: "Groq",
    logo: { light: "/provider-logos/Groq Logo_Black 25.svg", dark: "/provider-logos/Groq Logo_White 25.svg" },
    exampleConfig: {
      model: "groq/mixtral-8x7b-32768",
      temperature: 0.7,
      max_tokens: 32768,
    },
    docsUrl: "https://docs.litellm.ai/docs/providers/groq",
  },
  deepseek: {
    name: "DeepSeek",
    icon: "🌊",
    exampleConfig: {
      model: "deepseek/deepseek-chat",
      temperature: 0.7,
      max_tokens: 4096,
    },
    docsUrl: "https://docs.litellm.ai/docs/providers",
  },
  ollama: {
    name: "Ollama",
    logo: { light: "/provider-logos/ollama-black.png", dark: "/provider-logos/ollamae-white.png" },
    exampleConfig: {
      model: "ollama/llama2",
      api_base: "http://localhost:11434",
      temperature: 0.7,
    },
    docsUrl: "https://docs.litellm.ai/docs/providers/ollama",
  },
  together: {
    name: "Together AI",
    icon: "🤝",
    exampleConfig: {
      model: "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1",
      temperature: 0.7,
      max_tokens: 4096,
    },
    docsUrl: "https://docs.litellm.ai/docs/providers/togetherai",
  },
  azure: {
    name: "Azure OpenAI",
    icon: "☁️",
    exampleConfig: {
      model: "azure/gpt-4",
      api_base: "https://your-resource.openai.azure.com",
      api_version: "2023-05-15",
      api_key: "your-azure-api-key",
    },
    docsUrl: "https://docs.litellm.ai/docs/providers/azure",
  },
  lemonade: {
    name: "Lemonade",
    icon: "🍋",
    exampleConfig: {
      model: "openai/Qwen2.5-VL-7B-Instruct-GGUF",
      api_base: "http://host.docker.internal:8020/api/v1",
      vision: true,
    },
    docsUrl: "#",
    requiresApiKey: false,
  },
};

export function ModelManager({ apiKeys, authToken }: ModelManagerProps) {
  const [models, setModels] = useState<CustomModel[]>([]);
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [editingModel, setEditingModel] = useState<string | null>(null);
  const [newModel, setNewModel] = useState({
    name: "",
    provider: "",
    config: "",
  });
  const { theme } = useTheme();

  const { apiBaseUrl } = useMorphik();
  const api = useMemo(() => new ModelConfigAPI(authToken || null, apiBaseUrl), [authToken, apiBaseUrl]);

  const renderProviderIcon = (provider: string) => {
    const providerInfo = PROVIDER_INFO[provider as keyof typeof PROVIDER_INFO];
    if (!providerInfo) return <span className="text-xl">🔧</span>;

    if ("logo" in providerInfo && providerInfo.logo) {
      return (
        <img
          src={theme === "dark" ? providerInfo.logo.dark : providerInfo.logo.light}
          alt={`${providerInfo.name} logo`}
          className="h-5 w-5 object-contain"
        />
      );
    } else if ("icon" in providerInfo) {
      return <span className="text-xl">{providerInfo.icon || "🔧"}</span>;
    } else {
      return <span className="text-xl">🔧</span>;
    }
  };

  // Load saved models from backend or localStorage
  useEffect(() => {
    const loadModels = async () => {
      try {
        if (authToken) {
          const response = await fetch(`${apiBaseUrl}/models/custom`, {
            headers: {
              Authorization: `Bearer ${authToken}`,
            },
          });

          if (response.ok) {
            const customModels = await response.json();
            const transformedModels = customModels.map(
              (m: { id: string; name: string; provider: string; config: { model?: string; model_name?: string } }) => ({
                id: m.id,
                name: m.name,
                provider: m.provider,
                model_name: m.config.model || m.config.model_name || "",
                config: m.config,
              })
            );
            setModels(transformedModels);
            // Also update localStorage
            localStorage.setItem("morphik_custom_models", JSON.stringify(transformedModels));
          } else {
            throw new Error("Failed to load models");
          }
        } else {
          // Fall back to localStorage
          const savedModels = localStorage.getItem("morphik_custom_models");
          if (savedModels) {
            try {
              setModels(JSON.parse(savedModels));
            } catch (err) {
              console.error("Failed to parse saved models:", err);
            }
          }
        }
      } catch (err) {
        console.error("Failed to load models:", err);
        // Fall back to localStorage on error
        const savedModels = localStorage.getItem("morphik_custom_models");
        if (savedModels) {
          try {
            setModels(JSON.parse(savedModels));
          } catch (parseErr) {
            console.error("Failed to parse saved models:", parseErr);
          }
        }
      }
    };

    loadModels();
  }, [authToken]);

  // Save models to backend and localStorage
  const saveModels = async (updatedModels: CustomModel[]) => {
    setModels(updatedModels);

    // Always save to localStorage as fallback
    localStorage.setItem("morphik_custom_models", JSON.stringify(updatedModels));

    // Note: Individual model operations (add/delete) will handle backend updates
  };

  const handleAddModel = async () => {
    if (!newModel.name || !newModel.provider || !newModel.config) {
      showAlert("Please fill in all fields", { type: "error" });
      return;
    }

    try {
      const config = JSON.parse(newModel.config);

      // Auto-inject API key if available for the provider and provider requires it
      const providerInfo = PROVIDER_INFO[newModel.provider as keyof typeof PROVIDER_INFO];
      if (providerInfo && "requiresApiKey" in providerInfo && providerInfo.requiresApiKey === false) {
        // Provider doesn't require API key, skip injection
      } else if (apiKeys[newModel.provider]?.apiKey && !config.api_key) {
        config.api_key = apiKeys[newModel.provider].apiKey;
      }

      // Extract model_name from config
      const model_name = config.model || config.model_name || "";

      if (authToken) {
        // Save to backend using the new /models endpoint
        const response = await fetch(`${apiBaseUrl}/models`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${authToken}`,
          },
          body: JSON.stringify({
            name: newModel.name,
            provider: newModel.provider,
            config,
          }),
        });

        if (!response.ok) {
          throw new Error("Failed to save model");
        }

        const createdModel = await response.json();
        const updatedModels = [
          ...models,
          {
            id: createdModel.id,
            name: createdModel.name,
            provider: createdModel.provider,
            model_name: config.model || config.model_name || "",
            config: createdModel.config,
          },
        ];
        setModels(updatedModels);
        // Also save to localStorage for immediate availability in ModelSelector2
        localStorage.setItem("morphik_custom_models", JSON.stringify(updatedModels));
      } else {
        // Save to localStorage only
        const model: CustomModel = {
          id: `custom_${Date.now()}`,
          name: newModel.name,
          provider: newModel.provider,
          model_name,
          config,
        };

        saveModels([...models, model]);
      }

      setNewModel({ name: "", provider: "", config: "" });
      setShowAddDialog(false);
      showAlert("Model added successfully", { type: "success" });
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : "Unknown error";
      if (errorMessage.includes("JSON")) {
        showAlert("Invalid JSON configuration", { type: "error" });
      } else {
        showAlert(`Failed to add model: ${errorMessage}`, { type: "error" });
      }
    }
  };

  const handleDeleteModel = async (id: string) => {
    try {
      if (authToken) {
        // Delete from backend
        const response = await fetch(`${apiBaseUrl}/models/${id}`, {
          method: "DELETE",
          headers: {
            Authorization: `Bearer ${authToken}`,
          },
        });

        if (!response.ok) {
          throw new Error("Failed to delete model");
        }
      }

      // Update local state and localStorage
      const updatedModels = models.filter(m => m.id !== id);
      setModels(updatedModels);
      localStorage.setItem("morphik_custom_models", JSON.stringify(updatedModels));

      showAlert("Model deleted", { type: "success" });
    } catch (err) {
      console.error("Failed to delete model:", err);
      showAlert("Failed to delete model", { type: "error" });
    }
  };

  const handleUpdateModel = async (id: string, updates: Partial<CustomModel>) => {
    try {
      const updatedModels = models.map(m => (m.id === id ? { ...m, ...updates } : m));

      if (authToken) {
        // Update in backend
        const model = updatedModels.find(m => m.id === id);
        if (model) {
          await api.updateCustomModel(id, {
            name: model.name,
            provider: model.provider,
            model_name: model.model_name,
            config: model.config,
          });
        }
      }

      // Update local state and localStorage
      setModels(updatedModels);
      localStorage.setItem("morphik_custom_models", JSON.stringify(updatedModels));

      setEditingModel(null);
      showAlert("Model updated", { type: "success" });
    } catch (err) {
      console.error("Failed to update model:", err);
      showAlert("Failed to update model", { type: "error" });
    }
  };

  const handleProviderChange = (provider: string) => {
    setNewModel({
      ...newModel,
      provider,
      config: JSON.stringify(PROVIDER_INFO[provider as keyof typeof PROVIDER_INFO]?.exampleConfig || {}, null, 2),
    });
  };

  const availableProviders = Object.keys(PROVIDER_INFO).filter(provider => {
    const providerInfo = PROVIDER_INFO[provider as keyof typeof PROVIDER_INFO];
    // Include providers that don't require API keys or have API keys configured
    return (
      (providerInfo && "requiresApiKey" in providerInfo && providerInfo.requiresApiKey === false) ||
      apiKeys[provider]?.apiKey
    );
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold">Custom Models</h2>
          <p className="text-sm text-muted-foreground">
            Add custom LiteLLM-compatible models with your own configurations
          </p>
        </div>
        <Button onClick={() => setShowAddDialog(true)}>
          <Plus className="mr-2 h-4 w-4" />
          Add Model
        </Button>
      </div>

      {models.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <p className="mb-4 text-muted-foreground">No custom models configured</p>
            <Button onClick={() => setShowAddDialog(true)} variant="outline">
              <Plus className="mr-2 h-4 w-4" />
              Add Your First Model
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4">
          {models.map(model => (
            <Card key={model.id}>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {renderProviderIcon(model.provider)}
                    {editingModel === model.id ? (
                      <Input
                        value={model.name}
                        onChange={e => handleUpdateModel(model.id, { name: e.target.value })}
                        className="h-8 w-48"
                      />
                    ) : (
                      <CardTitle className="text-lg">{model.name}</CardTitle>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    {editingModel === model.id ? (
                      <>
                        <Button size="sm" variant="ghost" onClick={() => setEditingModel(null)}>
                          <X className="h-4 w-4" />
                        </Button>
                      </>
                    ) : (
                      <>
                        <Button size="sm" variant="ghost" onClick={() => setEditingModel(model.id)}>
                          <Edit2 className="h-4 w-4" />
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => handleDeleteModel(model.id)}
                          className="text-red-600 hover:text-red-700"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </>
                    )}
                  </div>
                </div>
                <CardDescription>
                  Provider: {PROVIDER_INFO[model.provider as keyof typeof PROVIDER_INFO]?.name || model.provider}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="rounded-lg bg-muted/50 p-3">
                  <div className="mb-2 flex items-center justify-between">
                    <span className="text-xs font-medium text-muted-foreground">Configuration</span>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => {
                        navigator.clipboard.writeText(JSON.stringify(model.config, null, 2));
                        showAlert("Configuration copied", { type: "success" });
                      }}
                    >
                      <Copy className="h-3 w-3" />
                    </Button>
                  </div>
                  <pre className="overflow-x-auto text-xs">{JSON.stringify(model.config, null, 2)}</pre>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Add Model Dialog */}
      <Dialog open={showAddDialog} onOpenChange={setShowAddDialog}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Add Custom Model</DialogTitle>
            <DialogDescription>Configure a custom LiteLLM-compatible model with your API key</DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div>
              <Label htmlFor="model-name">Model Name</Label>
              <Input
                id="model-name"
                placeholder="e.g., GPT-4 Turbo, Mixtral 8x7B"
                value={newModel.name}
                onChange={e => setNewModel({ ...newModel, name: e.target.value })}
              />
            </div>

            <div>
              <Label htmlFor="provider">Provider</Label>
              <Select value={newModel.provider} onValueChange={handleProviderChange}>
                <SelectTrigger id="provider">
                  <SelectValue placeholder="Select a provider" />
                </SelectTrigger>
                <SelectContent>
                  {availableProviders.length === 0 ? (
                    <SelectItem value="none" disabled>
                      No API keys configured - add them in API Keys tab
                    </SelectItem>
                  ) : (
                    availableProviders.map(provider => (
                      <SelectItem key={provider} value={provider}>
                        <div className="flex items-center gap-2">
                          {renderProviderIcon(provider)}
                          <span>{PROVIDER_INFO[provider as keyof typeof PROVIDER_INFO]?.name || provider}</span>
                        </div>
                      </SelectItem>
                    ))
                  )}
                </SelectContent>
              </Select>
              {newModel.provider && (
                <Button
                  variant="link"
                  size="sm"
                  className="mt-1 h-auto p-0"
                  onClick={() =>
                    window.open(PROVIDER_INFO[newModel.provider as keyof typeof PROVIDER_INFO]?.docsUrl, "_blank")
                  }
                >
                  View LiteLLM docs for {newModel.provider}
                  <ExternalLink className="ml-1 h-3 w-3" />
                </Button>
              )}
            </div>

            <div>
              <Label htmlFor="config">Model Configuration (LiteLLM format)</Label>
              <Textarea
                id="config"
                placeholder='{"model": "gpt-4", "temperature": 0.7}'
                value={newModel.config}
                onChange={e => setNewModel({ ...newModel, config: e.target.value })}
                rows={10}
                className="font-mono text-sm"
              />
              <p className="mt-1 text-xs text-muted-foreground">
                Enter a valid JSON configuration. The API key will be automatically added from your saved keys.
              </p>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setShowAddDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleAddModel}>Add Model</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
