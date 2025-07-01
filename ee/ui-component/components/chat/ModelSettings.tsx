"use client";

import React, { useState, useEffect, useMemo } from "react";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Eye, EyeOff, Save, Trash2, ExternalLink } from "lucide-react";
import { showAlert } from "@/components/ui/alert-system";
import { ModelConfigAPI } from "@/lib/modelConfigApi";
import { ModelConfigResponse } from "@/components/types";

interface ModelSettingsProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  authToken?: string | null;
}

interface APIKeyConfig {
  [provider: string]: {
    apiKey?: string;
    baseUrl?: string;
    [key: string]: unknown;
  };
}

const PROVIDERS = [
  {
    id: "openai",
    name: "OpenAI",
    icon: "🟢",
    fields: [
      { key: "apiKey", label: "API Key", type: "password", required: true },
      { key: "baseUrl", label: "Base URL (Optional)", type: "text", placeholder: "https://api.openai.com/v1" },
    ],
    docsUrl: "https://platform.openai.com/api-keys",
  },
  {
    id: "anthropic",
    name: "Anthropic",
    icon: "🔶",
    fields: [
      { key: "apiKey", label: "API Key", type: "password", required: true },
      { key: "baseUrl", label: "Base URL (Optional)", type: "text", placeholder: "https://api.anthropic.com" },
    ],
    docsUrl: "https://console.anthropic.com/settings/keys",
  },
  {
    id: "google",
    name: "Google Gemini",
    icon: "🔵",
    fields: [{ key: "apiKey", label: "API Key", type: "password", required: true }],
    docsUrl: "https://makersuite.google.com/app/apikey",
  },
  {
    id: "groq",
    name: "Groq",
    icon: "⚡",
    fields: [{ key: "apiKey", label: "API Key", type: "password", required: true }],
    docsUrl: "https://console.groq.com/keys",
  },
  {
    id: "deepseek",
    name: "DeepSeek",
    icon: "🌊",
    fields: [{ key: "apiKey", label: "API Key", type: "password", required: true }],
    docsUrl: "https://platform.deepseek.com/api_keys",
  },
];

export function ModelSettings({ open, onOpenChange, authToken }: ModelSettingsProps) {
  const [config, setConfig] = useState<APIKeyConfig>({});
  const [showKeys, setShowKeys] = useState<Record<string, boolean>>({});
  const [saving, setSaving] = useState(false);
  const [activeTab, setActiveTab] = useState("openai");
  const [backendConfigs, setBackendConfigs] = useState<ModelConfigResponse[]>([]);

  const api = useMemo(() => new ModelConfigAPI(authToken || null), [authToken]);

  // Load saved configuration from backend or localStorage
  useEffect(() => {
    const loadConfig = async () => {
      try {
        if (authToken) {
          // Try to load from backend first
          const configs = await api.listConfigs();
          setBackendConfigs(configs);

          // Convert backend configs to APIKeyConfig format
          const configMap: APIKeyConfig = {};
          for (const backendConfig of configs) {
            configMap[backendConfig.provider] = backendConfig.config_data;
          }

          // If no backend configs, try to sync from localStorage
          if (configs.length === 0) {
            await api.syncFromLocalStorage();
            const updatedConfigs = await api.listConfigs();
            setBackendConfigs(updatedConfigs);

            for (const backendConfig of updatedConfigs) {
              configMap[backendConfig.provider] = backendConfig.config_data;
            }
          }

          setConfig(configMap);
        } else {
          // No auth token, fall back to localStorage
          const savedConfig = localStorage.getItem("morphik_api_keys");
          if (savedConfig) {
            try {
              setConfig(JSON.parse(savedConfig));
            } catch (err) {
              console.error("Failed to parse saved API keys:", err);
            }
          }
        }
      } catch (err) {
        console.error("Failed to load configurations:", err);
        // Fall back to localStorage on error
        const savedConfig = localStorage.getItem("morphik_api_keys");
        if (savedConfig) {
          try {
            setConfig(JSON.parse(savedConfig));
          } catch (parseErr) {
            console.error("Failed to parse saved API keys:", parseErr);
          }
        }
      }
    };

    if (open) {
      loadConfig();
    }
  }, [open, authToken, api]);

  const handleSave = async () => {
    setSaving(true);
    try {
      if (authToken) {
        // Save to backend
        for (const [provider, providerConfig] of Object.entries(config)) {
          const existingConfig = backendConfigs.find(c => c.provider === provider);

          if (existingConfig) {
            // Update existing config
            await api.updateConfig(existingConfig.id, { config_data: providerConfig });
          } else if (providerConfig.apiKey) {
            // Create new config only if API key is provided
            await api.createConfig({
              provider,
              config_data: providerConfig,
            });
          }
        }

        // Delete configs that were removed
        for (const backendConfig of backendConfigs) {
          if (!config[backendConfig.provider]) {
            await api.deleteConfig(backendConfig.id);
          }
        }
      }

      // Always save to localStorage as fallback
      localStorage.setItem("morphik_api_keys", JSON.stringify(config));

      showAlert("API keys saved successfully", {
        type: "success",
        duration: 3000,
      });
      onOpenChange(false);
    } catch (err) {
      console.error("Failed to save API keys:", err);
      showAlert("Failed to save API keys", {
        type: "error",
        duration: 5000,
      });
    } finally {
      setSaving(false);
    }
  };

  const handleFieldChange = (provider: string, field: string, value: string) => {
    setConfig(prev => ({
      ...prev,
      [provider]: {
        ...prev[provider],
        [field]: value,
      },
    }));
  };

  const handleClearProvider = (provider: string) => {
    setConfig(prev => {
      const newConfig = { ...prev };
      delete newConfig[provider];
      return newConfig;
    });
  };

  const toggleShowKey = (providerId: string) => {
    setShowKeys(prev => ({
      ...prev,
      [providerId]: !prev[providerId],
    }));
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[80vh] max-w-2xl flex-col overflow-hidden">
        <DialogHeader>
          <DialogTitle>Model Settings</DialogTitle>
          <DialogDescription>
            Configure API keys for different AI providers. Your keys are stored securely in your browser.
          </DialogDescription>
        </DialogHeader>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="flex flex-1 flex-col overflow-hidden">
          <TabsList className="grid w-full grid-cols-5">
            {PROVIDERS.map(provider => (
              <TabsTrigger key={provider.id} value={provider.id} className="flex items-center gap-1 text-xs">
                <span>{provider.icon}</span>
                <span className="hidden sm:inline">{provider.name}</span>
              </TabsTrigger>
            ))}
          </TabsList>

          <div className="flex-1 overflow-y-auto">
            {PROVIDERS.map(provider => (
              <TabsContent key={provider.id} value={provider.id} className="mt-4 space-y-4">
                <div className="rounded-lg border p-4">
                  <div className="mb-4 flex items-center justify-between">
                    <h3 className="flex items-center gap-2 text-lg font-semibold">
                      <span>{provider.icon}</span>
                      {provider.name}
                    </h3>
                    <Button variant="outline" size="sm" onClick={() => window.open(provider.docsUrl, "_blank")}>
                      <ExternalLink className="mr-1 h-3 w-3" />
                      Get API Key
                    </Button>
                  </div>

                  <div className="space-y-4">
                    {provider.fields.map(field => (
                      <div key={field.key}>
                        <Label htmlFor={`${provider.id}-${field.key}`}>
                          {field.label}
                          {field.required && <span className="ml-1 text-red-500">*</span>}
                        </Label>
                        <div className="relative mt-1">
                          <Input
                            id={`${provider.id}-${field.key}`}
                            type={
                              field.type === "password" && !showKeys[`${provider.id}-${field.key}`]
                                ? "password"
                                : "text"
                            }
                            placeholder={field.placeholder}
                            value={(config[provider.id]?.[field.key] as string) || ""}
                            onChange={e => handleFieldChange(provider.id, field.key, e.target.value)}
                            className="pr-10"
                          />
                          {field.type === "password" && (
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              className="absolute right-1 top-1/2 h-7 w-7 -translate-y-1/2 p-0"
                              onClick={() => toggleShowKey(`${provider.id}-${field.key}`)}
                            >
                              {showKeys[`${provider.id}-${field.key}`] ? (
                                <EyeOff className="h-3.5 w-3.5" />
                              ) : (
                                <Eye className="h-3.5 w-3.5" />
                              )}
                            </Button>
                          )}
                        </div>
                      </div>
                    ))}

                    {config[provider.id] && Object.keys(config[provider.id]).length > 0 && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleClearProvider(provider.id)}
                        className="text-red-600 hover:text-red-700"
                      >
                        <Trash2 className="mr-1 h-3.5 w-3.5" />
                        Clear {provider.name} Configuration
                      </Button>
                    )}
                  </div>
                </div>
              </TabsContent>
            ))}
          </div>
        </Tabs>

        <div className="flex justify-end gap-2 border-t pt-4">
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={saving}>
            <Save className="mr-1 h-4 w-4" />
            {saving ? "Saving..." : "Save Changes"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
