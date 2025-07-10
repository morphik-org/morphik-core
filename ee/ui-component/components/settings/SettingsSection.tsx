"use client";

import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Eye, EyeOff, Save, Trash2, ExternalLink, Key, ChevronLeft, Bot } from "lucide-react";
import { showAlert } from "@/components/ui/alert-system";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ModelManager } from "./ModelManager";
import { useHeader } from "@/contexts/header-context";

interface SettingsSectionProps {
  authToken?: string | null;
  onBackClick?: () => void;
  initialTab?: string;
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
    description: "GPT-4, GPT-3.5, and other OpenAI models",
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
    description: "Claude 3.5 Sonnet, Haiku, and other Anthropic models",
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
    description: "Gemini Pro and Flash models",
    fields: [{ key: "apiKey", label: "API Key", type: "password", required: true }],
    docsUrl: "https://makersuite.google.com/app/apikey",
  },
  {
    id: "groq",
    name: "Groq",
    icon: "⚡",
    description: "Fast inference for Llama and other models",
    fields: [{ key: "apiKey", label: "API Key", type: "password", required: true }],
    docsUrl: "https://console.groq.com/keys",
  },
  {
    id: "deepseek",
    name: "DeepSeek",
    icon: "🌊",
    description: "DeepSeek Chat and Coder models",
    fields: [{ key: "apiKey", label: "API Key", type: "password", required: true }],
    docsUrl: "https://platform.deepseek.com/api_keys",
  },
];

export function SettingsSection({ authToken, onBackClick, initialTab = "api-keys" }: SettingsSectionProps) {
  const [activeTab, setActiveTab] = useState(initialTab);
  const [config, setConfig] = useState<APIKeyConfig>({});
  const [showKeys, setShowKeys] = useState<Record<string, boolean>>({});
  const [saving, setSaving] = useState(false);
  const { setCustomBreadcrumbs } = useHeader();

  // Load saved configuration from localStorage and backend
  useEffect(() => {
    const loadConfig = async () => {
      // First load from localStorage (only in browser)
      let savedConfig: string | null = null;
      if (typeof window !== "undefined") {
        savedConfig = localStorage.getItem("morphik_api_keys");
        if (savedConfig) {
          try {
            setConfig(JSON.parse(savedConfig));
          } catch (err) {
            console.error("Failed to parse saved API keys:", err);
          }
        }
      }

      // Then try to load from backend if we have authToken
      if (authToken) {
        try {
          const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || "https://api.morphik.ai"}/api-keys`, {
            headers: {
              Authorization: `Bearer ${authToken}`,
            },
          });

          if (response.ok) {
            const apiKeys = await response.json();
            // Merge with local config, backend takes precedence
            const mergedConfig: APIKeyConfig = {};

            for (const [provider, providerData] of Object.entries(apiKeys)) {
              const data = providerData as { configured?: boolean; baseUrl?: string };
              if (data.configured) {
                mergedConfig[provider] = {
                  apiKey: (savedConfig && JSON.parse(savedConfig)[provider]?.apiKey) || "",
                  baseUrl: data.baseUrl,
                };
              }
            }

            // Merge with any local-only keys
            if (savedConfig) {
              const localConfig = JSON.parse(savedConfig);
              for (const [provider, data] of Object.entries(localConfig)) {
                if (!mergedConfig[provider]) {
                  mergedConfig[provider] = data as { apiKey?: string; baseUrl?: string; [key: string]: unknown };
                }
              }
            }

            setConfig(mergedConfig);
          }
        } catch (err) {
          console.error("Failed to load API keys from backend:", err);
        }
      }
    };

    loadConfig();
  }, [authToken]);

  useEffect(() => {
    setCustomBreadcrumbs([{ label: "Home", href: "/" }, { label: "Settings" }]);
    return () => setCustomBreadcrumbs(null);
  }, [setCustomBreadcrumbs]);

  const handleSave = async () => {
    setSaving(true);
    try {
      // Save to localStorage first (only in browser)
      if (typeof window !== "undefined") {
        localStorage.setItem("morphik_api_keys", JSON.stringify(config));
      }

      // If we have authToken, also save to backend
      if (authToken) {
        const savePromises = [];

        for (const [provider, providerConfig] of Object.entries(config)) {
          if (providerConfig.apiKey) {
            savePromises.push(
              fetch(`${process.env.NEXT_PUBLIC_API_URL || "https://api.morphik.ai"}/api-keys`, {
                method: "POST",
                headers: {
                  "Content-Type": "application/json",
                  Authorization: `Bearer ${authToken}`,
                },
                body: JSON.stringify({
                  provider,
                  api_key: providerConfig.apiKey,
                  base_url: providerConfig.baseUrl,
                }),
              })
            );
          }
        }

        const results = await Promise.allSettled(savePromises);
        const failed = results.filter(r => r.status === "rejected");

        if (failed.length > 0) {
          console.error("Some API keys failed to save:", failed);
          showAlert(`Saved locally. ${failed.length} provider(s) failed to sync to cloud.`, {
            type: "warning",
            duration: 5000,
          });
        } else {
          showAlert("API keys saved successfully", {
            type: "success",
            duration: 3000,
          });
        }
      } else {
        showAlert("API keys saved locally", {
          type: "success",
          duration: 3000,
        });
      }
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

  const toggleShowKey = (providerId: string, fieldKey: string) => {
    const key = `${providerId}-${fieldKey}`;
    setShowKeys(prev => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  const hasUnsavedChanges = () => {
    if (typeof window === "undefined") return false;
    const savedConfig = localStorage.getItem("morphik_api_keys");
    if (!savedConfig) return Object.keys(config).length > 0;
    try {
      return JSON.stringify(config) !== savedConfig;
    } catch {
      return true;
    }
  };

  return (
    <div className="flex h-full">
      {/* Sidebar */}
      <div className="w-64 border-r bg-muted/10">
        <div className="p-4">
          {onBackClick && (
            <Button variant="ghost" size="sm" onClick={onBackClick} className="mb-4 w-full justify-start">
              <ChevronLeft className="mr-2 h-4 w-4" />
              Back
            </Button>
          )}

          <nav className="mt-4 space-y-1">
            <button
              onClick={() => setActiveTab("api-keys")}
              className={`flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                activeTab === "api-keys" ? "bg-accent text-accent-foreground" : "hover:bg-accent/50"
              }`}
            >
              <Key className="h-4 w-4" />
              API Keys
            </button>
            <button
              onClick={() => setActiveTab("models")}
              className={`flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                activeTab === "models" ? "bg-accent text-accent-foreground" : "hover:bg-accent/50"
              }`}
            >
              <Bot className="h-4 w-4" />
              Custom Models
            </button>
          </nav>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1">
        <ScrollArea className="h-full">
          <div className="p-6">
            {activeTab === "api-keys" && (
              <>
                <div className="mb-6">
                  <p className="text-sm text-muted-foreground">
                    Configure API keys for different AI providers. Your keys are stored securely in your browser.
                  </p>
                </div>

                <div className="grid gap-6">
                  {PROVIDERS.map(provider => (
                    <Card key={provider.id}>
                      <CardHeader>
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <span className="text-2xl">{provider.icon}</span>
                            <div>
                              <CardTitle>{provider.name}</CardTitle>
                              <CardDescription>{provider.description}</CardDescription>
                            </div>
                          </div>
                          <Button variant="outline" size="sm" onClick={() => window.open(provider.docsUrl, "_blank")}>
                            <ExternalLink className="mr-1 h-3 w-3" />
                            Get API Key
                          </Button>
                        </div>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        {provider.fields.map(field => {
                          const fieldKey = `${provider.id}-${field.key}`;
                          return (
                            <div key={field.key}>
                              <Label htmlFor={fieldKey}>
                                {field.label}
                                {field.required && <span className="ml-1 text-red-500">*</span>}
                              </Label>
                              <div className="relative mt-1">
                                <Input
                                  id={fieldKey}
                                  type={field.type === "password" && !showKeys[fieldKey] ? "password" : "text"}
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
                                    onClick={() => toggleShowKey(provider.id, field.key)}
                                  >
                                    {showKeys[fieldKey] ? (
                                      <EyeOff className="h-3.5 w-3.5" />
                                    ) : (
                                      <Eye className="h-3.5 w-3.5" />
                                    )}
                                  </Button>
                                )}
                              </div>
                            </div>
                          );
                        })}

                        {config[provider.id] && Object.keys(config[provider.id]).some(k => config[provider.id][k]) && (
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
                      </CardContent>
                    </Card>
                  ))}
                </div>

                {/* Save Button */}
                {hasUnsavedChanges() && (
                  <div className="fixed bottom-6 right-6">
                    <Button onClick={handleSave} disabled={saving} size="lg">
                      <Save className="mr-2 h-4 w-4" />
                      {saving ? "Saving..." : "Save Changes"}
                    </Button>
                  </div>
                )}
              </>
            )}

            {activeTab === "models" && <ModelManager apiKeys={config} authToken={authToken} />}
          </div>
        </ScrollArea>
      </div>
    </div>
  );
}
