"use client";

import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Eye, EyeOff, Save, Trash2, ExternalLink, Loader2 } from "lucide-react";
import { showAlert } from "@/components/ui/alert-system";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ModelManager } from "./ModelManager";
// import { useHeader } // Removed - MorphikUI handles breadcrumbs from "@/contexts/header-context";
import { useChatContext } from "@/components/chat/chat-context";
import { useTheme } from "next-themes";

interface SettingsSectionProps {
  authToken?: string | null;
  onBackClick?: () => void;
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
    logo: {
      light: "/provider-logos/OpenAI-black-monoblossom.png",
      dark: "/provider-logos/OpenAI-white-monoblossom.png",
    },
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
    logo: { light: "/provider-logos/Anthropic-black.png", dark: "/provider-logos/Anthropic-white.png" },
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
    logo: { light: "/provider-logos/gemini.svg", dark: "/provider-logos/gemini.svg" },
    description: "Gemini Pro and Flash models",
    fields: [{ key: "apiKey", label: "API Key", type: "password", required: true }],
    docsUrl: "https://makersuite.google.com/app/apikey",
  },
  {
    id: "groq",
    name: "Groq",
    logo: { light: "/provider-logos/Groq Logo_Black 25.svg", dark: "/provider-logos/Groq Logo_White 25.svg" },
    description: "Fast inference for Llama and other models",
    fields: [{ key: "apiKey", label: "API Key", type: "password", required: true }],
    docsUrl: "https://console.groq.com/keys",
  },
  {
    id: "lemonade",
    name: "Lemonade",
    icon: "🍋",
    description: "Local AI models via Lemonade server",
    fields: [{ key: "port", label: "Port", type: "text", placeholder: "8020", required: true }],
    docsUrl: "https://lemonade-server.ai/",
    requiresApiKey: false,
    setupInstructions:
      "If not installed, download Lemonade from https://lemonade-server.ai/ and run it locally. After installation, paste the port it is running on (default: 8020).",
  },
];

import { useMorphik } from "@/contexts/morphik-context";
import type { CustomModel } from "@/components/types";

const LEMONADE_HOST_OPTIONS = {
  direct: {
    label: "Morphik running directly on this machine",
    host: "localhost",
  },
  docker: {
    label: "Morphik running inside Docker (accessing host Lemonade)",
    host: "host.docker.internal",
  },
} as const;

type LemonadeHostOption = keyof typeof LEMONADE_HOST_OPTIONS;

export function SettingsSection({ authToken }: SettingsSectionProps) {
  const { apiBaseUrl } = useMorphik();
  const { activeSettingsTab } = useChatContext();
  const [config, setConfig] = useState<APIKeyConfig>({});
  const [showKeys, setShowKeys] = useState<Record<string, boolean>>({});
  const [saving, setSaving] = useState(false);
  const [isClient, setIsClient] = useState(false);
  // const { setCustomBreadcrumbs } = useHeader();
  const { theme } = useTheme();
  const [testingLemonade, setTestingLemonade] = useState(false);
  const [lemonadeStatus, setLemonadeStatus] = useState<{ type: "success" | "error"; message: string } | null>(null);

  const applyLemonadeDefaults = (cfg: APIKeyConfig) => {
    const next = { ...cfg };
    if (next.lemonade) {
      const current = { ...next.lemonade };
      const defaultMode = "direct" as LemonadeHostOption;
      const proposedMode = current.hostMode as LemonadeHostOption | undefined;
      const mode =
        proposedMode && Object.prototype.hasOwnProperty.call(LEMONADE_HOST_OPTIONS, proposedMode)
          ? proposedMode
          : defaultMode;
      const resolvedHost =
        typeof current.host === "string" && current.host.trim().length > 0
          ? current.host
          : LEMONADE_HOST_OPTIONS[mode].host;
      next.lemonade = {
        ...current,
        hostMode: mode,
        host: resolvedHost,
      };
    }
    return next;
  };

  const updateConfig = (updater: (prev: APIKeyConfig) => APIKeyConfig) => {
    setConfig(prev => applyLemonadeDefaults(updater(prev)));
  };

  const getNormalizedLemonadeMode = (mode: unknown): LemonadeHostOption => {
    const defaultMode = "direct" as LemonadeHostOption;
    const proposedMode = mode as LemonadeHostOption | undefined;
    if (proposedMode && Object.prototype.hasOwnProperty.call(LEMONADE_HOST_OPTIONS, proposedMode)) {
      return proposedMode;
    }
    return defaultMode;
  };

  const handleLemonadeHostModeChange = (mode: LemonadeHostOption) => {
    setLemonadeStatus(null);
    updateConfig(prev => {
      const current = prev.lemonade ?? {};
      return {
        ...prev,
        lemonade: {
          ...current,
          hostMode: mode,
          host: LEMONADE_HOST_OPTIONS[mode].host,
        },
      };
    });
  };

  const lemonadeHostMode = getNormalizedLemonadeMode(config.lemonade?.hostMode);

  const resolveBackendLemonadeHost = () =>
    typeof config.lemonade?.host === "string" && config.lemonade.host.trim().length > 0
      ? config.lemonade.host
      : LEMONADE_HOST_OPTIONS[lemonadeHostMode].host;

  const handleTestLemonade = async () => {
    if (!isClient) return;

    const port = String(config.lemonade?.port ?? "").trim();
    if (!port) {
      const message = "Enter the Lemonade port before testing.";
      setLemonadeStatus({ type: "error", message });
      showAlert(message, { type: "error", duration: 4000 });
      return;
    }

    const backendHost = resolveBackendLemonadeHost();
    const uiHost = "localhost";
    const lemonadeUiUrl = `http://${uiHost}:${port}/api/v1`;
    const lemonadeBackendUrl = `http://${backendHost}:${port}/api/v1`;
    const directApiBase = `http://${LEMONADE_HOST_OPTIONS.direct.host}:${port}/api/v1`;
    const dockerApiBase = `http://${LEMONADE_HOST_OPTIONS.docker.host}:${port}/api/v1`;
    const healthUrl = `${lemonadeUiUrl}/health`;

    setTestingLemonade(true);
    setLemonadeStatus(null);

    try {
      const healthResponse = await fetch(healthUrl, {
        method: "GET",
        signal: AbortSignal.timeout(3000),
      });

      if (!healthResponse.ok) {
        throw new Error(`Health check failed with status ${healthResponse.status}`);
      }
    } catch (err) {
      console.error("Lemonade health check failed:", err);
      const message = "Could not reach Lemonade. Ensure it's running on the selected host and port.";
      setLemonadeStatus({ type: "error", message });
      showAlert(message, { type: "error", duration: 5000 });
      setTestingLemonade(false);
      return;
    }

    let lemonadeModels: Array<{ id?: unknown; name?: unknown }> = [];
    try {
      const response = await fetch(`${lemonadeUiUrl}/models`);
      if (!response.ok) {
        throw new Error(`Models endpoint failed with status ${response.status}`);
      }
      const data = await response.json();
      const candidates = (data?.data ?? data?.models ?? []) as Array<{ id?: unknown; name?: unknown }>;
      lemonadeModels = Array.isArray(candidates) ? candidates : [];
    } catch (err) {
      console.error("Fetching Lemonade models failed:", err);
      const message = "Connected to Lemonade, but fetching models failed.";
      setLemonadeStatus({ type: "error", message });
      showAlert(message, { type: "warning", duration: 5000 });
      setTestingLemonade(false);
      return;
    }

    let addedCount = 0;
    const timestamp = Date.now();
    let existingModels: CustomModel[] = [];

    try {
      const stored = localStorage.getItem("morphik_custom_models");
      if (stored) {
        existingModels = JSON.parse(stored) as CustomModel[];
      }
    } catch (err) {
      console.error("Failed to parse stored custom models:", err);
    }

    const updatedModels: CustomModel[] = [...existingModels];
    const toKey = (value: string) => `lemonade:${value.replace(/^openai\//, "")}`;
    const existingKeys = new Set(
      existingModels
        .filter(model => model.provider === "lemonade")
        .map(model => {
          const configModel = typeof model.config?.model === "string" ? model.config.model : "";
          const keySource = (configModel || model.model_name || model.name || "").toString();
          return toKey(keySource);
        })
    );

    for (const model of lemonadeModels) {
      let rawName: string | null = null;
      if (typeof model.id === "string" && model.id.trim().length > 0) {
        rawName = model.id.trim();
      } else if (typeof model.name === "string" && model.name.trim().length > 0) {
        rawName = model.name.trim();
      }

      if (!rawName) {
        continue;
      }

      const key = toKey(rawName);
      if (existingKeys.has(key)) {
        continue;
      }

      const newModel: CustomModel = {
        id: `custom_lemonade_${timestamp}_${rawName}`,
        name: `Lemonade: ${rawName}`,
        provider: "lemonade",
        model_name: rawName,
        config: {
          model: `openai/${rawName}`,
          api_base: lemonadeBackendUrl,
          vision: rawName.toLowerCase().includes("vision") || rawName.toLowerCase().includes("vl"),
          lemonade_metadata: {
            host_mode: lemonadeHostMode,
            port,
            backend_host: backendHost,
            ui_api_base: lemonadeUiUrl,
            api_bases: {
              direct: directApiBase,
              docker: dockerApiBase,
              selected: lemonadeBackendUrl,
            },
          },
        },
      };

      if (authToken) {
        try {
          const response = await fetch(`${apiBaseUrl}/models`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${authToken}`,
            },
            body: JSON.stringify({
              name: newModel.name,
              provider: newModel.provider,
              config: newModel.config,
            }),
          });

          if (response.ok) {
            const created = await response.json();
            if (created?.id) {
              newModel.id = created.id;
            }
          }
        } catch (err) {
          console.error("Failed to persist Lemonade model to backend:", err);
        }
      }

      updatedModels.push(newModel);
      existingKeys.add(key);
      addedCount += 1;
    }

    try {
      localStorage.setItem("morphik_custom_models", JSON.stringify(updatedModels));
    } catch (err) {
      console.error("Failed to save Lemonade models to localStorage:", err);
    }

    const message =
      addedCount > 0
        ? `Connected. Imported ${addedCount} Lemonade model${addedCount === 1 ? "" : "s"}.`
        : "Connection successful. No new models detected.";

    setLemonadeStatus({ type: "success", message });
    showAlert(message, { type: "success", duration: 4000 });

    setTestingLemonade(false);
  };

  // Ensure client-side rendering is complete before showing dynamic content
  useEffect(() => {
    setIsClient(true);
  }, []);

  // Load saved configuration from localStorage and backend
  useEffect(() => {
    if (!isClient) return;

    const loadConfig = async () => {
      // First load from localStorage (only in browser)
      let savedConfig: string | null = null;
      if (typeof window !== "undefined") {
        savedConfig = localStorage.getItem("morphik_api_keys");
        if (savedConfig) {
          try {
            setConfig(applyLemonadeDefaults(JSON.parse(savedConfig)));
          } catch (err) {
            console.error("Failed to parse saved API keys:", err);
          }
        }
      }

      // Then try to load from backend if we have authToken
      if (authToken) {
        try {
          const response = await fetch(`${apiBaseUrl}/api-keys`, {
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

            setConfig(applyLemonadeDefaults(mergedConfig));
          }
        } catch (err) {
          console.error("Failed to load API keys from backend:", err);
        }
      }
    };

    loadConfig();
  }, [authToken, isClient]);

  // Removed - MorphikUI handles breadcrumbs centrally
  // useEffect(() => {
  //   setCustomBreadcrumbs([{ label: "Home", href: "/" }, { label: "Settings" }]);
  //   return () => setCustomBreadcrumbs(null);
  // }, [setCustomBreadcrumbs]);

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
              fetch(`${apiBaseUrl}/api-keys`, {
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
    if (provider === "lemonade") {
      setLemonadeStatus(null);
    }
    updateConfig(prev => ({
      ...prev,
      [provider]: {
        ...prev[provider],
        [field]: value,
      },
    }));
  };

  const handleClearProvider = (provider: string) => {
    if (provider === "lemonade") {
      setLemonadeStatus(null);
    }
    updateConfig(prev => {
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
    if (!isClient || typeof window === "undefined") return false;
    const savedConfig = localStorage.getItem("morphik_api_keys");
    if (!savedConfig) return Object.keys(config).length > 0;
    try {
      return JSON.stringify(config) !== savedConfig;
    } catch {
      return true;
    }
  };

  return (
    <div className="h-full">
      <ScrollArea className="h-full">
        <div className="p-4">
          {activeSettingsTab === "api-keys" && (
            <>
              <div className="mb-4">
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
                          {"logo" in provider && provider.logo ? (
                            <img
                              src={theme === "dark" ? provider.logo.dark : provider.logo.light}
                              alt={`${provider.name} logo`}
                              className="h-8 w-8 object-contain"
                            />
                          ) : "icon" in provider ? (
                            <span className="text-2xl">{provider.icon as string}</span>
                          ) : (
                            <span className="text-2xl">🔧</span>
                          )}
                          <div>
                            <CardTitle>{provider.name}</CardTitle>
                            <CardDescription>{provider.description}</CardDescription>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          {provider.id === "lemonade" && (
                            <Button
                              variant="secondary"
                              size="sm"
                              onClick={() => handleTestLemonade()}
                              disabled={testingLemonade}
                            >
                              {testingLemonade && <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />}Test &amp;
                              Import
                            </Button>
                          )}
                          <Button variant="outline" size="sm" onClick={() => window.open(provider.docsUrl, "_blank")}>
                            <ExternalLink className="mr-1 h-3 w-3" />
                            {provider.id === "lemonade" ? "Download Lemonade" : "Get API Key"}
                          </Button>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {"setupInstructions" in provider && provider.setupInstructions && (
                        <div className="rounded-lg bg-muted p-3 text-sm text-muted-foreground">
                          {provider.setupInstructions}
                        </div>
                      )}
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
                                value={isClient ? (config[provider.id]?.[field.key] as string) || "" : ""}
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

                      {provider.id === "lemonade" && (
                        <div className="space-y-3">
                          <div>
                            <Label htmlFor="lemonade-host-mode">Lemonade location</Label>
                            <Select
                              value={lemonadeHostMode}
                              onValueChange={value => handleLemonadeHostModeChange(value as LemonadeHostOption)}
                            >
                              <SelectTrigger id="lemonade-host-mode" className="mt-1">
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                {(Object.keys(LEMONADE_HOST_OPTIONS) as LemonadeHostOption[]).map(option => (
                                  <SelectItem key={option} value={option}>
                                    {LEMONADE_HOST_OPTIONS[option].label}
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          </div>

                          {lemonadeStatus && (
                            <span
                              className={`inline-block text-sm ${
                                lemonadeStatus.type === "success" ? "text-green-600" : "text-red-600"
                              }`}
                            >
                              {lemonadeStatus.message}
                            </span>
                          )}
                        </div>
                      )}

                      {isClient &&
                        config[provider.id] &&
                        Object.keys(config[provider.id]).some(k => config[provider.id][k]) && (
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
              {isClient && hasUnsavedChanges() && (
                <div className="fixed bottom-6 right-6">
                  <Button onClick={handleSave} disabled={saving} size="lg">
                    <Save className="mr-2 h-4 w-4" />
                    {saving ? "Saving..." : "Save Changes"}
                  </Button>
                </div>
              )}
            </>
          )}

          {activeSettingsTab === "models" && <ModelManager apiKeys={config} authToken={authToken} />}
        </div>
      </ScrollArea>
    </div>
  );
}
