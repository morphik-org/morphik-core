import { useState, useEffect, useCallback } from "react";
import type { QueryOptions } from "@/components/types";
import {
  AvailableModel,
  LEMONADE_DEFAULT_HOSTS,
  LemonadeHostOption,
  asPortString,
  asRecord,
  asString,
  fromRecord,
  inferLemonadeConnectionInfo,
} from "./chatModelUtils";

type ServerModelLike = {
  id: string;
  name: string;
  provider: string;
  description?: string;
  config?: Record<string, unknown>;
};

type SafeUpdateOption = <K extends keyof QueryOptions>(key: K, value: QueryOptions[K]) => void;

interface UseChatModelSelectorParams {
  apiBaseUrl: string;
  authToken: string | null;
  serverModels: ServerModelLike[];
  refreshServerModels: () => Promise<unknown>;
  safeUpdateOption: SafeUpdateOption;
}

interface UseChatModelSelectorResult {
  selectedModel: string;
  showModelSelector: boolean;
  setShowModelSelector: React.Dispatch<React.SetStateAction<boolean>>;
  availableModels: AvailableModel[];
  handleModelChange: (modelId: string) => void;
}

export const useChatModelSelector = ({
  apiBaseUrl,
  authToken,
  serverModels,
  refreshServerModels,
  safeUpdateOption,
}: UseChatModelSelectorParams): UseChatModelSelectorResult => {
  const [showModelSelector, setShowModelSelector] = useState(false);
  const [availableModels, setAvailableModels] = useState<AvailableModel[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");

  useEffect(() => {
    if (!showModelSelector) {
      return;
    }

    let cancelled = false;
    (async () => {
      try {
        await refreshServerModels();
      } catch (err) {
        if (!cancelled) {
          console.error("Failed to refresh models before opening selector:", err);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [showModelSelector, refreshServerModels]);

  useEffect(() => {
    let cancelled = false;

    const loadModelsAndConfig = async () => {
      const allModels: AvailableModel[] = serverModels.map(model => ({
        id: model.id,
        name: model.name,
        provider: model.provider,
        description: model.description,
        config: model.config,
      }));

      try {
        if (authToken) {
          const resp = await fetch(`${apiBaseUrl}/models/custom`, {
            headers: { Authorization: `Bearer ${authToken}` },
          });
          if (resp.ok) {
            const customModelsList = await resp.json();
            const customTransformed = customModelsList.map(
              (m: { id: string; name: string; provider: string; config?: Record<string, unknown> }) => ({
                id: `custom_${m.id}`,
                name: m.name,
                provider: m.provider,
                description: `Custom ${m.provider} model`,
                config: m.config,
              })
            );
            allModels.push(...customTransformed);
          }
        } else {
          const savedModels = localStorage.getItem("morphik_custom_models");
          if (savedModels) {
            try {
              const parsed = JSON.parse(savedModels);
              const customTransformed = parsed.map(
                (m: { id: string; name: string; provider: string; config?: Record<string, unknown> }) => ({
                  id: `custom_${m.id}`,
                  name: m.name,
                  provider: m.provider,
                  description: `Custom ${m.provider} model`,
                  config: m.config,
                })
              );
              allModels.push(...customTransformed);
            } catch (err) {
              console.error("Failed to parse custom models:", err);
            }
          }
        }
      } catch (err) {
        console.error("Failed to load custom models:", err);
      }

      const configured: Record<string, boolean> = {};
      try {
        if (authToken) {
          const resp = await fetch(`${apiBaseUrl}/api-keys`, {
            headers: { Authorization: `Bearer ${authToken}` },
          });
          if (resp.ok) {
            const apiKeys = await resp.json();
            for (const [prov, data] of Object.entries(apiKeys)) {
              const d = data as { configured?: boolean };
              configured[prov] = Boolean(d?.configured);
            }
          }
        } else if (typeof window !== "undefined") {
          const saved = localStorage.getItem("morphik_api_keys");
          if (saved) {
            try {
              const localCfg = JSON.parse(saved) as Record<string, { apiKey?: string }>;
              for (const [prov, val] of Object.entries(localCfg)) {
                configured[prov] = Boolean(val?.apiKey);
              }
            } catch (e) {
              console.error("Failed to parse local API keys:", e);
            }
          }
        }
      } catch (e) {
        console.error("Failed to load API key configuration:", e);
      }

      const lemonadeInfo = inferLemonadeConnectionInfo(allModels);
      if (lemonadeInfo) {
        const normalizeKey = (value: string | undefined) => {
          if (!value) return undefined;
          return value.replace(/^openai\//i, "").toLowerCase();
        };

        const knownLemonadeKeys = new Set<string>();
        const usedIds = new Set(allModels.map(model => model.id));

        for (const model of allModels) {
          if (model.provider !== "lemonade") continue;
          const configRecord = asRecord(model.config);
          const configModel = configRecord ? asString(fromRecord(configRecord, "model")) : undefined;
          const keySource = configModel || model.id || model.name;
          const normalized = normalizeKey(keySource);
          if (normalized) {
            knownLemonadeKeys.add(normalized);
          }
        }

        for (const base of lemonadeInfo.candidateApiBases) {
          if (cancelled) {
            break;
          }

          try {
            const timeoutSignal =
              typeof AbortSignal !== "undefined" && "timeout" in AbortSignal ? AbortSignal.timeout(4000) : undefined;
            const response = await fetch(`${base}/models`, {
              signal: timeoutSignal,
            });

            if (!response.ok) {
              continue;
            }

            const payload = await response.json();
            const candidatesRaw =
              Array.isArray(payload?.data) && payload.data.length > 0
                ? payload.data
                : Array.isArray(payload?.models) && payload.models.length > 0
                  ? payload.models
                  : Array.isArray(payload)
                    ? payload
                    : [];

            if (!Array.isArray(candidatesRaw) || candidatesRaw.length === 0) {
              continue;
            }

            for (const candidate of candidatesRaw) {
              const record = asRecord(candidate);
              if (!record) continue;
              const rawName = asString(fromRecord(record, "id")) || asString(fromRecord(record, "name"));
              if (!rawName) continue;

              const normalized = normalizeKey(rawName);
              if (!normalized || knownLemonadeKeys.has(normalized)) {
                continue;
              }

              const displayLabel = `Lemonade: ${rawName}`;
              const modelIdentifier = rawName.startsWith("openai/") ? rawName : `openai/${rawName}`;
              const newModelId = `lemonade_${normalized}`;
              const resolvedId = usedIds.has(newModelId)
                ? `lemonade_${normalized}_${Math.random().toString(36).slice(2, 8)}`
                : newModelId;

              knownLemonadeKeys.add(normalized);
              usedIds.add(resolvedId);

              allModels.push({
                id: resolvedId,
                name: displayLabel,
                provider: "lemonade",
                description: `Local Lemonade model (${modelIdentifier})`,
                config: {
                  model: modelIdentifier,
                  api_base: lemonadeInfo.selectedApiBase,
                  vision: normalized.includes("vision") || normalized.includes("vl"),
                  lemonade_metadata: {
                    host_mode: lemonadeInfo.hostMode,
                    port: lemonadeInfo.port,
                    backend_host: lemonadeInfo.backendHost,
                    ui_api_base: lemonadeInfo.uiApiBase,
                    api_bases: {
                      direct: lemonadeInfo.directApiBase,
                      docker: lemonadeInfo.dockerApiBase,
                      selected: lemonadeInfo.selectedApiBase,
                    },
                  },
                },
              });
            }

            break;
          } catch (err) {
            console.error(`Failed to load Lemonade models from ${base}`, err);
          }
        }
      }

      const doesProviderRequireKey = (prov: string) => {
        const requires = ["openai", "anthropic", "google", "groq", "deepseek", "together", "azure"];
        return requires.includes(prov);
      };

      const withEnabled = allModels.map(m => ({
        ...m,
        enabled: !doesProviderRequireKey(m.provider) || configured[m.provider] === true,
      }));

      if (!cancelled) {
        setAvailableModels(withEnabled);
      }
    };

    if (showModelSelector) {
      loadModelsAndConfig();
    }
    return () => {
      cancelled = true;
    };
  }, [showModelSelector, serverModels, authToken, apiBaseUrl]);

  const handleModelChange = useCallback(
    (modelId: string) => {
      setSelectedModel(modelId);

      if (modelId === "default") {
        safeUpdateOption("llm_config", undefined);
        return;
      }

      const apiKeysRaw = typeof window !== "undefined" ? localStorage.getItem("morphik_api_keys") : null;
      let parsedApiKeys: Record<string, unknown> | null = null;
      if (apiKeysRaw) {
        try {
          parsedApiKeys = JSON.parse(apiKeysRaw) as Record<string, unknown>;
        } catch (err) {
          console.error("Failed to parse API key configuration:", err);
        }
      }

      const lemonadeSettings = asRecord(fromRecord(parsedApiKeys, "lemonade"));

      const applyLemonadeOverrides = (configRecord: Record<string, unknown>) => {
        const metadata = asRecord(fromRecord(configRecord, "lemonade_metadata"));
        const apiBases = asRecord(metadata ? fromRecord(metadata, "api_bases") : undefined);

        const hostModeValue =
          asString(fromRecord(lemonadeSettings, "hostMode")) ||
          asString(fromRecord(lemonadeSettings, "host_mode")) ||
          asString(fromRecord(metadata, "host_mode"));

        const hostMode: LemonadeHostOption = hostModeValue === "docker" ? "docker" : "direct";

        const resolvedPort =
          asPortString(fromRecord(lemonadeSettings, "port")) ||
          asPortString(fromRecord(lemonadeSettings, "lemonade_port")) ||
          asPortString(fromRecord(metadata, "port")) ||
          asPortString(fromRecord(metadata, "lemonade_port"));

        let resolvedHost =
          asString(fromRecord(lemonadeSettings, "host")) || asString(fromRecord(metadata, "backend_host"));

        if (!resolvedHost) {
          resolvedHost = LEMONADE_DEFAULT_HOSTS[hostMode];
        }

        let resolvedApiBase =
          (hostMode === "docker"
            ? asString(fromRecord(apiBases, "docker"))
            : asString(fromRecord(apiBases, "direct"))) || asString(fromRecord(apiBases, "selected"));

        if (resolvedHost && resolvedPort) {
          resolvedApiBase = `http://${resolvedHost}:${resolvedPort}/api/v1`;
        }

        if (resolvedApiBase) {
          configRecord["api_base"] = resolvedApiBase;
        }

        delete configRecord["lemonade_metadata"];
      };

      if (modelId.startsWith("custom_")) {
        const savedModels = typeof window !== "undefined" ? localStorage.getItem("morphik_custom_models") : null;
        if (savedModels) {
          try {
            const customModels = JSON.parse(savedModels);
            const customModel = customModels.find((m: { id: string }) => `custom_${m.id}` === modelId);

            if (customModel) {
              const llmConfig: Record<string, unknown> = {
                ...(customModel.config as Record<string, unknown>),
              };

              if (customModel.provider === "lemonade") {
                applyLemonadeOverrides(llmConfig);
              }

              safeUpdateOption("llm_config", llmConfig);
              return;
            }
          } catch (err) {
            console.error("Failed to parse custom models:", err);
          }
        }

        const fallbackModel = availableModels.find(model => model.id === modelId);
        if (fallbackModel?.config) {
          const fallbackConfig = { ...(fallbackModel.config as Record<string, unknown>) };
          if (fallbackModel.provider === "lemonade") {
            applyLemonadeOverrides(fallbackConfig);
          }
          safeUpdateOption("llm_config", fallbackConfig);
          return;
        }
      }

      const existingModel = availableModels.find(model => model.id === modelId);
      if (existingModel?.config) {
        const config = { ...(existingModel.config as Record<string, unknown>) };
        if (existingModel.provider === "lemonade") {
          applyLemonadeOverrides(config);
        }
        safeUpdateOption("llm_config", config);
        return;
      }

      if (parsedApiKeys) {
        const providerConfig = parsedApiKeys as Record<string, { apiKey?: string; baseUrl?: string }>;
        const modelConfig: Record<string, unknown> = { model: modelId };

        if (modelId.startsWith("gpt")) {
          const openai = providerConfig.openai;
          if (openai?.apiKey) {
            modelConfig.api_key = openai.apiKey;
            if (openai.baseUrl) {
              modelConfig.base_url = openai.baseUrl;
            }
          }
        } else if (modelId.startsWith("claude")) {
          const anthropic = providerConfig.anthropic;
          if (anthropic?.apiKey) {
            modelConfig.api_key = anthropic.apiKey;
            if (anthropic.baseUrl) {
              modelConfig.base_url = anthropic.baseUrl;
            }
          }
        } else if (modelId.startsWith("gemini/")) {
          const google = providerConfig.google;
          if (google?.apiKey) {
            modelConfig.api_key = google.apiKey;
          }
        } else if (modelId.startsWith("groq/")) {
          const groq = providerConfig.groq;
          if (groq?.apiKey) {
            modelConfig.api_key = groq.apiKey;
          }
        } else if (modelId.startsWith("deepseek/")) {
          const deepseek = providerConfig.deepseek;
          if (deepseek?.apiKey) {
            modelConfig.api_key = deepseek.apiKey;
          }
        }

        safeUpdateOption("llm_config", modelConfig);
      }
    },
    [availableModels, safeUpdateOption]
  );

  return {
    selectedModel,
    showModelSelector,
    setShowModelSelector,
    availableModels,
    handleModelChange,
  };
};
