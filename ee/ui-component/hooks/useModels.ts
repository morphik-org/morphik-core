import { useState, useEffect, useCallback } from "react";

interface Model {
  id: string;
  name: string;
  provider: string;
  description?: string;
  config?: Record<string, unknown>;
  model?: string;
  modelKey?: string;
}

interface ModelAPIResponse {
  models?: Model[];
  chat_models?: Array<{
    config: { model_name?: string };
    model: string;
    id: string;
    provider: string;
  }>;
}

// Global cache for models
const modelsCache = new Map<string, { models: Model[]; timestamp: number }>();
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

export const clearModelsCache = (apiBaseUrl?: string) => {
  if (apiBaseUrl) {
    modelsCache.delete(apiBaseUrl);
  } else {
    modelsCache.clear();
  }
};

export function useModels(apiBaseUrl: string, authToken: string | null) {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchModels = useCallback(
    async (forceRefresh = false) => {
      const cacheKey = apiBaseUrl;
      const cached = modelsCache.get(cacheKey);

      // Check if we have valid cached data
      if (!forceRefresh && cached && Date.now() - cached.timestamp < CACHE_DURATION) {
        setModels(cached.models);
        setLoading(false);
        return cached.models;
      }

      try {
        setLoading(true);
        setError(null);

        const response = await fetch(`${apiBaseUrl}/models`, {
          headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch models: ${response.statusText}`);
        }

        const data: ModelAPIResponse = await response.json();
        let transformedModels: Model[] = [];

        // Handle different response formats
        if (data.models) {
          // Direct models format
          transformedModels = data.models.map(model => ({
            ...model,
            config: model.config,
            model: model.model,
            modelKey: model.modelKey,
          }));
        } else if (data.chat_models) {
          // Transform chat_models format
          transformedModels = data.chat_models.map(model => ({
            id: model.config.model_name || model.model,
            name: model.id.replace(/_/g, " ").replace(/\b\w/g, (l: string) => l.toUpperCase()),
            provider: model.provider,
            description: `Model: ${model.model}`,
            config: model.config,
            model: model.model,
            modelKey: model.id,
          }));
        }

        // Update cache
        modelsCache.set(cacheKey, {
          models: transformedModels,
          timestamp: Date.now(),
        });

        setModels(transformedModels);
        return transformedModels;
      } catch (err) {
        console.error("Failed to fetch models:", err);
        setError(err instanceof Error ? err : new Error("Failed to fetch models"));
        return [];
      } finally {
        setLoading(false);
      }
    },
    [apiBaseUrl, authToken]
  );

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  const refresh = useCallback(() => {
    return fetchModels(true);
  }, [fetchModels]);

  return { models, loading, error, refresh };
}
