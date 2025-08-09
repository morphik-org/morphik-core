import {
  ModelConfigResponse,
  ModelConfigCreate,
  ModelConfigUpdate,
  CustomModel,
  CustomModelCreate,
} from "@/components/types";

interface APIKeyConfig {
  [provider: string]: {
    apiKey?: string;
    baseUrl?: string;
    [key: string]: unknown;
  };
}

export class ModelConfigAPI {
  private authToken: string | null;
  private baseUrl: string;

  constructor(authToken: string | null, baseUrl?: string) {
    this.authToken = authToken;
    this.baseUrl = baseUrl || (process.env.NEXT_PUBLIC_API_URL as string) || "https://api.morphik.ai";
  }

  private getHeaders(): HeadersInit {
    const headers: HeadersInit = {
      "Content-Type": "application/json",
    };
    if (this.authToken) {
      headers["Authorization"] = `Bearer ${this.authToken}`;
    }
    return headers;
  }

  // List all model configurations
  async listConfigs(): Promise<ModelConfigResponse[]> {
    const response = await fetch(`${this.baseUrl}/model-config/`, {
      method: "GET",
      headers: this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Failed to list model configs: ${response.statusText}`);
    }

    return response.json();
  }

  // Get a specific model configuration
  async getConfig(configId: string): Promise<ModelConfigResponse> {
    const response = await fetch(`${this.baseUrl}/model-config/${configId}`, {
      method: "GET",
      headers: this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Failed to get model config: ${response.statusText}`);
    }

    return response.json();
  }

  // Create a new model configuration
  async createConfig(config: ModelConfigCreate): Promise<ModelConfigResponse> {
    const response = await fetch(`${this.baseUrl}/model-config/`, {
      method: "POST",
      headers: this.getHeaders(),
      body: JSON.stringify(config),
    });

    if (!response.ok) {
      throw new Error(`Failed to create model config: ${response.statusText}`);
    }

    return response.json();
  }

  // Update an existing model configuration
  async updateConfig(configId: string, update: ModelConfigUpdate): Promise<ModelConfigResponse> {
    const response = await fetch(`${this.baseUrl}/model-config/${configId}`, {
      method: "PUT",
      headers: this.getHeaders(),
      body: JSON.stringify(update),
    });

    if (!response.ok) {
      throw new Error(`Failed to update model config: ${response.statusText}`);
    }

    return response.json();
  }

  // Delete a model configuration
  async deleteConfig(configId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/model-config/${configId}`, {
      method: "DELETE",
      headers: this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Failed to delete model config: ${response.statusText}`);
    }
  }

  // List custom models
  async listCustomModels(): Promise<CustomModel[]> {
    const response = await fetch(`${this.baseUrl}/model-config/custom-models/list`, {
      method: "GET",
      headers: this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Failed to list custom models: ${response.statusText}`);
    }

    return response.json();
  }

  // Create a custom model
  async createCustomModel(model: CustomModelCreate): Promise<CustomModel> {
    const response = await fetch(`${this.baseUrl}/model-config/custom-models`, {
      method: "POST",
      headers: this.getHeaders(),
      body: JSON.stringify(model),
    });

    if (!response.ok) {
      throw new Error(`Failed to create custom model: ${response.statusText}`);
    }

    return response.json();
  }

  // Update a custom model
  async updateCustomModel(modelId: string, model: CustomModelCreate): Promise<CustomModel> {
    const response = await fetch(`${this.baseUrl}/model-config/custom-models/${modelId}`, {
      method: "PUT",
      headers: this.getHeaders(),
      body: JSON.stringify(model),
    });

    if (!response.ok) {
      throw new Error(`Failed to update custom model: ${response.statusText}`);
    }

    return response.json();
  }

  // Delete a custom model
  async deleteCustomModel(modelId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/model-config/custom-models/${modelId}`, {
      method: "DELETE",
      headers: this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Failed to delete custom model: ${response.statusText}`);
    }
  }

  // Helper method to sync localStorage data to backend
  async syncFromLocalStorage(): Promise<void> {
    try {
      // Get existing configs from backend
      const existingConfigs = await this.listConfigs();
      const existingProviders = new Set(existingConfigs.map(c => c.provider));

      // Get API keys from localStorage
      const savedApiKeys = localStorage.getItem("morphik_api_keys");
      if (savedApiKeys) {
        const apiKeys: APIKeyConfig = JSON.parse(savedApiKeys);

        // Create configs for each provider that has an API key
        for (const [provider, config] of Object.entries(apiKeys)) {
          if (config.apiKey && !existingProviders.has(provider)) {
            await this.createConfig({
              provider,
              config_data: config,
            });
          }
        }
      }

      // Get custom models from localStorage
      const savedModels = localStorage.getItem("morphik_custom_models");
      if (savedModels) {
        const models = JSON.parse(savedModels);
        const existingCustomModels = await this.listCustomModels();
        const existingModelNames = new Set(existingCustomModels.map(m => m.name));

        for (const model of models) {
          if (!existingModelNames.has(model.name)) {
            await this.createCustomModel({
              name: model.name,
              provider: model.provider,
              model_name: model.model_name,
              config: model.config,
            });
          }
        }
      }
    } catch (error) {
      console.error("Failed to sync from localStorage:", error);
    }
  }

  // Helper method to get merged config (backend + localStorage fallback)
  async getMergedConfig(): Promise<APIKeyConfig> {
    const config: APIKeyConfig = {};

    try {
      // Get configs from backend
      const backendConfigs = await this.listConfigs();
      for (const backendConfig of backendConfigs) {
        config[backendConfig.provider] = backendConfig.config_data;
      }
    } catch (error) {
      console.error("Failed to get backend configs:", error);
    }

    // Merge with localStorage (as fallback)
    const savedApiKeys = localStorage.getItem("morphik_api_keys");
    if (savedApiKeys) {
      const localConfig: APIKeyConfig = JSON.parse(savedApiKeys);
      for (const [provider, providerConfig] of Object.entries(localConfig)) {
        if (!config[provider]) {
          config[provider] = providerConfig;
        }
      }
    }

    return config;
  }
}
