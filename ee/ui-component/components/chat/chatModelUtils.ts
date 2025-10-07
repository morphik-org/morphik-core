export const LEMONADE_DEFAULT_HOSTS = {
  direct: "localhost",
  docker: "host.docker.internal",
} as const;

export type LemonadeHostOption = keyof typeof LEMONADE_DEFAULT_HOSTS;

export type AvailableModel = {
  id: string;
  name: string;
  provider: string;
  description?: string;
  enabled?: boolean;
  config?: Record<string, unknown>;
};

export type LemonadeConnectionInfo = {
  port: string;
  hostMode: LemonadeHostOption;
  backendHost: string;
  uiApiBase: string;
  directApiBase: string;
  dockerApiBase: string;
  selectedApiBase: string;
  candidateApiBases: string[];
};

export const asRecord = (value: unknown): Record<string, unknown> | null => {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return null;
};

export const asString = (value: unknown): string | undefined => {
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : undefined;
  }
  return undefined;
};

export const asPortString = (value: unknown): string | undefined => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value);
  }
  return asString(value);
};

export const fromRecord = (record: Record<string, unknown> | null, key: string): unknown => {
  if (!record) return undefined;
  return Object.prototype.hasOwnProperty.call(record, key) ? record[key] : undefined;
};

const normalizeApiBase = (value?: string | null): string | undefined => {
  if (!value || typeof value !== "string") return undefined;
  const trimmed = value.trim();
  if (trimmed.length === 0) return undefined;
  if (!/^https?:\/\//i.test(trimmed)) return undefined;
  return trimmed.replace(/\/+$/, "");
};

const pushCandidateBase = (list: string[], value: unknown) => {
  const candidate = typeof value === "string" ? value : asString(value);
  const normalized = normalizeApiBase(candidate);
  if (normalized && !list.includes(normalized)) {
    list.push(normalized);
  }
};

const ensureApiBase = (value: string | undefined, fallback: string) => {
  return normalizeApiBase(value) ?? normalizeApiBase(fallback) ?? fallback;
};

const parseUrlForHostPort = (value?: string) => {
  const normalized = normalizeApiBase(value);
  if (!normalized) {
    return { host: undefined, port: undefined };
  }

  try {
    const url = new URL(normalized);
    const port = url.port || (url.protocol.toLowerCase() === "https:" ? "443" : "80");
    return { host: url.hostname, port };
  } catch {
    return { host: undefined, port: undefined };
  }
};

export const inferLemonadeConnectionInfo = (existingModels: AvailableModel[]): LemonadeConnectionInfo | null => {
  let port: string | undefined;
  let hostMode: LemonadeHostOption = "direct";
  let backendHost: string | undefined;
  let uiApiBase: string | undefined;
  let directApiBase: string | undefined;
  let dockerApiBase: string | undefined;
  let selectedApiBase: string | undefined;
  const discoveredBases: string[] = [];

  const updateHostMode = (value: string | undefined) => {
    if (value === "docker") {
      hostMode = "docker";
    }
  };

  const loadMetadata = (metadata: Record<string, unknown>) => {
    port = port ?? asPortString(fromRecord(metadata, "port"));
    updateHostMode(asString(fromRecord(metadata, "host_mode")));
    backendHost = backendHost ?? asString(fromRecord(metadata, "backend_host"));
    const uiCandidate = asString(fromRecord(metadata, "ui_api_base"));
    if (uiCandidate) {
      const normalizedUi = normalizeApiBase(uiCandidate) ?? uiCandidate;
      uiApiBase = uiApiBase ?? normalizedUi;
      pushCandidateBase(discoveredBases, normalizedUi);
    }
    const apiBases = asRecord(fromRecord(metadata, "api_bases"));
    if (apiBases) {
      const directCandidate = asString(fromRecord(apiBases, "direct"));
      if (directCandidate) {
        const normalizedDirect = normalizeApiBase(directCandidate) ?? directCandidate;
        directApiBase = directApiBase ?? normalizedDirect;
        pushCandidateBase(discoveredBases, normalizedDirect);
      }
      const dockerCandidate = asString(fromRecord(apiBases, "docker"));
      if (dockerCandidate) {
        const normalizedDocker = normalizeApiBase(dockerCandidate) ?? dockerCandidate;
        dockerApiBase = dockerApiBase ?? normalizedDocker;
        pushCandidateBase(discoveredBases, normalizedDocker);
      }
      const selectedCandidate = asString(fromRecord(apiBases, "selected"));
      if (selectedCandidate) {
        const normalizedSelected = normalizeApiBase(selectedCandidate) ?? selectedCandidate;
        selectedApiBase = selectedApiBase ?? normalizedSelected;
        pushCandidateBase(discoveredBases, normalizedSelected);
      }
    }
  };

  if (typeof window !== "undefined") {
    try {
      const savedConfigRaw = window.localStorage?.getItem("morphik_api_keys");
      if (savedConfigRaw) {
        const parsedConfig = JSON.parse(savedConfigRaw) as Record<string, unknown>;
        const lemonadeConfig = asRecord(fromRecord(parsedConfig, "lemonade"));
        if (lemonadeConfig) {
          port = port ?? asPortString(fromRecord(lemonadeConfig, "port"));
          updateHostMode(
            asString(fromRecord(lemonadeConfig, "hostMode")) || asString(fromRecord(lemonadeConfig, "host_mode"))
          );
          backendHost = asString(fromRecord(lemonadeConfig, "host")) ?? backendHost;

          const savedApiBase = asString(fromRecord(lemonadeConfig, "api_base"));
          if (savedApiBase) {
            const normalizedSavedApiBase = normalizeApiBase(savedApiBase) ?? savedApiBase;
            selectedApiBase = selectedApiBase ?? normalizedSavedApiBase;
            pushCandidateBase(discoveredBases, normalizedSavedApiBase);
          }

          const savedUiBase = asString(fromRecord(lemonadeConfig, "ui_api_base"));
          if (savedUiBase) {
            const normalizedSavedUi = normalizeApiBase(savedUiBase) ?? savedUiBase;
            uiApiBase = uiApiBase ?? normalizedSavedUi;
            pushCandidateBase(discoveredBases, normalizedSavedUi);
          }

          const savedApiBases = asRecord(fromRecord(lemonadeConfig, "api_bases"));
          if (savedApiBases) {
            const directCandidate = asString(fromRecord(savedApiBases, "direct"));
            if (directCandidate) {
              const normalizedDirect = normalizeApiBase(directCandidate) ?? directCandidate;
              directApiBase = directApiBase ?? normalizedDirect;
              pushCandidateBase(discoveredBases, normalizedDirect);
            }

            const dockerCandidate = asString(fromRecord(savedApiBases, "docker"));
            if (dockerCandidate) {
              const normalizedDocker = normalizeApiBase(dockerCandidate) ?? dockerCandidate;
              dockerApiBase = dockerApiBase ?? normalizedDocker;
              pushCandidateBase(discoveredBases, normalizedDocker);
            }

            const selectedCandidate = asString(fromRecord(savedApiBases, "selected"));
            if (selectedCandidate) {
              const normalizedSelected = normalizeApiBase(selectedCandidate) ?? selectedCandidate;
              selectedApiBase = selectedApiBase ?? normalizedSelected;
              pushCandidateBase(discoveredBases, normalizedSelected);
            }
          }
        }
      }
    } catch (err) {
      console.error("Failed to parse saved Lemonade configuration:", err);
    }
  }

  for (const model of existingModels) {
    if (model.provider !== "lemonade") continue;
    const configRecord = asRecord(model.config);
    if (configRecord) {
      pushCandidateBase(discoveredBases, fromRecord(configRecord, "api_base"));
      const metadata = asRecord(fromRecord(configRecord, "lemonade_metadata"));
      if (metadata) {
        loadMetadata(metadata);
      }
    }
  }

  if (!port || !backendHost) {
    const inferenceCandidates = [selectedApiBase, directApiBase, dockerApiBase, uiApiBase, ...discoveredBases];
    for (const candidate of inferenceCandidates) {
      if (!candidate) continue;
      const { host, port: inferredPort } = parseUrlForHostPort(candidate);
      if (!port && inferredPort) {
        port = inferredPort;
      }
      if (!backendHost && host) {
        backendHost = host;
      }
      if (port && backendHost) {
        break;
      }
    }
  }

  if (!port) {
    return null;
  }

  const resolvedBackendHost = backendHost ?? LEMONADE_DEFAULT_HOSTS[hostMode];
  const finalUiApiBase = ensureApiBase(uiApiBase, `http://localhost:${port}/api/v1`);
  const finalDirectApiBase = ensureApiBase(directApiBase, `http://${LEMONADE_DEFAULT_HOSTS.direct}:${port}/api/v1`);
  const finalDockerApiBase = ensureApiBase(dockerApiBase, `http://${LEMONADE_DEFAULT_HOSTS.docker}:${port}/api/v1`);
  const finalSelectedApiBase = ensureApiBase(selectedApiBase, `http://${resolvedBackendHost}:${port}/api/v1`);

  const candidateApiBases: string[] = [];
  pushCandidateBase(candidateApiBases, finalUiApiBase);
  pushCandidateBase(candidateApiBases, finalSelectedApiBase);
  pushCandidateBase(candidateApiBases, finalDirectApiBase);
  pushCandidateBase(candidateApiBases, finalDockerApiBase);
  for (const base of discoveredBases) {
    pushCandidateBase(candidateApiBases, base);
  }

  return {
    port,
    hostMode,
    backendHost: resolvedBackendHost,
    uiApiBase: finalUiApiBase,
    directApiBase: finalDirectApiBase,
    dockerApiBase: finalDockerApiBase,
    selectedApiBase: finalSelectedApiBase,
    candidateApiBases,
  };
};
