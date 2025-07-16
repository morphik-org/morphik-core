# Ping

Types:

- <code><a href="./src/resources/ping.ts">PingCheckResponse</a></code>

Methods:

- <code title="get /ping">client.ping.<a href="./src/resources/ping.ts">check</a>() -> PingCheckResponse</code>

# Models

Types:

- <code><a href="./src/resources/models.ts">ModelResponse</a></code>
- <code><a href="./src/resources/models.ts">ModelListResponse</a></code>
- <code><a href="./src/resources/models.ts">ModelDeleteResponse</a></code>
- <code><a href="./src/resources/models.ts">ModelListAvailableResponse</a></code>
- <code><a href="./src/resources/models.ts">ModelListCustomResponse</a></code>

Methods:

- <code title="post /models">client.models.<a href="./src/resources/models.ts">create</a>({ ...params }) -> ModelResponse</code>
- <code title="get /models">client.models.<a href="./src/resources/models.ts">list</a>() -> ModelListResponse</code>
- <code title="delete /models/{model_id}">client.models.<a href="./src/resources/models.ts">delete</a>(modelID) -> ModelDeleteResponse</code>
- <code title="get /models/available">client.models.<a href="./src/resources/models.ts">listAvailable</a>() -> unknown</code>
- <code title="get /models/custom">client.models.<a href="./src/resources/models.ts">listCustom</a>() -> ModelListCustomResponse</code>

# Ingest

Types:

- <code><a href="./src/resources/ingest.ts">Document</a></code>
- <code><a href="./src/resources/ingest.ts">TextRequest</a></code>
- <code><a href="./src/resources/ingest.ts">IngestBatchIngestFilesResponse</a></code>

Methods:

- <code title="post /ingest/files">client.ingest.<a href="./src/resources/ingest.ts">batchIngestFiles</a>({ ...params }) -> IngestBatchIngestFilesResponse</code>
- <code title="post /ingest/file">client.ingest.<a href="./src/resources/ingest.ts">ingestFile</a>({ ...params }) -> Document</code>
- <code title="post /ingest/text">client.ingest.<a href="./src/resources/ingest.ts">ingestText</a>({ ...params }) -> Document</code>

# Folders

Types:

- <code><a href="./src/resources/folders/folders.ts">Folder</a></code>
- <code><a href="./src/resources/folders/folders.ts">FolderListResponse</a></code>
- <code><a href="./src/resources/folders/folders.ts">FolderDeleteResponse</a></code>
- <code><a href="./src/resources/folders/folders.ts">FolderListSummariesResponse</a></code>
- <code><a href="./src/resources/folders/folders.ts">FolderSetRuleResponse</a></code>

Methods:

- <code title="post /folders">client.folders.<a href="./src/resources/folders/folders.ts">create</a>({ ...params }) -> Folder</code>
- <code title="get /folders/{folder_id}">client.folders.<a href="./src/resources/folders/folders.ts">retrieve</a>(folderID) -> Folder</code>
- <code title="get /folders">client.folders.<a href="./src/resources/folders/folders.ts">list</a>() -> FolderListResponse</code>
- <code title="delete /folders/{folder_name}">client.folders.<a href="./src/resources/folders/folders.ts">delete</a>(folderName) -> FolderDeleteResponse</code>
- <code title="get /folders/summary">client.folders.<a href="./src/resources/folders/folders.ts">listSummaries</a>() -> FolderListSummariesResponse</code>
- <code title="post /folders/{folder_id}/set_rule">client.folders.<a href="./src/resources/folders/folders.ts">setRule</a>(folderID, { ...params }) -> FolderSetRuleResponse</code>

## Documents

Types:

- <code><a href="./src/resources/folders/documents.ts">DocumentAddResponse</a></code>

Methods:

- <code title="post /folders/{folder_id}/documents/{document_id}">client.folders.documents.<a href="./src/resources/folders/documents.ts">add</a>(documentID, { ...params }) -> DocumentAddResponse</code>
- <code title="delete /folders/{folder_id}/documents/{document_id}">client.folders.documents.<a href="./src/resources/folders/documents.ts">remove</a>(documentID, { ...params }) -> DocumentDeleteResponse</code>

## Workflows

Types:

- <code><a href="./src/resources/folders/workflows.ts">WorkflowListResponse</a></code>
- <code><a href="./src/resources/folders/workflows.ts">WorkflowAssociateResponse</a></code>
- <code><a href="./src/resources/folders/workflows.ts">WorkflowDisassociateResponse</a></code>

Methods:

- <code title="get /folders/{folder_id}/workflows">client.folders.workflows.<a href="./src/resources/folders/workflows.ts">list</a>(folderID) -> WorkflowListResponse</code>
- <code title="post /folders/{folder_id}/workflows/{workflow_id}">client.folders.workflows.<a href="./src/resources/folders/workflows.ts">associate</a>(workflowID, { ...params }) -> WorkflowAssociateResponse</code>
- <code title="delete /folders/{folder_id}/workflows/{workflow_id}">client.folders.workflows.<a href="./src/resources/folders/workflows.ts">disassociate</a>(workflowID, { ...params }) -> WorkflowDisassociateResponse</code>

# Workflows

Types:

- <code><a href="./src/resources/workflows/workflows.ts">Workflow</a></code>
- <code><a href="./src/resources/workflows/workflows.ts">WorkflowRun</a></code>
- <code><a href="./src/resources/workflows/workflows.ts">WorkflowListResponse</a></code>
- <code><a href="./src/resources/workflows/workflows.ts">WorkflowDeleteResponse</a></code>

Methods:

- <code title="post /workflows">client.workflows.<a href="./src/resources/workflows/workflows.ts">create</a>({ ...params }) -> Workflow</code>
- <code title="get /workflows/{workflow_id}">client.workflows.<a href="./src/resources/workflows/workflows.ts">retrieve</a>(workflowID) -> Workflow</code>
- <code title="put /workflows/{workflow_id}">client.workflows.<a href="./src/resources/workflows/workflows.ts">update</a>(workflowID, { ...params }) -> Workflow</code>
- <code title="get /workflows">client.workflows.<a href="./src/resources/workflows/workflows.ts">list</a>() -> WorkflowListResponse</code>
- <code title="delete /workflows/{workflow_id}">client.workflows.<a href="./src/resources/workflows/workflows.ts">delete</a>(workflowID) -> WorkflowDeleteResponse</code>
- <code title="post /workflows/{workflow_id}/run/{document_id}">client.workflows.<a href="./src/resources/workflows/workflows.ts">run</a>(documentID, { ...params }) -> WorkflowRun</code>

## Runs

Types:

- <code><a href="./src/resources/workflows/runs.ts">RunListResponse</a></code>
- <code><a href="./src/resources/workflows/runs.ts">RunDeleteResponse</a></code>

Methods:

- <code title="get /workflows/runs/{run_id}">client.workflows.runs.<a href="./src/resources/workflows/runs.ts">retrieve</a>(runID) -> WorkflowRun</code>
- <code title="get /workflows/{workflow_id}/runs">client.workflows.runs.<a href="./src/resources/workflows/runs.ts">list</a>(workflowID) -> RunListResponse</code>
- <code title="delete /workflows/runs/{run_id}">client.workflows.runs.<a href="./src/resources/workflows/runs.ts">delete</a>(runID) -> RunDeleteResponse</code>

# ModelConfig

Types:

- <code><a href="./src/resources/model-config/model-config.ts">ModelConfig</a></code>
- <code><a href="./src/resources/model-config/model-config.ts">ModelConfigListResponse</a></code>
- <code><a href="./src/resources/model-config/model-config.ts">ModelConfigDeleteResponse</a></code>

Methods:

- <code title="post /model-config/">client.modelConfig.<a href="./src/resources/model-config/model-config.ts">create</a>({ ...params }) -> ModelConfig</code>
- <code title="get /model-config/{config_id}">client.modelConfig.<a href="./src/resources/model-config/model-config.ts">retrieve</a>(configID) -> ModelConfig</code>
- <code title="put /model-config/{config_id}">client.modelConfig.<a href="./src/resources/model-config/model-config.ts">update</a>(configID, { ...params }) -> ModelConfig</code>
- <code title="get /model-config/">client.modelConfig.<a href="./src/resources/model-config/model-config.ts">list</a>() -> ModelConfigListResponse</code>
- <code title="delete /model-config/{config_id}">client.modelConfig.<a href="./src/resources/model-config/model-config.ts">delete</a>(configID) -> ModelConfigDeleteResponse</code>

## CustomModels

Types:

- <code><a href="./src/resources/model-config/custom-models.ts">CustomModel</a></code>
- <code><a href="./src/resources/model-config/custom-models.ts">CustomModelListResponse</a></code>

Methods:

- <code title="post /model-config/custom-models">client.modelConfig.customModels.<a href="./src/resources/model-config/custom-models.ts">create</a>({ ...params }) -> CustomModel</code>
- <code title="get /model-config/custom-models/list">client.modelConfig.customModels.<a href="./src/resources/model-config/custom-models.ts">list</a>() -> CustomModelListResponse</code>

# APIKeys

Types:

- <code><a href="./src/resources/api-keys.ts">APIKeyCreateResponse</a></code>
- <code><a href="./src/resources/api-keys.ts">APIKeyListResponse</a></code>

Methods:

- <code title="post /api-keys">client.apiKeys.<a href="./src/resources/api-keys.ts">create</a>({ ...params }) -> APIKeyCreateResponse</code>
- <code title="get /api-keys">client.apiKeys.<a href="./src/resources/api-keys.ts">list</a>() -> APIKeyListResponse</code>

# Logs

Types:

- <code><a href="./src/resources/logs.ts">LogListResponse</a></code>

Methods:

- <code title="get /logs/">client.logs.<a href="./src/resources/logs.ts">list</a>({ ...params }) -> LogListResponse</code>

# Cache

Types:

- <code><a href="./src/resources/cache.ts">CompletionResponse</a></code>
- <code><a href="./src/resources/cache.ts">CacheCreateResponse</a></code>
- <code><a href="./src/resources/cache.ts">CacheRetrieveResponse</a></code>
- <code><a href="./src/resources/cache.ts">CacheUpdateResponse</a></code>
- <code><a href="./src/resources/cache.ts">CacheAddDocsResponse</a></code>

Methods:

- <code title="post /cache/create">client.cache.<a href="./src/resources/cache.ts">create</a>({ ...params }) -> CacheCreateResponse</code>
- <code title="get /cache/{name}">client.cache.<a href="./src/resources/cache.ts">retrieve</a>(name) -> CacheRetrieveResponse</code>
- <code title="post /cache/{name}/update">client.cache.<a href="./src/resources/cache.ts">update</a>(name) -> CacheUpdateResponse</code>
- <code title="post /cache/{name}/add_docs">client.cache.<a href="./src/resources/cache.ts">addDocs</a>(name, [ ...body ]) -> CacheAddDocsResponse</code>
- <code title="post /cache/{name}/query">client.cache.<a href="./src/resources/cache.ts">query</a>(name, { ...params }) -> CompletionResponse</code>

# Graph

Types:

- <code><a href="./src/resources/graph/graph.ts">EntityExtractionPromptOverride</a></code>
- <code><a href="./src/resources/graph/graph.ts">EntityResolutionPromptOverride</a></code>
- <code><a href="./src/resources/graph/graph.ts">Graph</a></code>
- <code><a href="./src/resources/graph/graph.ts">GraphPromptOverrides</a></code>
- <code><a href="./src/resources/graph/graph.ts">GraphListResponse</a></code>
- <code><a href="./src/resources/graph/graph.ts">GraphDeleteResponse</a></code>
- <code><a href="./src/resources/graph/graph.ts">GraphStatusResponse</a></code>
- <code><a href="./src/resources/graph/graph.ts">GraphVisualizationResponse</a></code>

Methods:

- <code title="post /graph/create">client.graph.<a href="./src/resources/graph/graph.ts">create</a>({ ...params }) -> Graph</code>
- <code title="get /graph/{name}">client.graph.<a href="./src/resources/graph/graph.ts">retrieve</a>(name, { ...params }) -> Graph</code>
- <code title="post /graph/{name}/update">client.graph.<a href="./src/resources/graph/graph.ts">update</a>(name, { ...params }) -> Graph</code>
- <code title="get /graph/">client.graph.<a href="./src/resources/graph/graph.ts">list</a>({ ...params }) -> GraphListResponse</code>
- <code title="delete /graph/{name}">client.graph.<a href="./src/resources/graph/graph.ts">delete</a>(name) -> GraphDeleteResponse</code>
- <code title="get /graph/{name}/status">client.graph.<a href="./src/resources/graph/graph.ts">status</a>(name, { ...params }) -> GraphStatusResponse</code>
- <code title="get /graph/{name}/visualization">client.graph.<a href="./src/resources/graph/graph.ts">visualization</a>(name, { ...params }) -> GraphVisualizationResponse</code>

## Workflow

Types:

- <code><a href="./src/resources/graph/workflow.ts">WorkflowStatusResponse</a></code>

Methods:

- <code title="get /graph/workflow/{workflow_id}/status">client.graph.workflow.<a href="./src/resources/graph/workflow.ts">status</a>(workflowID, { ...params }) -> WorkflowStatusResponse</code>

# Ee

Types:

- <code><a href="./src/resources/ee/ee.ts">EeCreateAppResponse</a></code>

Methods:

- <code title="post /ee/create_app">client.ee.<a href="./src/resources/ee/ee.ts">createApp</a>({ ...params }) -> EeCreateAppResponse</code>

## Apps

Types:

- <code><a href="./src/resources/ee/apps.ts">AppCreateResponse</a></code>
- <code><a href="./src/resources/ee/apps.ts">AppDeleteResponse</a></code>

Methods:

- <code title="post /ee/apps">client.ee.apps.<a href="./src/resources/ee/apps.ts">create</a>({ ...params }) -> AppCreateResponse</code>
- <code title="delete /ee/apps">client.ee.apps.<a href="./src/resources/ee/apps.ts">delete</a>({ ...params }) -> AppDeleteResponse</code>

## Connectors

Types:

- <code><a href="./src/resources/ee/connectors/connectors.ts">ConnectorDisconnectResponse</a></code>
- <code><a href="./src/resources/ee/connectors/connectors.ts">ConnectorGetAuthStatusResponse</a></code>
- <code><a href="./src/resources/ee/connectors/connectors.ts">ConnectorHandleOAuthCallbackResponse</a></code>
- <code><a href="./src/resources/ee/connectors/connectors.ts">ConnectorIngestFileResponse</a></code>
- <code><a href="./src/resources/ee/connectors/connectors.ts">ConnectorListFilesResponse</a></code>

Methods:

- <code title="post /ee/connectors/{connector_type}/disconnect">client.ee.connectors.<a href="./src/resources/ee/connectors/connectors.ts">disconnect</a>(connectorType) -> ConnectorDisconnectResponse</code>
- <code title="get /ee/connectors/{connector_type}/auth_status">client.ee.connectors.<a href="./src/resources/ee/connectors/connectors.ts">getAuthStatus</a>(connectorType) -> ConnectorGetAuthStatusResponse</code>
- <code title="get /ee/connectors/{connector_type}/oauth2callback">client.ee.connectors.<a href="./src/resources/ee/connectors/connectors.ts">handleOAuthCallback</a>(connectorType, { ...params }) -> unknown</code>
- <code title="post /ee/connectors/{connector_type}/ingest">client.ee.connectors.<a href="./src/resources/ee/connectors/connectors.ts">ingestFile</a>(connectorType, { ...params }) -> ConnectorIngestFileResponse</code>
- <code title="get /ee/connectors/{connector_type}/files">client.ee.connectors.<a href="./src/resources/ee/connectors/connectors.ts">listFiles</a>(connectorType, { ...params }) -> ConnectorListFilesResponse</code>

### Auth

Types:

- <code><a href="./src/resources/ee/connectors/auth.ts">AuthFinalizeManualAuthResponse</a></code>
- <code><a href="./src/resources/ee/connectors/auth.ts">AuthGetInitiateAuthURLResponse</a></code>

Methods:

- <code title="post /ee/connectors/{connector_type}/auth/finalize">client.ee.connectors.auth.<a href="./src/resources/ee/connectors/auth.ts">finalizeManualAuth</a>(connectorType, { ...params }) -> AuthFinalizeManualAuthResponse</code>
- <code title="get /ee/connectors/{connector_type}/auth/initiate_url">client.ee.connectors.auth.<a href="./src/resources/ee/connectors/auth.ts">getInitiateAuthURL</a>(connectorType, { ...params }) -> AuthGetInitiateAuthURLResponse</code>

# Retrieve

Types:

- <code><a href="./src/resources/retrieve/retrieve.ts">RetrieveCreateDocsResponse</a></code>

Methods:

- <code title="post /retrieve/docs">client.retrieve.<a href="./src/resources/retrieve/retrieve.ts">createDocs</a>({ ...params }) -> RetrieveCreateDocsResponse</code>

## Chunks

Types:

- <code><a href="./src/resources/retrieve/chunks.ts">ChunkResult</a></code>
- <code><a href="./src/resources/retrieve/chunks.ts">RetrieveRequest</a></code>
- <code><a href="./src/resources/retrieve/chunks.ts">ChunkCreateResponse</a></code>
- <code><a href="./src/resources/retrieve/chunks.ts">ChunkCreateGroupedResponse</a></code>

Methods:

- <code title="post /retrieve/chunks">client.retrieve.chunks.<a href="./src/resources/retrieve/chunks.ts">create</a>({ ...params }) -> ChunkCreateResponse</code>
- <code title="post /retrieve/chunks/grouped">client.retrieve.chunks.<a href="./src/resources/retrieve/chunks.ts">createGrouped</a>({ ...params }) -> ChunkCreateGroupedResponse</code>

# Batch

Types:

- <code><a href="./src/resources/batch.ts">BatchRetrieveChunksResponse</a></code>
- <code><a href="./src/resources/batch.ts">BatchRetrieveDocumentsResponse</a></code>

Methods:

- <code title="post /batch/chunks">client.batch.<a href="./src/resources/batch.ts">retrieveChunks</a>({ ...params }) -> BatchRetrieveChunksResponse</code>
- <code title="post /batch/documents">client.batch.<a href="./src/resources/batch.ts">retrieveDocuments</a>({ ...params }) -> BatchRetrieveDocumentsResponse</code>

# Query

Methods:

- <code title="post /query">client.query.<a href="./src/resources/query.ts">generateCompletion</a>({ ...params }) -> CompletionResponse</code>

# Agent

Types:

- <code><a href="./src/resources/agent.ts">AgentExecuteQueryResponse</a></code>

Methods:

- <code title="post /agent">client.agent.<a href="./src/resources/agent.ts">executeQuery</a>({ ...params }) -> AgentExecuteQueryResponse</code>

# Usage

Types:

- <code><a href="./src/resources/usage.ts">UsageListRecentResponse</a></code>
- <code><a href="./src/resources/usage.ts">UsageRetrieveStatsResponse</a></code>

Methods:

- <code title="get /usage/recent">client.usage.<a href="./src/resources/usage.ts">listRecent</a>({ ...params }) -> UsageListRecentResponse</code>
- <code title="get /usage/stats">client.usage.<a href="./src/resources/usage.ts">retrieveStats</a>() -> UsageRetrieveStatsResponse</code>

# Local

Types:

- <code><a href="./src/resources/local.ts">LocalGenerateUriResponse</a></code>

Methods:

- <code title="post /local/generate_uri">client.local.<a href="./src/resources/local.ts">generateUri</a>({ ...params }) -> LocalGenerateUriResponse</code>

# Cloud

Types:

- <code><a href="./src/resources/cloud.ts">CloudDeleteAppResponse</a></code>
- <code><a href="./src/resources/cloud.ts">CloudGenerateUriResponse</a></code>

Methods:

- <code title="delete /cloud/apps">client.cloud.<a href="./src/resources/cloud.ts">deleteApp</a>({ ...params }) -> CloudDeleteAppResponse</code>
- <code title="post /cloud/generate_uri">client.cloud.<a href="./src/resources/cloud.ts">generateUri</a>({ ...params }) -> CloudGenerateUriResponse</code>

# Documents

Types:

- <code><a href="./src/resources/documents/documents.ts">DocumentDeleteResponse</a></code>
- <code><a href="./src/resources/documents/documents.ts">DocumentListResponse</a></code>
- <code><a href="./src/resources/documents/documents.ts">DocumentDownloadFileResponse</a></code>
- <code><a href="./src/resources/documents/documents.ts">DocumentGetDownloadURLResponse</a></code>
- <code><a href="./src/resources/documents/documents.ts">DocumentGetStatusResponse</a></code>

Methods:

- <code title="get /documents/{document_id}">client.documents.<a href="./src/resources/documents/documents.ts">retrieve</a>(documentID) -> Document</code>
- <code title="post /documents/">client.documents.<a href="./src/resources/documents/documents.ts">list</a>({ ...params }) -> DocumentListResponse</code>
- <code title="delete /documents/{document_id}">client.documents.<a href="./src/resources/documents/documents.ts">delete</a>(documentID) -> DocumentDeleteResponse</code>
- <code title="get /documents/{document_id}/file">client.documents.<a href="./src/resources/documents/documents.ts">downloadFile</a>(documentID) -> unknown</code>
- <code title="get /documents/filename/{filename}">client.documents.<a href="./src/resources/documents/documents.ts">getByFilename</a>(filename, { ...params }) -> Document</code>
- <code title="get /documents/{document_id}/download_url">client.documents.<a href="./src/resources/documents/documents.ts">getDownloadURL</a>(documentID, { ...params }) -> DocumentGetDownloadURLResponse</code>
- <code title="get /documents/{document_id}/status">client.documents.<a href="./src/resources/documents/documents.ts">getStatus</a>(documentID) -> DocumentGetStatusResponse</code>
- <code title="post /documents/{document_id}/update_file">client.documents.<a href="./src/resources/documents/documents.ts">updateFile</a>(documentID, { ...params }) -> Document</code>
- <code title="post /documents/{document_id}/update_metadata">client.documents.<a href="./src/resources/documents/documents.ts">updateMetadata</a>(documentID, { ...params }) -> Document</code>
- <code title="post /documents/{document_id}/update_text">client.documents.<a href="./src/resources/documents/documents.ts">updateText</a>(documentID, { ...params }) -> Document</code>

## Chat

Types:

- <code><a href="./src/resources/documents/chat.ts">ChatCompleteChatResponse</a></code>
- <code><a href="./src/resources/documents/chat.ts">ChatRetrieveHistoryResponse</a></code>

Methods:

- <code title="post /document/chat/{chat_id}/complete">client.documents.chat.<a href="./src/resources/documents/chat.ts">completeChat</a>(chatID, { ...params }) -> unknown</code>
- <code title="get /document/chat/{chat_id}">client.documents.chat.<a href="./src/resources/documents/chat.ts">retrieveHistory</a>(chatID) -> unknown</code>

# Chat

Types:

- <code><a href="./src/resources/chat.ts">ChatListResponse</a></code>
- <code><a href="./src/resources/chat.ts">ChatRetrieveHistoryResponse</a></code>
- <code><a href="./src/resources/chat.ts">ChatUpdateTitleResponse</a></code>

Methods:

- <code title="get /chats">client.chat.<a href="./src/resources/chat.ts">list</a>({ ...params }) -> ChatListResponse</code>
- <code title="get /chat/{chat_id}">client.chat.<a href="./src/resources/chat.ts">retrieveHistory</a>(chatID) -> ChatRetrieveHistoryResponse</code>
- <code title="patch /chats/{chat_id}/title">client.chat.<a href="./src/resources/chat.ts">updateTitle</a>(chatID, { ...params }) -> ChatUpdateTitleResponse</code>
