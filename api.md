# Ping

Types:

- <code><a href="./src/resources/ping.ts">PingCheckResponse</a></code>

Methods:

- <code title="get /ping">client.ping.<a href="./src/resources/ping.ts">check</a>() -> unknown</code>

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

- <code><a href="./src/resources/ee/connectors/auth.ts">AuthGetInitiateAuthURLResponse</a></code>

Methods:

- <code title="get /ee/connectors/{connector_type}/auth/initiate_url">client.ee.connectors.auth.<a href="./src/resources/ee/connectors/auth.ts">getInitiateAuthURL</a>(connectorType, { ...params }) -> AuthGetInitiateAuthURLResponse</code>

# Ingest

Types:

- <code><a href="./src/resources/ingest.ts">Document</a></code>
- <code><a href="./src/resources/ingest.ts">IngestText</a></code>
- <code><a href="./src/resources/ingest.ts">IngestBatchIngestFilesResponse</a></code>

Methods:

- <code title="post /ingest/files">client.ingest.<a href="./src/resources/ingest.ts">batchIngestFiles</a>({ ...params }) -> IngestBatchIngestFilesResponse</code>
- <code title="post /ingest/file">client.ingest.<a href="./src/resources/ingest.ts">ingestFile</a>({ ...params }) -> Document</code>
- <code title="post /ingest/text">client.ingest.<a href="./src/resources/ingest.ts">ingestText</a>({ ...params }) -> Document</code>

# Retrieve

Types:

- <code><a href="./src/resources/retrieve.ts">ChunkResult</a></code>
- <code><a href="./src/resources/retrieve.ts">RetrieveRequest</a></code>
- <code><a href="./src/resources/retrieve.ts">RetrieveRetrieveChunksResponse</a></code>
- <code><a href="./src/resources/retrieve.ts">RetrieveRetrieveDocsResponse</a></code>

Methods:

- <code title="post /retrieve/chunks">client.retrieve.<a href="./src/resources/retrieve.ts">retrieveChunks</a>({ ...params }) -> RetrieveRetrieveChunksResponse</code>
- <code title="post /retrieve/docs">client.retrieve.<a href="./src/resources/retrieve.ts">retrieveDocs</a>({ ...params }) -> RetrieveRetrieveDocsResponse</code>

# Batch

Types:

- <code><a href="./src/resources/batch.ts">BatchRetrieveChunksResponse</a></code>
- <code><a href="./src/resources/batch.ts">BatchRetrieveDocumentsResponse</a></code>

Methods:

- <code title="post /batch/chunks">client.batch.<a href="./src/resources/batch.ts">retrieveChunks</a>({ ...params }) -> BatchRetrieveChunksResponse</code>
- <code title="post /batch/documents">client.batch.<a href="./src/resources/batch.ts">retrieveDocuments</a>({ ...params }) -> BatchRetrieveDocumentsResponse</code>

# Query

Types:

- <code><a href="./src/resources/query.ts">CompletionResponse</a></code>
- <code><a href="./src/resources/query.ts">EntityExtractionPromptOverride</a></code>
- <code><a href="./src/resources/query.ts">EntityResolutionPromptOverride</a></code>

Methods:

- <code title="post /query">client.query.<a href="./src/resources/query.ts">generateCompletion</a>({ ...params }) -> CompletionResponse</code>

# Agent

Types:

- <code><a href="./src/resources/agent.ts">AgentProcessQueryResponse</a></code>

Methods:

- <code title="post /agent">client.agent.<a href="./src/resources/agent.ts">processQuery</a>({ ...params }) -> AgentProcessQueryResponse</code>

# Documents

Types:

- <code><a href="./src/resources/documents.ts">DocumentListResponse</a></code>
- <code><a href="./src/resources/documents.ts">DocumentDeleteResponse</a></code>
- <code><a href="./src/resources/documents.ts">DocumentGetStatusResponse</a></code>

Methods:

- <code title="get /documents/{document_id}">client.documents.<a href="./src/resources/documents.ts">retrieve</a>(documentID) -> Document</code>
- <code title="post /documents">client.documents.<a href="./src/resources/documents.ts">list</a>({ ...params }) -> DocumentListResponse</code>
- <code title="delete /documents/{document_id}">client.documents.<a href="./src/resources/documents.ts">delete</a>(documentID) -> unknown</code>
- <code title="get /documents/{document_id}/status">client.documents.<a href="./src/resources/documents.ts">getStatus</a>(documentID) -> DocumentGetStatusResponse</code>
- <code title="get /documents/filename/{filename}">client.documents.<a href="./src/resources/documents.ts">retrieveByFilename</a>(filename, { ...params }) -> Document</code>
- <code title="post /documents/{document_id}/update_file">client.documents.<a href="./src/resources/documents.ts">updateFile</a>(documentID, { ...params }) -> Document</code>
- <code title="post /documents/{document_id}/update_metadata">client.documents.<a href="./src/resources/documents.ts">updateMetadata</a>(documentID, { ...params }) -> Document</code>
- <code title="post /documents/{document_id}/update_text">client.documents.<a href="./src/resources/documents.ts">updateText</a>(documentID, { ...params }) -> Document</code>

# Usage

Types:

- <code><a href="./src/resources/usage.ts">UsageListRecentResponse</a></code>
- <code><a href="./src/resources/usage.ts">UsageRetrieveStatsResponse</a></code>

Methods:

- <code title="get /usage/recent">client.usage.<a href="./src/resources/usage.ts">listRecent</a>({ ...params }) -> UsageListRecentResponse</code>
- <code title="get /usage/stats">client.usage.<a href="./src/resources/usage.ts">retrieveStats</a>() -> UsageRetrieveStatsResponse</code>

# Cache

Types:

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

# Folders

Types:

- <code><a href="./src/resources/folders/folders.ts">Folder</a></code>
- <code><a href="./src/resources/folders/folders.ts">FolderListResponse</a></code>
- <code><a href="./src/resources/folders/folders.ts">FolderDeleteResponse</a></code>
- <code><a href="./src/resources/folders/folders.ts">FolderSetRuleResponse</a></code>

Methods:

- <code title="post /folders">client.folders.<a href="./src/resources/folders/folders.ts">create</a>({ ...params }) -> Folder</code>
- <code title="get /folders/{folder_id}">client.folders.<a href="./src/resources/folders/folders.ts">retrieve</a>(folderID) -> Folder</code>
- <code title="get /folders">client.folders.<a href="./src/resources/folders/folders.ts">list</a>() -> FolderListResponse</code>
- <code title="delete /folders/{folder_name}">client.folders.<a href="./src/resources/folders/folders.ts">delete</a>(folderName) -> unknown</code>
- <code title="post /folders/{folder_id}/set_rule">client.folders.<a href="./src/resources/folders/folders.ts">setRule</a>(folderID, { ...params }) -> unknown</code>

## Documents

Types:

- <code><a href="./src/resources/folders/documents.ts">DocumentAddResponse</a></code>
- <code><a href="./src/resources/folders/documents.ts">DocumentRemoveResponse</a></code>

Methods:

- <code title="post /folders/{folder_id}/documents/{document_id}">client.folders.documents.<a href="./src/resources/folders/documents.ts">add</a>(documentID, { ...params }) -> unknown</code>
- <code title="delete /folders/{folder_id}/documents/{document_id}">client.folders.documents.<a href="./src/resources/folders/documents.ts">remove</a>(documentID, { ...params }) -> unknown</code>

# Local

Types:

- <code><a href="./src/resources/local.ts">LocalGenerateUriResponse</a></code>

Methods:

- <code title="post /local/generate_uri">client.local.<a href="./src/resources/local.ts">generateUri</a>({ ...params }) -> LocalGenerateUriResponse</code>

# Cloud

Types:

- <code><a href="./src/resources/cloud.ts">CloudDeleteAppsResponse</a></code>
- <code><a href="./src/resources/cloud.ts">CloudGenerateUriResponse</a></code>

Methods:

- <code title="delete /cloud/apps">client.cloud.<a href="./src/resources/cloud.ts">deleteApps</a>({ ...params }) -> CloudDeleteAppsResponse</code>
- <code title="post /cloud/generate_uri">client.cloud.<a href="./src/resources/cloud.ts">generateUri</a>({ ...params }) -> CloudGenerateUriResponse</code>

# Graphs

Types:

- <code><a href="./src/resources/graphs.ts">Graph</a></code>
- <code><a href="./src/resources/graphs.ts">GraphPromptOverrides</a></code>
- <code><a href="./src/resources/graphs.ts">GraphListResponse</a></code>

Methods:

- <code title="post /graph/create">client.graphs.<a href="./src/resources/graphs.ts">create</a>({ ...params }) -> Graph</code>
- <code title="get /graph/{name}">client.graphs.<a href="./src/resources/graphs.ts">retrieve</a>(name, { ...params }) -> Graph</code>
- <code title="post /graph/{name}/update">client.graphs.<a href="./src/resources/graphs.ts">update</a>(name, { ...params }) -> Graph</code>
- <code title="get /graphs">client.graphs.<a href="./src/resources/graphs.ts">list</a>({ ...params }) -> GraphListResponse</code>
