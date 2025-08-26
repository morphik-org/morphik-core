import { Document, Folder } from "../../types";

export interface DocumentsApiConfig {
  apiBaseUrl: string;
  authToken: string | null;
}

export interface FolderSummary extends Folder {
  id: string;
  name: string;
  document_count?: number;
}

export interface DocumentsResponse {
  documents: Document[];
  total?: number;
}

export interface BatchDocumentsRequest {
  document_ids: string[];
}

export interface CreateFolderRequest {
  name: string;
  parent_folder_id?: string;
}

export interface UploadFilesRequest {
  files: File[];
  folder_name?: string;
  metadata?: Record<string, any>;
}

export class DocumentsApi {
  private config: DocumentsApiConfig;

  constructor(config: DocumentsApiConfig) {
    this.config = config;
  }

  private get headers() {
    return {
      ...(this.config.authToken ? { Authorization: `Bearer ${this.config.authToken}` } : {}),
      "Content-Type": "application/json",
    };
  }

  private async parseDocumentResponse(response: Response): Promise<Document[]> {
    const data = await response.json();
    return Array.isArray(data) ? data : data.documents || data.data || [];
  }

  async fetchFolders(): Promise<FolderSummary[]> {
    try {
      const response = await fetch(`${this.config.apiBaseUrl}/folders/summary`, {
        method: "GET",
        headers: this.headers,
      });

      if (response.ok) {
        const data = await response.json();
        return Array.isArray(data) ? data : data.folders || [];
      }
      return [];
    } catch (error) {
      console.error("[DocumentsApi] Error fetching folders:", error);
      return [];
    }
  }

  async fetchAllDocuments(): Promise<Document[]> {
    try {
      const response = await fetch(`${this.config.apiBaseUrl}/documents`, {
        method: "POST",
        headers: this.headers,
        body: JSON.stringify({}),
      });

      if (response.ok) {
        return await this.parseDocumentResponse(response);
      }
      return [];
    } catch (error) {
      console.error("[DocumentsApi] Error fetching all documents:", error);
      return [];
    }
  }

  async fetchUnorganizedDocuments(): Promise<Document[]> {
    try {
      const allDocs = await this.fetchAllDocuments();
      return allDocs.filter(doc => !doc.folder_name || doc.folder_name.trim() === "");
    } catch (error) {
      console.error("[DocumentsApi] Error fetching unorganized documents:", error);
      return [];
    }
  }

  async fetchFolderDetail(folderId: string): Promise<{ folder: Folder; document_ids: string[] } | null> {
    try {
      const response = await fetch(`${this.config.apiBaseUrl}/folders/${folderId}`, {
        method: "GET",
        headers: this.headers,
      });

      if (response.ok) {
        const data = await response.json();
        return {
          folder: data,
          document_ids: Array.isArray(data.document_ids) ? data.document_ids : [],
        };
      }
      return null;
    } catch (error) {
      console.error(`[DocumentsApi] Error fetching folder detail ${folderId}:`, error);
      return null;
    }
  }

  async fetchDocumentsByIds(documentIds: string[]): Promise<Document[]> {
    if (documentIds.length === 0) return [];

    try {
      const response = await fetch(`${this.config.apiBaseUrl}/batch/documents`, {
        method: "POST",
        headers: this.headers,
        body: JSON.stringify({ document_ids: documentIds }),
      });

      if (response.ok) {
        return await this.parseDocumentResponse(response);
      }
      return [];
    } catch (error) {
      console.error("[DocumentsApi] Error fetching documents by IDs:", error);
      return [];
    }
  }

  async fetchDocumentsInFolder(folderName: string): Promise<Document[]> {
    // Fetch all documents and filter by folder_name
    // This ensures we get recently uploaded documents that have folder_name set
    // but might not be in the folder's document_ids list yet (as per backend comment:
    // "Folder assignment is handled in the background worker to avoid race conditions")
    try {
      const allDocs = await this.fetchAllDocuments();
      return allDocs.filter(doc => doc.folder_name === folderName);
    } catch (error) {
      console.error(`[DocumentsApi] Error fetching documents in folder ${folderName}:`, error);
      return [];
    }
  }

  async fetchDocumentDetail(documentId: string): Promise<Document | null> {
    try {
      const response = await fetch(`${this.config.apiBaseUrl}/documents/${documentId}`, {
        method: "GET",
        headers: this.headers,
      });

      if (response.ok) {
        return await response.json();
      }
      return null;
    } catch (error) {
      console.error(`[DocumentsApi] Error fetching document detail ${documentId}:`, error);
      return null;
    }
  }

  async deleteDocument(documentId: string): Promise<boolean> {
    try {
      const response = await fetch(`${this.config.apiBaseUrl}/documents/${documentId}`, {
        method: "DELETE",
        headers: this.headers,
      });

      return response.ok;
    } catch (error) {
      console.error(`[DocumentsApi] Error deleting document ${documentId}:`, error);
      return false;
    }
  }

  async deleteFolder(folderId: string): Promise<boolean> {
    try {
      const response = await fetch(`${this.config.apiBaseUrl}/folders/${folderId}`, {
        method: "DELETE",
        headers: this.headers,
      });

      return response.ok;
    } catch (error) {
      console.error(`[DocumentsApi] Error deleting folder ${folderId}:`, error);
      return false;
    }
  }

  async createFolder(name: string, parentFolderId?: string): Promise<Folder | null> {
    try {
      const body: CreateFolderRequest = { name };
      if (parentFolderId) {
        body.parent_folder_id = parentFolderId;
      }

      const response = await fetch(`${this.config.apiBaseUrl}/folders`, {
        method: "POST",
        headers: this.headers,
        body: JSON.stringify(body),
      });

      if (response.ok) {
        const data = await response.json();
        return data;
      } else {
        const errorText = await response.text();
        console.error("[DocumentsApi] Failed to create folder:", response.status, errorText);
      }
      return null;
    } catch (error) {
      console.error("[DocumentsApi] Error creating folder:", error);
      return null;
    }
  }

  async uploadFiles(files: File[], folderName?: string, metadata?: Record<string, any>): Promise<boolean> {
    try {
      const formData = new FormData();

      files.forEach(file => {
        formData.append("files", file);
      });

      if (folderName) {
        formData.append("folder_name", folderName);
      }

      if (metadata) {
        formData.append("metadata", JSON.stringify(metadata));
      }

      const response = await fetch(`${this.config.apiBaseUrl}/ingest/files`, {
        method: "POST",
        headers: {
          ...(this.config.authToken ? { Authorization: `Bearer ${this.config.authToken}` } : {}),
        },
        body: formData,
      });

      return response.ok;
    } catch (error) {
      console.error("[DocumentsApi] Error uploading files:", error);
      return false;
    }
  }

  async downloadDocument(documentId: string): Promise<Blob | null> {
    try {
      const response = await fetch(`${this.config.apiBaseUrl}/documents/${documentId}/download`, {
        method: "GET",
        headers: {
          ...(this.config.authToken ? { Authorization: `Bearer ${this.config.authToken}` } : {}),
        },
      });

      if (response.ok) {
        return await response.blob();
      }
      return null;
    } catch (error) {
      console.error(`[DocumentsApi] Error downloading document ${documentId}:`, error);
      return null;
    }
  }

  updateConfig(config: Partial<DocumentsApiConfig>) {
    this.config = { ...this.config, ...config };
  }
}
