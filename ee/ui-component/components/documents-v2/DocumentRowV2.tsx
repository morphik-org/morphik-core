"use client";

import React from "react";
import { FileSystemItem, ProcessingStatus } from "./DocumentsProviderV2";
import { useDocumentsV2 } from "./DocumentsProviderV2";
import { ProcessingProgress } from "../types";
import { TableCell, TableRow } from "../ui/table";
import { Checkbox } from "../ui/checkbox";
import { Button } from "../ui/button";
import { Progress } from "../ui/progress";
import { toast } from "sonner";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "../ui/dropdown-menu";
import {
  Folder,
  File,
  FileText,
  FileImage,
  FileVideo,
  FileAudio,
  FileCode,
  FileArchive,
  MoreVertical,
  Eye,
  Download,
  Trash,
  Loader2,
  AlertCircle,
  CheckCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface DocumentRowV2Props {
  item: FileSystemItem;
  selected: boolean;
  onSelect: () => void;
  onClick: () => void;
  onDoubleClick: () => void;
  processingStatus?: ProcessingStatus;
}

// Helper to get file icon based on content type
function getFileIcon(contentType: string) {
  if (!contentType) return File;

  const type = contentType.toLowerCase();

  if (type.includes("image")) return FileImage;
  if (type.includes("video")) return FileVideo;
  if (type.includes("audio")) return FileAudio;
  if (type.includes("pdf")) return FileText;
  if (type.includes("zip") || type.includes("tar") || type.includes("rar")) return FileArchive;
  if (type.includes("json") || type.includes("xml") || type.includes("javascript") || type.includes("typescript"))
    return FileCode;
  if (type.includes("text")) return FileText;

  return File;
}

// Helper to format file type display
function getFileTypeDisplay(contentType: string): string {
  if (!contentType) return "File";

  const type = contentType.toLowerCase();

  if (type.includes("pdf")) return "PDF";
  if (type.includes("image")) return "Image";
  if (type.includes("video")) return "Video";
  if (type.includes("audio")) return "Audio";
  if (type.includes("zip")) return "Archive";
  if (type.includes("json")) return "JSON";
  if (type.includes("xml")) return "XML";
  if (type.includes("text/plain")) return "Text";
  if (type.includes("javascript")) return "JavaScript";
  if (type.includes("typescript")) return "TypeScript";

  // Extract extension from MIME type if possible
  const parts = type.split("/");
  if (parts.length > 1) {
    return parts[1].toUpperCase();
  }

  return "File";
}

// Helper to format relative time
function formatRelativeTime(dateString: string | undefined): string {
  if (!dateString) return "-";

  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSecs = Math.floor(diffMs / 1000);
  const diffMins = Math.floor(diffSecs / 60);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffDays > 30) {
    return date.toLocaleDateString();
  } else if (diffDays > 0) {
    return `${diffDays} day${diffDays === 1 ? "" : "s"} ago`;
  } else if (diffHours > 0) {
    return `${diffHours} hour${diffHours === 1 ? "" : "s"} ago`;
  } else if (diffMins > 0) {
    return `${diffMins} minute${diffMins === 1 ? "" : "s"} ago`;
  } else {
    return "Just now";
  }
}

export function DocumentRowV2({
  item,
  selected,
  onSelect,
  onClick,
  onDoubleClick,
  processingStatus,
}: DocumentRowV2Props) {
  const { deleteDocument, deleteFolder, onViewInPDFViewer, apiBaseUrl, authToken } = useDocumentsV2();

  // Extract display values based on item type
  const displayName = item.type === "folder" ? item.data.name : item.data.filename || item.data.external_id;

  const contentType = item.type === "folder" ? "folder" : item.data.content_type;

  const lastModified =
    item.type === "folder"
      ? (item.data.system_metadata?.updated_at as string | undefined)
      : (item.data.system_metadata?.updated_at as string | undefined);

  const status =
    item.type === "document"
      ? processingStatus || (item.data.system_metadata?.status as ProcessingStatus | undefined)
      : undefined;

  const FileIcon = item.type === "folder" ? Folder : getFileIcon(contentType);

  const handleDelete = async (e: React.MouseEvent) => {
    e.stopPropagation();

    // Use browser confirm dialog instead of AlertDialog to avoid blocking issues
    const confirmMessage =
      item.type === "folder"
        ? `Are you sure you want to delete the folder "${displayName}"? This action cannot be undone.`
        : `Are you sure you want to delete "${displayName}"? This action cannot be undone.`;

    if (!window.confirm(confirmMessage)) {
      return;
    }

    try {
      if (item.type === "document") {
        const success = await deleteDocument(item.data.external_id);
        if (success) {
          toast.success(`Document "${displayName}" deleted successfully`);
        } else {
          toast.error(`Failed to delete "${displayName}"`);
        }
      } else if (item.type === "folder") {
        const success = await deleteFolder(item.data.id);
        if (success) {
          toast.success(`Folder "${displayName}" deleted successfully`);
        } else {
          toast.error(`Failed to delete folder "${displayName}"`);
        }
      }
    } catch {
      toast.error(`Failed to delete "${displayName}"`);
    }
  };

  const handleDownload = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (item.type === "document") {
      try {
        const response = await fetch(`${apiBaseUrl}/documents/${item.data.external_id}/download`, {
          method: "GET",
          headers: {
            ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
          },
        });

        if (response.ok) {
          const blob = await response.blob();
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = item.data.filename || item.data.external_id;
          document.body.appendChild(a);
          a.click();
          window.URL.revokeObjectURL(url);
          document.body.removeChild(a);
          toast.success(`Downloaded "${displayName}"`);
        } else {
          toast.error(`Failed to download "${displayName}"`);
        }
      } catch (error) {
        console.error("Download error:", error);
        toast.error(`Failed to download "${displayName}"`);
      }
    }
  };

  const handlePreview = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (item.type === "document" && onViewInPDFViewer) {
      onViewInPDFViewer(item.data.external_id);
    }
  };

  return (
    <TableRow
      className={cn("cursor-pointer hover:bg-muted/50", selected && "bg-muted")}
      onClick={onClick}
      onDoubleClick={onDoubleClick}
    >
      <TableCell className="w-12">
        <Checkbox
          checked={selected}
          onCheckedChange={onSelect}
          onClick={e => e.stopPropagation()}
          aria-label={`Select ${displayName}`}
        />
      </TableCell>

      <TableCell>
        <div className="flex items-center gap-2">
          <FileIcon
            className={cn("h-5 w-5 flex-shrink-0", item.type === "folder" ? "text-blue-500" : "text-muted-foreground")}
          />
          <span className="truncate font-medium" title={displayName}>
            {displayName}
          </span>
        </div>
      </TableCell>

      <TableCell className="w-32 text-sm text-muted-foreground">
        {item.type === "folder" ? "Folder" : getFileTypeDisplay(contentType)}
      </TableCell>

      <TableCell className="w-40 text-sm text-muted-foreground">{formatRelativeTime(lastModified)}</TableCell>

      <TableCell className="w-32">
        {status === "processing" && (
          <div className="flex items-center gap-2">
            <Loader2 className="h-3 w-3 animate-spin" />
            {item.type === "document" && (item.data.system_metadata?.progress as ProcessingProgress | undefined) ? (
              <Progress
                value={(item.data.system_metadata.progress as ProcessingProgress).percentage || 0}
                className="h-1.5 w-16"
              />
            ) : (
              <span className="text-xs">Processing...</span>
            )}
          </div>
        )}
        {status === "failed" && (
          <div className="flex items-center gap-2 text-destructive">
            <AlertCircle className="h-3 w-3" />
            <span className="text-xs">Failed</span>
          </div>
        )}
        {status === "completed" && (
          <div className="flex items-center gap-2 text-green-600">
            <CheckCircle className="h-3 w-3" />
            <span className="text-xs">Ready</span>
          </div>
        )}
        {!status && item.type === "document" && (
          <div className="flex items-center gap-2 text-muted-foreground">
            <CheckCircle className="h-3 w-3" />
            <span className="text-xs">Ready</span>
          </div>
        )}
      </TableCell>

      <TableCell className="w-20 text-right">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" className="h-8 w-8" onClick={e => e.stopPropagation()}>
              <MoreVertical className="h-4 w-4" />
              <span className="sr-only">Open menu</span>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            {item.type === "document" && (
              <>
                <DropdownMenuItem onClick={handlePreview}>
                  <Eye className="mr-2 h-4 w-4" />
                  Preview
                </DropdownMenuItem>
                <DropdownMenuItem onClick={handleDownload}>
                  <Download className="mr-2 h-4 w-4" />
                  Download
                </DropdownMenuItem>
                <DropdownMenuSeparator />
              </>
            )}
            <DropdownMenuItem className="text-destructive focus:text-destructive" onClick={handleDelete}>
              <Trash className="mr-2 h-4 w-4" />
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </TableCell>
    </TableRow>
  );
}
