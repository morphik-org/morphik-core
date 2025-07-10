import React, { useEffect, useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import {
  Plus,
  RefreshCcw,
  Edit2,
  Play,
  FileText,
  Copy,
  CheckCircle,
  Trash2,
  ArrowLeft,
  Clock,
  Loader2,
  AlertCircle,
  ChevronRight,
  Layers,
  FileJson,
  Info,
  Eye,
  EyeOff,
  FolderPlus,
  Folder,
} from "lucide-react";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { WorkflowCreateDialog } from "./WorkflowCreateDialog";
import { WorkflowEditDialog } from "./WorkflowEditDialog";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useHeader } from "@/contexts/header-context";

interface WorkflowSectionProps {
  apiBaseUrl: string;
  authToken?: string | null;
}

interface Workflow {
  id: string;
  name: string;
  description?: string;
  steps: ConfiguredAction[];
}

interface ConfiguredAction {
  action_id: string;
  parameters: {
    schema?: SchemaField[] | { type: string; properties: Record<string, unknown>; required?: string[] };
    prompt_template?: string;
    metadata_key?: string;
    source?: string;
    [key: string]: unknown;
  };
}

interface SchemaField {
  name: string;
  type: "string" | "number" | "boolean" | "array" | "object";
  description?: string;
  required?: boolean;
}

interface DocumentMeta {
  external_id: string;
  filename?: string;
  content_type: string;
}

interface WorkflowRun {
  id: string;
  workflow_id: string;
  document_id: string;
  status: string;
  final_output?: unknown;
  error?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  results_per_step?: unknown[];
}

interface FolderSummary {
  id: string;
  name: string;
  document_count: number;
  updated_at: string;
}

interface FolderWithWorkflows extends FolderSummary {
  workflow_ids: string[];
}

interface ActionDefinition {
  id: string;
  name: string;
  description: string;
  parameters_schema: Record<string, unknown>;
  output_schema: Record<string, unknown>;
}

// Available actions - in production this would come from API
const AVAILABLE_ACTIONS: ActionDefinition[] = [
  {
    id: "morphik.actions.extract_structured",
    name: "Extract Structured Data",
    description: "Extract structured JSON data from documents using AI",
    parameters_schema: {
      type: "object",
      properties: {
        schema: { type: "object", description: "JSON Schema for extraction" },
      },
    },
    output_schema: { type: "object" },
  },
  {
    id: "morphik.actions.apply_instruction",
    name: "Apply Custom Instruction",
    description: "Apply a custom AI instruction to transform the document",
    parameters_schema: {
      type: "object",
      properties: {
        prompt_template: { type: "string", description: "Instruction template with {input_text} placeholder" },
      },
    },
    output_schema: { type: "string" },
  },
  {
    id: "morphik.actions.convert_to_markdown",
    name: "Convert to Markdown",
    description: "Convert documents to markdown",
    parameters_schema: {
      type: "object",
      properties: {
        api_key_env: {
          type: "string",
          description: "Environment variable name containing the Gemini API key",
          default: "GEMINI_API_KEY",
        },
        model: {
          type: "string",
          description: "Gemini model to use for conversion",
          default: "gemini-2.5-pro",
        },
        temperature: {
          type: "number",
          description: "Temperature for generation (0-1)",
          default: 0,
        },
        custom_prompt: {
          type: "string",
          description: "Optional custom prompt to append to the conversion request",
          default: "",
        },
      },
      required: [],
    },
    output_schema: {
      type: "object",
      properties: {
        markdown: { type: "string", description: "The converted markdown content" },
        original_filename: { type: "string" },
        mime_type: { type: "string" },
        model_used: { type: "string" },
      },
    },
  },
  {
    id: "morphik.actions.ingest_output",
    name: "Ingest Output",
    description: "Ingest workflow output as a new document",
    parameters_schema: {
      type: "object",
      properties: {
        filename: {
          type: "string",
          description: "Filename for the ingested document",
          default: "workflow_output.md",
        },
        source: {
          type: "string",
          enum: ["previous_step", "all_steps"],
          description: "Source of content to ingest",
          default: "previous_step",
        },
        content_field: {
          type: "string",
          description: "Field name containing the content to ingest",
          default: "markdown",
        },
        metadata: {
          type: "object",
          description: "Additional metadata to attach to the document",
          default: {},
        },
      },
      required: [],
    },
    output_schema: {
      type: "object",
      properties: {
        document_id: { type: "string", description: "ID of the ingested document" },
        filename: { type: "string" },
        status: { type: "string" },
      },
    },
  },
  {
    id: "morphik.actions.save_to_metadata",
    name: "Save to Metadata",
    description: "Save the output from previous step to document metadata",
    parameters_schema: {
      type: "object",
      properties: {
        metadata_key: { type: "string", description: "Key to store the data under" },
        source: {
          type: "string",
          enum: ["previous_step", "all_steps"],
          description: "Whether to save output from previous step or all steps",
          default: "previous_step",
        },
      },
    },
    output_schema: { type: "object" },
  },
];

const WorkflowSection: React.FC<WorkflowSectionProps> = ({ apiBaseUrl, authToken }) => {
  const { setCustomBreadcrumbs, setRightContent } = useHeader();
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [docs, setDocs] = useState<DocumentMeta[]>([]);
  const [selectedWorkflow, setSelectedWorkflow] = useState<Workflow | null>(null);
  const [workflowRuns, setWorkflowRuns] = useState<WorkflowRun[]>([]);
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [isEditOpen, setIsEditOpen] = useState(false);
  const [isRunOpen, setIsRunOpen] = useState(false);
  const [, setSelectedStepIndex] = useState<number | null>(null);
  const [runningWorkflow, setRunningWorkflow] = useState<Workflow | null>(null);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [copiedField, setCopiedField] = useState<string>("");
  const [isRunning, setIsRunning] = useState(false);
  const [runProgress, setRunProgress] = useState<{ [key: string]: number }>({});
  const [pollingIntervals, setPollingIntervals] = useState<{ [key: string]: NodeJS.Timeout }>({});
  const [expandedRuns, setExpandedRuns] = useState<Set<string>>(new Set());
  const [folders, setFolders] = useState<FolderSummary[]>([]);
  const [isFolderDialogOpen, setIsFolderDialogOpen] = useState(false);
  const [selectedWorkflowForFolder, setSelectedWorkflowForFolder] = useState<Workflow | null>(null);
  const [workflowFolders, setWorkflowFolders] = useState<{ [key: string]: string[] }>({});
  const [isWorkflowStepsExpanded, setIsWorkflowStepsExpanded] = useState(true);

  // Form states for workflow creation/editing
  const [workflowForm, setWorkflowForm] = useState<{
    name: string;
    description: string;
    steps: ConfiguredAction[];
  }>({
    name: "",
    description: "",
    steps: [],
  });

  const headers = React.useMemo(() => {
    const h: HeadersInit = { "Content-Type": "application/json" };
    if (authToken) h["Authorization"] = `Bearer ${authToken}`;
    return h;
  }, [authToken]);

  // Handle URL query parameter for workflow ID
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const workflowId = urlParams.get("id");
    if (workflowId && workflows.length > 0) {
      const workflow = workflows.find(w => w.id === workflowId);
      if (workflow) {
        setSelectedWorkflow(workflow);
      }
    }
  }, [workflows]);

  const fetchWorkflows = useCallback(async () => {
    try {
      const res = await fetch(`${apiBaseUrl}/workflows`, { headers });
      if (res.ok) {
        const data: Workflow[] = await res.json();
        setWorkflows(data);
      }
    } catch (error) {
      console.error("Failed to fetch workflows:", error);
    }
  }, [apiBaseUrl, headers]);

  const fetchDocs = async () => {
    try {
      const res = await fetch(`${apiBaseUrl}/documents`, {
        method: "POST",
        headers,
        body: JSON.stringify({}),
      });
      if (res.ok) {
        const data = await res.json();
        setDocs(Array.isArray(data) ? data : []);
      }
    } catch (error) {
      console.error("Failed to fetch documents:", error);
    }
  };

  const fetchWorkflowRuns = async (workflowId: string) => {
    try {
      const res = await fetch(`${apiBaseUrl}/workflows/${workflowId}/runs`, { headers });
      if (res.ok) {
        const data: WorkflowRun[] = await res.json();
        setWorkflowRuns(data);
      }
    } catch (error) {
      console.error("Failed to fetch workflow runs:", error);
      setWorkflowRuns([]);
    }
  };

  const fetchFolders = async () => {
    try {
      const res = await fetch(`${apiBaseUrl}/folders/summary`, { headers });
      if (res.ok) {
        const data = await res.json();
        setFolders(data);
      }
    } catch (error) {
      console.error("Failed to fetch folders:", error);
    }
  };

  const fetchWorkflowFolders = useCallback(
    async (workflowId: string) => {
      try {
        // Get all folders and check which ones have this workflow
        const res = await fetch(`${apiBaseUrl}/folders`, { headers });
        if (res.ok) {
          const allFolders = await res.json();
          const associatedFolderIds = allFolders
            .filter((folder: FolderWithWorkflows) => folder.workflow_ids?.includes(workflowId))
            .map((folder: FolderWithWorkflows) => folder.id);
          setWorkflowFolders(prev => ({ ...prev, [workflowId]: associatedFolderIds }));
        }
      } catch (error) {
        console.error("Failed to fetch workflow folders:", error);
      }
    },
    [apiBaseUrl, headers]
  );

  const toggleFolderWorkflow = async (folderId: string, workflowId: string, isAssociated: boolean) => {
    try {
      const method = isAssociated ? "DELETE" : "POST";
      const res = await fetch(`${apiBaseUrl}/folders/${folderId}/workflows/${workflowId}`, {
        method,
        headers,
      });
      if (res.ok) {
        // Refresh workflow folders
        await fetchWorkflowFolders(workflowId);
      }
    } catch (error) {
      console.error("Failed to toggle folder workflow association:", error);
    }
  };

  const pollRunStatus = useCallback(
    async (runId: string) => {
      try {
        const res = await fetch(`${apiBaseUrl}/workflows/runs/${runId}`, { headers });
        if (res.ok) {
          const run = await res.json();
          setWorkflowRuns(prev => prev.map(r => (r.id === runId ? run : r)));

          // Update progress based on status
          if (run.status === "running") {
            setRunProgress(prev => ({ ...prev, [runId]: 50 }));
          } else if (run.status === "completed") {
            setRunProgress(prev => ({ ...prev, [runId]: 100 }));
            // Stop polling
            if (pollingIntervals[runId]) {
              clearInterval(pollingIntervals[runId]);
              setPollingIntervals(prev => {
                const newIntervals = { ...prev };
                delete newIntervals[runId];
                return newIntervals;
              });
            }
          } else if (run.status === "failed") {
            setRunProgress(prev => ({ ...prev, [runId]: 0 }));
            // Stop polling
            if (pollingIntervals[runId]) {
              clearInterval(pollingIntervals[runId]);
              setPollingIntervals(prev => {
                const newIntervals = { ...prev };
                delete newIntervals[runId];
                return newIntervals;
              });
            }
          }
        } else if (res.status === 404) {
          // Run no longer exists – stop polling and remove it from state
          if (pollingIntervals[runId]) {
            clearInterval(pollingIntervals[runId]);
            setPollingIntervals(prev => {
              const newIntervals = { ...prev };
              delete newIntervals[runId];
              return newIntervals;
            });
          }
          setWorkflowRuns(prev => prev.filter(r => r.id !== runId));
        }
      } catch (error) {
        console.error("Failed to poll run status:", error);
      }
    },
    [apiBaseUrl, headers, pollingIntervals]
  );

  useEffect(() => {
    fetchWorkflows();
    fetchDocs();
    fetchFolders();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Header breadcrumbs & controls
  useEffect(() => {
    setCustomBreadcrumbs([{ label: "Home", href: "/" }, { label: "Workflows" }]);

    const right = (
      <div className="flex items-center gap-2">
        <Button variant="outline" size="sm" onClick={() => fetchWorkflows()} title="Refresh workflows">
          <RefreshCcw className="mr-2 h-4 w-4" /> Refresh
        </Button>
        <Button
          variant="default"
          size="sm"
          onClick={() => {
            setWorkflowForm({ name: "", description: "", steps: [] });
            setSelectedStepIndex(null);
            setIsCreateOpen(true);
          }}
        >
          <Plus className="mr-2 h-4 w-4" /> New Workflow
        </Button>
      </div>
    );

    setRightContent(right);

    return () => {
      setCustomBreadcrumbs(null);
      setRightContent(null);
    };
  }, [setCustomBreadcrumbs, setRightContent, fetchWorkflows]);

  useEffect(() => {
    if (selectedWorkflow) {
      fetchWorkflowRuns(selectedWorkflow.id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedWorkflow]);

  // Cleanup polling intervals on unmount
  useEffect(() => {
    return () => {
      Object.values(pollingIntervals).forEach(interval => clearInterval(interval));
    };
  }, [pollingIntervals]);

  const createWorkflow = async () => {
    if (!workflowForm.name.trim() || workflowForm.steps.length === 0) return;

    const newWorkflow = {
      ...workflowForm,
      owner_id: "local",
    };

    try {
      const res = await fetch(`${apiBaseUrl}/workflows`, {
        method: "POST",
        headers,
        body: JSON.stringify(newWorkflow),
      });

      if (res.ok) {
        setIsCreateOpen(false);
        setWorkflowForm({ name: "", description: "", steps: [] });
        fetchWorkflows();
      }
    } catch (error) {
      console.error("Failed to create workflow:", error);
    }
  };

  const updateWorkflow = async () => {
    if (!selectedWorkflow || !workflowForm.name.trim()) return;

    try {
      const res = await fetch(`${apiBaseUrl}/workflows/${selectedWorkflow.id}`, {
        method: "PUT",
        headers,
        body: JSON.stringify({
          name: workflowForm.name,
          description: workflowForm.description,
          steps: workflowForm.steps,
        }),
      });

      if (res.ok) {
        setIsEditOpen(false);
        fetchWorkflows();
        const updated = await res.json();
        setSelectedWorkflow(updated);
      }
    } catch (error) {
      console.error("Failed to update workflow:", error);
    }
  };

  const runWorkflow = async () => {
    if (!runningWorkflow || selectedDocIds.length === 0) return;

    setIsRunning(true);
    try {
      const runPromises = selectedDocIds.map(async docId => {
        const res = await fetch(`${apiBaseUrl}/workflows/${runningWorkflow.id}/run/${docId}`, {
          method: "POST",
          headers,
        });
        if (res.ok) {
          const run = await res.json();
          // Start polling for this run
          setRunProgress(prev => ({ ...prev, [run.id]: 10 }));
          const interval = setInterval(() => pollRunStatus(run.id), 2000);
          setPollingIntervals(prev => ({ ...prev, [run.id]: interval }));
          return run;
        }
        return null;
      });

      const runs = await Promise.all(runPromises);
      const successfulRuns = runs.filter(r => r !== null);

      if (successfulRuns.length > 0) {
        setIsRunOpen(false);
        setSelectedWorkflow(runningWorkflow);
        await fetchWorkflowRuns(runningWorkflow.id);
        setSelectedDocIds([]);
      }
    } catch (error) {
      console.error("Failed to run workflow:", error);
    } finally {
      setIsRunning(false);
    }
  };

  const copyToClipboard = async (text: string, fieldName: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedField(fieldName);
      setTimeout(() => setCopiedField(""), 2000);
    } catch (error) {
      console.error("Failed to copy:", error);
    }
  };

  const renderExtractedData = (output: unknown, isExpanded: boolean = true, isMarkdown: boolean = false) => {
    if (!output) return null;

    // Handle different output types
    if (typeof output === "string") {
      // If it's markdown content, render it as markdown
      if (isMarkdown) {
        return (
          <div className="prose prose-sm dark:prose-invert max-w-none space-y-2 rounded-lg bg-muted/30 p-4">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                p: ({ children }) => <p className="mb-2 text-sm">{children}</p>,
                h1: ({ children }) => <h1 className="mb-3 text-xl font-bold">{children}</h1>,
                h2: ({ children }) => <h2 className="mb-2 text-lg font-semibold">{children}</h2>,
                h3: ({ children }) => <h3 className="mb-2 text-base font-medium">{children}</h3>,
                ul: ({ children }) => <ul className="mb-2 list-disc pl-5 text-sm">{children}</ul>,
                ol: ({ children }) => <ol className="mb-2 list-decimal pl-5 text-sm">{children}</ol>,
                li: ({ children }) => <li className="mb-1">{children}</li>,
                a: ({ href, children }) => (
                  <a
                    href={href}
                    className="text-primary underline hover:no-underline"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    {children}
                  </a>
                ),
                blockquote: ({ children }) => (
                  <blockquote className="my-2 border-l-4 border-muted-foreground/50 pl-4 italic">{children}</blockquote>
                ),
                code: ({ inline, children }: { inline?: boolean; children?: React.ReactNode }) =>
                  inline ? (
                    <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">{children}</code>
                  ) : (
                    <code className="block rounded bg-muted p-2 font-mono text-xs">{children}</code>
                  ),
                pre: ({ children }) => <pre className="overflow-x-auto">{children}</pre>,
              }}
            >
              {output}
            </ReactMarkdown>
          </div>
        );
      }

      return (
        <div className="space-y-2">
          <Textarea value={output} readOnly className="min-h-[100px] resize-none bg-muted/30 font-mono text-sm" />
        </div>
      );
    }

    // For objects, render each field
    const renderValue = (key: string, value: unknown, depth: number = 0) => {
      const stringValue = typeof value === "object" ? JSON.stringify(value, null, 2) : String(value);
      const isCopied = copiedField === `${key}-${depth}`;
      const isObject = typeof value === "object" && value !== null;
      const isMarkdownField = key === "markdown" && typeof value === "string" && isMarkdown;

      return (
        <div key={`${key}-${depth}`} className={`space-y-2 ${depth > 0 ? "ml-4 border-l-2 border-gray-200 pl-4" : ""}`}>
          <div className="flex items-center justify-between">
            <Label className="text-sm font-medium capitalize">{key.replace(/_/g, " ")}</Label>
            <div className="flex items-center gap-1">
              {isObject && (
                <Badge variant="outline" className="text-xs">
                  {Array.isArray(value) ? `Array[${value.length}]` : "Object"}
                </Badge>
              )}
              <Button
                variant="ghost"
                size="sm"
                onClick={() => copyToClipboard(stringValue, `${key}-${depth}`)}
                className="h-6 w-6 p-0"
              >
                {isCopied ? (
                  <CheckCircle className="h-3 w-3 text-green-600 dark:text-green-400" />
                ) : (
                  <Copy className="h-3 w-3 text-muted-foreground" />
                )}
              </Button>
            </div>
          </div>
          {isExpanded && (
            <div className="relative">
              {isMarkdownField ? (
                <div className="prose prose-sm dark:prose-invert mt-2 max-w-none rounded-lg bg-muted/30 p-4">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      p: ({ children }) => <p className="mb-2 text-sm">{children}</p>,
                      h1: ({ children }) => <h1 className="mb-3 text-xl font-bold">{children}</h1>,
                      h2: ({ children }) => <h2 className="mb-2 text-lg font-semibold">{children}</h2>,
                      h3: ({ children }) => <h3 className="mb-2 text-base font-medium">{children}</h3>,
                      ul: ({ children }) => <ul className="mb-2 list-disc pl-5 text-sm">{children}</ul>,
                      ol: ({ children }) => <ol className="mb-2 list-decimal pl-5 text-sm">{children}</ol>,
                      li: ({ children }) => <li className="mb-1">{children}</li>,
                      a: ({ href, children }) => (
                        <a
                          href={href}
                          className="text-primary underline hover:no-underline"
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          {children}
                        </a>
                      ),
                      blockquote: ({ children }) => (
                        <blockquote className="my-2 border-l-4 border-muted-foreground/50 pl-4 italic">
                          {children}
                        </blockquote>
                      ),
                      code: ({ inline, children }: { inline?: boolean; children?: React.ReactNode }) =>
                        inline ? (
                          <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">{children}</code>
                        ) : (
                          <code className="block rounded bg-muted p-2 font-mono text-xs">{children}</code>
                        ),
                      pre: ({ children }) => <pre className="overflow-x-auto">{children}</pre>,
                    }}
                  >
                    {value as string}
                  </ReactMarkdown>
                </div>
              ) : isObject && depth < 2 ? (
                <div className="mt-2 space-y-2">
                  {Object.entries(value).map(([k, v]) => renderValue(k, v, depth + 1))}
                </div>
              ) : (
                <Textarea
                  value={stringValue}
                  readOnly
                  className="min-h-[60px] resize-none border-border/50 bg-muted/30 font-mono text-sm text-foreground dark:bg-muted/20"
                />
              )}
            </div>
          )}
        </div>
      );
    };

    return <div className="space-y-4">{Object.entries(output).map(([key, value]) => renderValue(key, value))}</div>;
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "completed":
        return (
          <Badge variant="default" className="border-green-500/20 bg-green-500/10 text-green-700 dark:text-green-400">
            <CheckCircle className="mr-1 h-3 w-3" />
            Completed
          </Badge>
        );
      case "running":
        return (
          <Badge variant="secondary" className="border-blue-500/20 bg-blue-500/10 text-blue-700 dark:text-blue-400">
            <Loader2 className="mr-1 h-3 w-3 animate-spin" />
            Running
          </Badge>
        );
      case "failed":
        return (
          <Badge variant="destructive" className="border-red-500/20 bg-red-500/10 text-red-700 dark:text-red-400">
            <AlertCircle className="mr-1 h-3 w-3" />
            Failed
          </Badge>
        );
      case "queued":
        return (
          <Badge variant="outline" className="border-amber-500/20 bg-amber-500/10 text-amber-700 dark:text-amber-400">
            <Clock className="mr-1 h-3 w-3" />
            Queued
          </Badge>
        );
      default:
        return (
          <Badge variant="outline" className="border-muted-foreground/20 bg-muted/50 text-muted-foreground">
            {status}
          </Badge>
        );
    }
  };

  // Main workflow list view
  if (!selectedWorkflow) {
    return (
      <TooltipProvider>
        <div className="space-y-6">
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
            {workflows.length === 0 ? (
              <div className="col-span-full">
                <Card className="border-2 border-dashed border-muted-foreground/25 bg-muted/5 dark:bg-muted/10">
                  <CardContent className="flex flex-col items-center justify-center py-16">
                    <div className="mb-4 rounded-full bg-primary/10 p-4">
                      <Layers className="h-12 w-12 text-primary" />
                    </div>
                    <h3 className="mb-2 text-lg font-semibold text-foreground">No workflows yet</h3>
                    <p className="mb-6 max-w-sm text-center text-muted-foreground">
                      Create your first workflow to automate document processing
                    </p>
                    <Button
                      onClick={() => {
                        setWorkflowForm({ name: "", description: "", steps: [] });
                        setSelectedStepIndex(null);
                        setIsCreateOpen(true);
                      }}
                      size="lg"
                    >
                      <Plus className="mr-2 h-4 w-4" />
                      Create Workflow
                    </Button>
                  </CardContent>
                </Card>
              </div>
            ) : (
              workflows.map((workflow: Workflow) => (
                <Card
                  key={workflow.id}
                  className="group cursor-pointer border-border/50 bg-card transition-all duration-200 hover:scale-[1.02] hover:border-primary/50 hover:shadow-lg"
                  onClick={() => {
                    setSelectedWorkflow(workflow);
                    // Add the workflow ID to the URL
                    const url = new URL(window.location.href);
                    url.searchParams.set("id", workflow.id);
                    window.history.pushState({}, "", url.toString());
                  }}
                >
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg text-foreground">{workflow.name}</CardTitle>
                      <div className="flex gap-1">
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="opacity-0 transition-opacity duration-200 hover:bg-accent group-hover:opacity-100"
                              onClick={e => {
                                e.stopPropagation();
                                setWorkflowForm({
                                  name: workflow.name,
                                  description: workflow.description || "",
                                  steps: workflow.steps,
                                });
                                setSelectedWorkflow(workflow);
                                setIsEditOpen(true);
                              }}
                            >
                              <Edit2 className="h-4 w-4" />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>Edit Workflow</TooltipContent>
                        </Tooltip>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="opacity-0 transition-opacity duration-200 hover:bg-accent group-hover:opacity-100"
                              onClick={e => {
                                e.stopPropagation();
                                setRunningWorkflow(workflow);
                                setIsRunOpen(true);
                              }}
                            >
                              <Play className="h-4 w-4" />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>Run Workflow</TooltipContent>
                        </Tooltip>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="opacity-0 transition-opacity duration-200 hover:bg-accent group-hover:opacity-100"
                              onClick={async e => {
                                e.stopPropagation();
                                setSelectedWorkflowForFolder(workflow);
                                await fetchWorkflowFolders(workflow.id);
                                setIsFolderDialogOpen(true);
                              }}
                            >
                              <FolderPlus className="h-4 w-4" />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>Manage Folder Associations</TooltipContent>
                        </Tooltip>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="text-destructive opacity-0 transition-opacity duration-200 hover:bg-destructive/10 group-hover:opacity-100"
                              onClick={async e => {
                                e.stopPropagation();
                                if (
                                  window.confirm(
                                    `Are you sure you want to delete "${workflow.name}"? This will remove it from all folders and delete all associated data.`
                                  )
                                ) {
                                  try {
                                    const res = await fetch(`${apiBaseUrl}/workflows/${workflow.id}`, {
                                      method: "DELETE",
                                      headers,
                                    });
                                    if (res.ok) {
                                      await fetchWorkflows();
                                      // If we're viewing this workflow, go back to list
                                      const deletedWorkflowId = workflow.id;
                                      const currentWorkflow = selectedWorkflow as Workflow | null;
                                      if (currentWorkflow && currentWorkflow.id === deletedWorkflowId) {
                                        setSelectedWorkflow(null);
                                        // Clear the URL query parameter
                                        const url = new URL(window.location.href);
                                        url.searchParams.delete("id");
                                        window.history.pushState({}, "", url.toString());
                                      }
                                    } else {
                                      console.error("Failed to delete workflow");
                                    }
                                  } catch (error) {
                                    console.error("Error deleting workflow:", error);
                                  }
                                }
                              }}
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>Delete Workflow</TooltipContent>
                        </Tooltip>
                      </div>
                    </div>
                    {workflow.description && <CardDescription>{workflow.description}</CardDescription>}
                  </CardHeader>
                  <CardContent className="pt-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <Badge
                        variant="secondary"
                        className="border-secondary bg-secondary/50 text-xs text-secondary-foreground"
                      >
                        {workflow.steps.length} step{workflow.steps.length !== 1 ? "s" : ""}
                      </Badge>
                      {workflow.steps.map((step, idx) => {
                        const action = AVAILABLE_ACTIONS.find(a => a.id === step.action_id);
                        return action ? (
                          <Badge
                            key={idx}
                            variant="outline"
                            className="border-primary/20 bg-primary/5 text-xs text-primary"
                          >
                            {action.name}
                          </Badge>
                        ) : null;
                      })}
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </div>

          {/* Run Workflow Dialog */}
          <RunWorkflowDialog
            isOpen={isRunOpen}
            onClose={() => setIsRunOpen(false)}
            workflow={runningWorkflow}
            docs={docs}
            selectedDocIds={selectedDocIds}
            setSelectedDocIds={setSelectedDocIds}
            onRun={runWorkflow}
            isRunning={isRunning}
          />

          {/* Create Workflow Dialog */}
          <WorkflowCreateDialog
            isOpen={isCreateOpen}
            onClose={() => setIsCreateOpen(false)}
            workflowForm={workflowForm}
            setWorkflowForm={setWorkflowForm}
            availableActions={AVAILABLE_ACTIONS}
            onCreateWorkflow={createWorkflow}
            ExtractStructuredParams={ExtractStructuredParams}
          />

          {/* Edit Workflow Dialog */}
          <WorkflowEditDialog
            isOpen={isEditOpen}
            onClose={() => setIsEditOpen(false)}
            workflowForm={workflowForm}
            setWorkflowForm={setWorkflowForm}
            availableActions={AVAILABLE_ACTIONS}
            onUpdateWorkflow={updateWorkflow}
            ExtractStructuredParams={ExtractStructuredParams}
          />

          {/* Folder Association Dialog */}
          <Dialog open={isFolderDialogOpen} onOpenChange={setIsFolderDialogOpen}>
            <DialogContent className="max-w-md">
              <DialogHeader>
                <DialogTitle>Manage Folder Associations</DialogTitle>
                <DialogDescription>
                  Select folders to automatically run &ldquo;{selectedWorkflowForFolder?.name}&rdquo; when documents are
                  added
                </DialogDescription>
              </DialogHeader>
              <div className="max-h-[400px] space-y-2 overflow-y-auto py-4">
                {folders.length === 0 ? (
                  <p className="py-8 text-center text-muted-foreground">No folders available</p>
                ) : (
                  folders.map(folder => {
                    const isAssociated = selectedWorkflowForFolder
                      ? workflowFolders[selectedWorkflowForFolder.id]?.includes(folder.id)
                      : false;
                    return (
                      <Card
                        key={folder.id}
                        className={cn(
                          "cursor-pointer transition-all hover:border-primary/50",
                          isAssociated && "border-primary bg-primary/5"
                        )}
                        onClick={() => {
                          if (selectedWorkflowForFolder) {
                            toggleFolderWorkflow(folder.id, selectedWorkflowForFolder.id, isAssociated);
                          }
                        }}
                      >
                        <CardContent className="flex items-center justify-between p-4">
                          <div className="flex items-center gap-3">
                            <Folder
                              className={cn("h-5 w-5", isAssociated ? "text-primary" : "text-muted-foreground")}
                            />
                            <div>
                              <p className="font-medium">{folder.name}</p>
                              <p className="text-sm text-muted-foreground">
                                {folder.document_count} document{folder.document_count !== 1 ? "s" : ""}
                              </p>
                            </div>
                          </div>
                          {isAssociated && <CheckCircle className="h-5 w-5 text-primary" />}
                        </CardContent>
                      </Card>
                    );
                  })
                )}
              </div>
              <div className="flex justify-end">
                <Button onClick={() => setIsFolderDialogOpen(false)}>Done</Button>
              </div>
            </DialogContent>
          </Dialog>
        </div>
      </TooltipProvider>
    );
  }

  // Workflow details view with runs table
  return (
    <TooltipProvider>
      <div className="space-y-6 p-6">
        <div className="flex items-center gap-4">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              setSelectedWorkflow(null);
              // Clear the URL query parameter
              const url = new URL(window.location.href);
              url.searchParams.delete("id");
              window.history.pushState({}, "", url.toString());
            }}
            className="p-2 transition-colors hover:bg-accent"
          >
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <div className="flex-1">
            <h2 className="text-2xl font-bold text-foreground">{selectedWorkflow.name}</h2>
            {selectedWorkflow.description && <p className="text-muted-foreground">{selectedWorkflow.description}</p>}
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              onClick={() => fetchWorkflowRuns(selectedWorkflow.id)}
              className="transition-colors hover:bg-accent"
            >
              <RefreshCcw className="mr-2 h-4 w-4" />
              Refresh
            </Button>
            <Button
              variant="outline"
              onClick={() => {
                setWorkflowForm({
                  name: selectedWorkflow.name,
                  description: selectedWorkflow.description || "",
                  steps: selectedWorkflow.steps,
                });
                setIsEditOpen(true);
              }}
              className="transition-colors hover:bg-accent"
            >
              <Edit2 className="mr-2 h-4 w-4" />
              Edit
            </Button>
            <Button
              onClick={() => {
                setRunningWorkflow(selectedWorkflow);
                setIsRunOpen(true);
              }}
              className="bg-primary transition-colors hover:bg-primary/90"
            >
              <Play className="mr-2 h-4 w-4" />
              Run on Documents
            </Button>
          </div>
        </div>

        {/* Workflow Steps Visualization */}
        <Card className="border-border/50 bg-card">
          <CardHeader className="cursor-pointer" onClick={() => setIsWorkflowStepsExpanded(!isWorkflowStepsExpanded)}>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-foreground">Workflow Steps</CardTitle>
                <CardDescription>
                  This workflow consists of {selectedWorkflow.steps.length} step
                  {selectedWorkflow.steps.length !== 1 ? "s" : ""}
                </CardDescription>
              </div>
              <Button variant="ghost" size="sm" className="p-2">
                {isWorkflowStepsExpanded ? (
                  <ChevronRight className="h-4 w-4 rotate-90" />
                ) : (
                  <ChevronRight className="h-4 w-4" />
                )}
              </Button>
            </div>
          </CardHeader>
          {isWorkflowStepsExpanded && (
            <CardContent>
              <div className="flex items-center gap-4 overflow-x-auto pb-2">
                {selectedWorkflow.steps.map((step, index) => {
                  const action = AVAILABLE_ACTIONS.find(a => a.id === step.action_id);
                  return (
                    <React.Fragment key={index}>
                      <div className="flex-shrink-0">
                        <Card className="w-64 border-border/50 bg-card transition-all duration-200 hover:scale-105 hover:shadow-md">
                          <CardHeader className="pb-2">
                            <div className="flex items-center gap-2">
                              <div
                                className={cn(
                                  "flex h-8 w-8 items-center justify-center rounded-full text-xs font-bold",
                                  action?.id.includes("extract") && "bg-blue-500/10 text-blue-600 dark:text-blue-400",
                                  action?.id.includes("instruction") &&
                                    "bg-purple-500/10 text-purple-600 dark:text-purple-400",
                                  action?.id.includes("save") && "bg-green-500/10 text-green-600 dark:text-green-400"
                                )}
                              >
                                {index + 1}
                              </div>
                              <CardTitle className="text-sm text-foreground">{action?.name}</CardTitle>
                            </div>
                          </CardHeader>
                          <CardContent className="pt-2">
                            <p className="line-clamp-2 text-xs text-muted-foreground">{action?.description}</p>
                          </CardContent>
                        </Card>
                      </div>
                      {index < selectedWorkflow.steps.length - 1 && (
                        <ChevronRight className="h-5 w-5 flex-shrink-0 animate-pulse text-muted-foreground/70" />
                      )}
                    </React.Fragment>
                  );
                })}
              </div>
            </CardContent>
          )}
        </Card>

        <div className="overflow-hidden rounded-md border border-border/50 bg-card shadow-sm">
          <div className="border-b bg-card p-4">
            <h3 className="text-lg font-semibold">Workflow Runs</h3>
            <p className="text-sm text-muted-foreground">All executions of this workflow on different documents</p>
          </div>

          {workflowRuns.length === 0 ? (
            <div className="py-16 text-center">
              <Clock className="mx-auto mb-4 h-12 w-12 text-muted-foreground/50" />
              <p className="text-muted-foreground">
                No runs yet. Execute this workflow on a document to see results here.
              </p>
            </div>
          ) : (
            <ScrollArea className="h-[calc(100vh-500px)]">
              <Table>
                <TableHeader className="sticky top-0 z-10 border-b bg-background">
                  <TableRow>
                    <TableHead className="w-[40%] py-3 text-foreground">Document</TableHead>
                    <TableHead className="w-[20%] py-3 text-foreground">Date</TableHead>
                    <TableHead className="w-[15%] py-3 text-foreground">Status</TableHead>
                    <TableHead className="w-[25%] py-3 text-right text-foreground">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {workflowRuns.map(run => {
                    const isExpanded = expandedRuns.has(run.id);
                    const progress = runProgress[run.id] || 0;
                    const doc = docs.find(d => d.external_id === run.document_id);

                    return (
                      <React.Fragment key={run.id}>
                        <TableRow className="transition-colors hover:bg-muted/30">
                          <TableCell className="py-4 font-medium">
                            <div className="flex items-center gap-2">
                              <FileText className="h-4 w-4 flex-shrink-0 text-muted-foreground" />
                              <span className="truncate text-foreground" title={doc?.filename || run.document_id}>
                                {doc?.filename || run.document_id}
                              </span>
                            </div>
                          </TableCell>
                          <TableCell className="py-4 text-sm text-muted-foreground">
                            {(() => {
                              const dateStr = (run.started_at ?? run.created_at) as string | undefined;
                              if (!dateStr) return "Not started";
                              const d = new Date(dateStr);
                              return isNaN(d.getTime()) ? "-" : d.toLocaleString();
                            })()}
                          </TableCell>
                          <TableCell className="py-4">
                            <div className="flex items-center gap-2">
                              {getStatusBadge(run.status)}
                              {run.status === "running" && <Progress value={progress} className="h-2 w-20 bg-muted" />}
                            </div>
                          </TableCell>
                          <TableCell className="py-4 text-right">
                            <div className="flex items-center justify-end gap-2">
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() =>
                                  setExpandedRuns(prev => {
                                    const newSet = new Set(prev);
                                    if (newSet.has(run.id)) {
                                      newSet.delete(run.id);
                                    } else {
                                      newSet.add(run.id);
                                    }
                                    return newSet;
                                  })
                                }
                                className="transition-colors hover:bg-accent"
                              >
                                {isExpanded ? (
                                  <>
                                    <EyeOff className="mr-2 h-4 w-4" />
                                    Hide Details
                                  </>
                                ) : (
                                  <>
                                    <Eye className="mr-2 h-4 w-4" />
                                    View Details
                                  </>
                                )}
                              </Button>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={async () => {
                                      if (
                                        confirm(
                                          `Are you sure you want to delete this run? This will only delete the run record, not any data that was added to document metadata.`
                                        )
                                      ) {
                                        try {
                                          const response = await fetch(`${apiBaseUrl}/workflows/runs/${run.id}`, {
                                            method: "DELETE",
                                            headers: {
                                              "Content-Type": "application/json",
                                              ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
                                            },
                                          });
                                          if (response.ok) {
                                            // Refresh the workflow runs
                                            const runsResponse = await fetch(
                                              `${apiBaseUrl}/workflows/${selectedWorkflow.id}/runs`,
                                              {
                                                headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
                                              }
                                            );
                                            if (runsResponse.ok) {
                                              const runs = await runsResponse.json();
                                              setWorkflowRuns(runs);
                                            }
                                          } else {
                                            console.error("Failed to delete run");
                                          }
                                        } catch (error) {
                                          console.error("Error deleting run:", error);
                                        }
                                      }
                                    }}
                                    className="text-destructive transition-colors hover:bg-destructive/10"
                                  >
                                    <Trash2 className="h-4 w-4" />
                                  </Button>
                                </TooltipTrigger>
                                <TooltipContent>Delete Run</TooltipContent>
                              </Tooltip>
                            </div>
                          </TableCell>
                        </TableRow>
                        {isExpanded && (
                          <TableRow>
                            <TableCell colSpan={4} className="p-0">
                              <div className="border-t bg-muted/20 p-6 dark:bg-muted/10">
                                <Tabs defaultValue="output" className="w-full">
                                  <TabsList className="grid w-full max-w-[400px] grid-cols-3 bg-muted">
                                    <TabsTrigger value="output" className="data-[state=active]:bg-background">
                                      Output
                                    </TabsTrigger>
                                    <TabsTrigger value="steps" className="data-[state=active]:bg-background">
                                      Step Results
                                    </TabsTrigger>
                                    <TabsTrigger
                                      value="error"
                                      disabled={!run.error}
                                      className="data-[state=active]:bg-background"
                                    >
                                      Error Details
                                    </TabsTrigger>
                                  </TabsList>

                                  <TabsContent value="output" className="mt-4">
                                    {run.status === "completed" && run.final_output ? (
                                      <div className="space-y-3">
                                        {renderExtractedData(
                                          run.final_output,
                                          true,
                                          // Check if the last step was convert_to_markdown
                                          selectedWorkflow.steps[selectedWorkflow.steps.length - 1]?.action_id ===
                                            "morphik.actions.convert_to_markdown" &&
                                            typeof run.final_output === "object" &&
                                            run.final_output !== null &&
                                            "markdown" in run.final_output
                                        )}
                                      </div>
                                    ) : run.status === "running" ? (
                                      <div className="flex items-center gap-2 py-4 text-blue-600 dark:text-blue-400">
                                        <Loader2 className="h-4 w-4 animate-spin" />
                                        Processing document...
                                      </div>
                                    ) : run.status === "failed" ? (
                                      <Alert variant="destructive" className="border-destructive/50 bg-destructive/10">
                                        <AlertCircle className="h-4 w-4" />
                                        <AlertTitle>Execution Failed</AlertTitle>
                                        <AlertDescription>
                                          The workflow failed to complete. Check the error details tab for more
                                          information.
                                        </AlertDescription>
                                      </Alert>
                                    ) : (
                                      <p className="py-4 text-muted-foreground">No output available</p>
                                    )}
                                  </TabsContent>

                                  <TabsContent value="steps" className="mt-4">
                                    {run.results_per_step && run.results_per_step.length > 0 ? (
                                      <div className="space-y-4">
                                        {run.results_per_step.map((result, idx) => {
                                          const step = selectedWorkflow.steps[idx];
                                          const action = AVAILABLE_ACTIONS.find(a => a.id === step?.action_id);
                                          return (
                                            <Card key={idx} className="border-border/50 bg-card">
                                              <CardHeader className="pb-2">
                                                <div className="flex items-center gap-2">
                                                  <Badge
                                                    variant="outline"
                                                    className="border-primary/20 bg-primary/10 text-xs text-primary"
                                                  >
                                                    Step {idx + 1}
                                                  </Badge>
                                                  <span className="text-sm font-medium text-foreground">
                                                    {action?.name}
                                                  </span>
                                                </div>
                                              </CardHeader>
                                              <CardContent>
                                                <div className="text-sm">
                                                  {renderExtractedData(
                                                    result,
                                                    false,
                                                    // Check if this step was convert_to_markdown
                                                    step?.action_id === "morphik.actions.convert_to_markdown" &&
                                                      typeof result === "object" &&
                                                      result !== null &&
                                                      "markdown" in result
                                                  )}
                                                </div>
                                              </CardContent>
                                            </Card>
                                          );
                                        })}
                                      </div>
                                    ) : (
                                      <p className="py-4 text-muted-foreground">No intermediate results available</p>
                                    )}
                                  </TabsContent>

                                  <TabsContent value="error" className="mt-4">
                                    {run.error && (
                                      <Alert variant="destructive" className="border-destructive/50 bg-destructive/10">
                                        <AlertCircle className="h-4 w-4" />
                                        <AlertTitle>Error Details</AlertTitle>
                                        <AlertDescription className="mt-2">
                                          <pre className="overflow-x-auto whitespace-pre-wrap font-mono text-xs">
                                            {run.error}
                                          </pre>
                                        </AlertDescription>
                                      </Alert>
                                    )}
                                  </TabsContent>
                                </Tabs>
                              </div>
                            </TableCell>
                          </TableRow>
                        )}
                      </React.Fragment>
                    );
                  })}
                </TableBody>
              </Table>
            </ScrollArea>
          )}
        </div>

        {/* Run Workflow Dialog */}
        <RunWorkflowDialog
          isOpen={isRunOpen}
          onClose={() => setIsRunOpen(false)}
          workflow={runningWorkflow}
          docs={docs}
          selectedDocIds={selectedDocIds}
          setSelectedDocIds={setSelectedDocIds}
          onRun={runWorkflow}
          isRunning={isRunning}
        />

        {/* Edit Workflow Dialog */}
        <WorkflowEditDialog
          isOpen={isEditOpen}
          onClose={() => setIsEditOpen(false)}
          workflowForm={workflowForm}
          setWorkflowForm={setWorkflowForm}
          availableActions={AVAILABLE_ACTIONS}
          onUpdateWorkflow={updateWorkflow}
          ExtractStructuredParams={ExtractStructuredParams}
        />
      </div>
    </TooltipProvider>
  );
};

// Helper Components

interface ExtractStructuredParameters {
  schema?: {
    type: string;
    properties: Record<string, { type?: string; description?: string }>;
    required?: string[];
  };
  [key: string]: unknown;
}

const ExtractStructuredParams: React.FC<{
  parameters: ExtractStructuredParameters;
  onChange: (params: ExtractStructuredParameters) => void;
}> = ({ parameters, onChange }) => {
  const [schemaFields, setSchemaFields] = useState<SchemaField[]>(() => {
    // Initialize from existing schema if present
    if (parameters.schema?.properties) {
      return Object.entries(parameters.schema.properties).map(([name, prop]) => ({
        name,
        type: (prop.type || "string") as SchemaField["type"],
        description: (prop.description as string) || "",
        required: parameters.schema?.required?.includes(name) || false,
      }));
    }
    return [{ name: "title", type: "string", description: "Document title", required: true }];
  });

  const [isJsonDialogOpen, setIsJsonDialogOpen] = useState(false);
  const [jsonInput, setJsonInput] = useState("");
  const [jsonError, setJsonError] = useState("");

  const addField = () => {
    setSchemaFields([...schemaFields, { name: "", type: "string", description: "", required: true }]);
  };

  const updateField = (index: number, field: Partial<SchemaField>) => {
    const updated = [...schemaFields];
    updated[index] = { ...updated[index], ...field };
    setSchemaFields(updated);

    // Update parent parameters
    const schema = generateSchema(updated);
    onChange({ ...parameters, schema });
  };

  const removeField = (index: number) => {
    const updated = schemaFields.filter((_, i) => i !== index);
    setSchemaFields(updated);

    // Update parent parameters
    const schema = generateSchema(updated);
    onChange({ ...parameters, schema });
  };

  const generateSchema = (fields: SchemaField[]) => {
    const properties: Record<string, { type: string; description: string; items?: { type: string } }> = {};
    const required: string[] = [];

    fields.forEach(field => {
      if (field.name) {
        const propertySchema: { type: string; description: string; items?: { type: string } } = {
          type: field.type,
          description: field.description || "",
        };
        if (field.type === "array") {
          // Default to array of strings if not specified. This avoids OpenAI schema errors.
          propertySchema.items = { type: "string" };
        }
        properties[field.name] = propertySchema;
        if (field.required) {
          required.push(field.name);
        }
      }
    });

    return {
      type: "object",
      properties,
      required: required.length > 0 ? required : undefined,
    };
  };

  const inferFieldType = (value: unknown): SchemaField["type"] => {
    if (value === null || value === undefined) return "string";
    if (typeof value === "string") return "string";
    if (typeof value === "number") return "number";
    if (typeof value === "boolean") return "boolean";
    if (Array.isArray(value)) return "array";
    if (typeof value === "object") return "object";
    return "string";
  };

  const parseJsonToFields = (json: string) => {
    try {
      const parsed = JSON.parse(json);
      setJsonError("");

      if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
        setJsonError("Please provide a JSON object (not an array or primitive value)");
        return;
      }

      const fields: SchemaField[] = [];

      // Check if it's a schema-like format with properties
      if (parsed.properties && typeof parsed.properties === "object") {
        // Handle JSON Schema format
        Object.entries(parsed.properties).forEach(([key, prop]: [string, unknown]) => {
          const propObj = prop as Record<string, unknown>;
          fields.push({
            name: key,
            type: ((propObj.type as string) || "string") as SchemaField["type"],
            description: (propObj.description as string) || "",
            required: parsed.required?.includes(key) || false,
          });
        });
      } else {
        // Handle field definition format (only top-level)
        Object.entries(parsed).forEach(([key, value]) => {
          // Check if value is a field definition object
          if (value && typeof value === "object" && !Array.isArray(value)) {
            const valueObj = value as Record<string, unknown>;
            if (valueObj.type || valueObj.description !== undefined) {
              // This looks like a field definition
              fields.push({
                name: key,
                type: ((valueObj.type as string) || "string") as SchemaField["type"],
                description: (valueObj.description as string) || "",
                required: (valueObj.required as boolean) || false,
              });
            } else {
              // Regular value - infer type
              fields.push({
                name: key,
                type: inferFieldType(value),
                description: "",
                required: false,
              });
            }
          } else {
            // Regular value - infer type
            fields.push({
              name: key,
              type: inferFieldType(value),
              description: "",
              required: false,
            });
          }
        });
      }

      setSchemaFields(fields);
      const schema = generateSchema(fields);
      onChange({ ...parameters, schema });
      setIsJsonDialogOpen(false);
      setJsonInput("");
    } catch (error) {
      setJsonError(`Invalid JSON: ${error instanceof Error ? error.message : "Unknown error"}`);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <Label>Data Schema</Label>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => setIsJsonDialogOpen(true)}>
            <FileJson className="mr-1 h-3 w-3" />
            Create from JSON
          </Button>
          <Button variant="outline" size="sm" onClick={addField}>
            <Plus className="mr-1 h-3 w-3" />
            Add Field
          </Button>
        </div>
      </div>

      <div className="space-y-3">
        {schemaFields.map((field, index) => (
          <div key={index} className="flex items-end gap-2">
            <div className="flex-1">
              <Label className="text-xs text-foreground">Field Name</Label>
              <Input
                value={field.name}
                onChange={e => updateField(index, { name: e.target.value })}
                placeholder="e.g., amount"
                className="h-8 border-input bg-background"
              />
            </div>
            <div className="w-32">
              <Label className="text-xs text-foreground">Type</Label>
              <Select
                value={field.type}
                onValueChange={value => updateField(index, { type: value as SchemaField["type"] })}
              >
                <SelectTrigger className="h-8 border-input bg-background">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="border-border bg-popover">
                  <SelectItem value="string">Text</SelectItem>
                  <SelectItem value="number">Number</SelectItem>
                  <SelectItem value="boolean">True/False</SelectItem>
                  <SelectItem value="array">List</SelectItem>
                  <SelectItem value="object">Object</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex-1">
              <Label className="text-xs text-foreground">Description</Label>
              <Input
                value={field.description || ""}
                onChange={e => updateField(index, { description: e.target.value })}
                placeholder="What to extract..."
                className="h-8 border-input bg-background"
              />
            </div>
            <div className="flex items-center gap-2">
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      checked={field.required || false}
                      onChange={e => updateField(index, { required: e.target.checked })}
                      className="mr-1"
                    />
                    <Label className="cursor-pointer text-xs">Req</Label>
                  </div>
                </TooltipTrigger>
                <TooltipContent>Required field</TooltipContent>
              </Tooltip>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => removeField(index)}
                className="h-8 w-8 p-0 transition-colors hover:bg-destructive/10"
              >
                <Trash2 className="h-3 w-3" />
              </Button>
            </div>
          </div>
        ))}
      </div>

      <Dialog open={isJsonDialogOpen} onOpenChange={setIsJsonDialogOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Create Schema from JSON</DialogTitle>
            <DialogDescription>
              Paste your JSON object below. Fields will be automatically inferred from the structure.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="json-input">JSON Input</Label>
              <Textarea
                id="json-input"
                value={jsonInput}
                onChange={e => {
                  setJsonInput(e.target.value);
                  setJsonError("");
                }}
                placeholder={`Examples:
1. Simple JSON: {"name": "John", "age": 30, "isActive": true}
2. With descriptions: {"name": {"type": "string", "description": "Person's full name"}, "age": {"type": "number", "description": "Age in years"}}
3. JSON Schema: {"properties": {"name": {"type": "string", "description": "Full name"}}, "required": ["name"]}`}
                className="h-64 font-mono text-sm"
              />
              {jsonError && (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{jsonError}</AlertDescription>
                </Alert>
              )}
            </div>
            <div className="flex justify-end gap-2">
              <Button
                variant="outline"
                onClick={() => {
                  setIsJsonDialogOpen(false);
                  setJsonInput("");
                  setJsonError("");
                }}
              >
                Cancel
              </Button>
              <Button onClick={() => parseJsonToFields(jsonInput)}>
                <FileJson className="mr-2 h-4 w-4" />
                Import Fields
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
};

const RunWorkflowDialog: React.FC<{
  isOpen: boolean;
  onClose: () => void;
  workflow: Workflow | null;
  docs: DocumentMeta[];
  selectedDocIds: string[];
  setSelectedDocIds: (ids: string[]) => void;
  onRun: () => void;
  isRunning: boolean;
}> = ({ isOpen, onClose, workflow, docs, selectedDocIds, setSelectedDocIds, onRun, isRunning }) => {
  if (!workflow) return null;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="flex max-h-[90vh] max-w-2xl flex-col overflow-hidden border-border bg-background">
        <DialogHeader>
          <DialogTitle className="text-foreground">Run Workflow: {workflow.name}</DialogTitle>
        </DialogHeader>
        <div className="flex-1 space-y-4 overflow-y-auto py-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between px-1">
              <Label className="text-foreground">Select Documents ({selectedDocIds.length} selected)</Label>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setSelectedDocIds(docs.map(d => d.external_id))}
                  className="transition-colors hover:bg-accent"
                >
                  Select All
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setSelectedDocIds([])}
                  className="transition-colors hover:bg-accent"
                >
                  Clear
                </Button>
              </div>
            </div>
            <div className="h-64 overflow-y-auto rounded-md border border-border/50">
              <Table>
                <TableHeader className="sticky top-0 border-b bg-background">
                  <TableRow>
                    <TableHead className="w-[50px] py-3 text-foreground">Select</TableHead>
                    <TableHead className="py-3 text-foreground">Document Name</TableHead>
                    <TableHead className="w-[150px] py-3 text-foreground">Type</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {docs.map(doc => {
                    const checked = selectedDocIds.includes(doc.external_id);
                    return (
                      <TableRow
                        key={doc.external_id}
                        className="cursor-pointer transition-colors hover:bg-muted/30"
                        onClick={() => {
                          setSelectedDocIds(
                            checked
                              ? selectedDocIds.filter(id => id !== doc.external_id)
                              : [...selectedDocIds, doc.external_id]
                          );
                        }}
                      >
                        <TableCell className="py-3">
                          <input
                            type="checkbox"
                            checked={checked}
                            readOnly
                            className="accent-primary"
                            onClick={e => e.stopPropagation()}
                          />
                        </TableCell>
                        <TableCell className="py-3 font-medium">
                          <div className="flex items-center gap-2">
                            <FileText className="h-4 w-4 flex-shrink-0 text-muted-foreground" />
                            <span className="truncate text-foreground" title={doc.filename || doc.external_id}>
                              {doc.filename || doc.external_id}
                            </span>
                          </div>
                        </TableCell>
                        <TableCell className="py-3 text-sm text-muted-foreground">{doc.content_type}</TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          </div>

          {selectedDocIds.length > 0 && (
            <Alert>
              <Info className="h-4 w-4" />
              <AlertTitle>Batch Processing</AlertTitle>
              <AlertDescription>
                The workflow will run on {selectedDocIds.length} document{selectedDocIds.length !== 1 ? "s" : ""} in
                parallel. You can monitor the progress of each run separately.
              </AlertDescription>
            </Alert>
          )}
        </div>

        <div className="flex justify-end gap-2 border-t pt-4">
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={onRun} disabled={selectedDocIds.length === 0 || isRunning}>
            {isRunning ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Running...
              </>
            ) : (
              <>
                <Play className="mr-2 h-4 w-4" />
                Run Workflow
              </>
            )}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default WorkflowSection;
