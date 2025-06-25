import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import { Edit2, FileJson, Save, Trash2, ChevronDown, ChevronUp } from "lucide-react";

interface SchemaField {
  name: string;
  type: "string" | "number" | "boolean" | "array" | "object";
  description?: string;
  required?: boolean;
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

interface ActionDefinition {
  id: string;
  name: string;
  description: string;
}

interface WorkflowStepInlineConfigProps {
  step: ConfiguredAction;
  index: number;
  availableActions: ActionDefinition[];
  onUpdate: (index: number, updates: Partial<ConfiguredAction>) => void;
  onRemove: (index: number) => void;
  ExtractStructuredParams: React.ComponentType<{
    parameters: Record<string, unknown>;
    onChange: (params: Record<string, unknown>) => void;
  }>;
}

export const WorkflowStepInlineConfig: React.FC<WorkflowStepInlineConfigProps> = ({
  step,
  index,
  availableActions,
  onUpdate,
  onRemove,
  ExtractStructuredParams,
}) => {
  const [isConfigOpen, setIsConfigOpen] = useState(false);
  const action = availableActions.find(a => a.id === step.action_id);

  // If no action is selected, show inline selector
  if (!step.action_id) {
    return (
      <div className="relative">
        {index > 0 && (
          <div className="absolute -top-4 left-8 h-4 w-0.5 bg-gradient-to-b from-transparent via-border to-border" />
        )}
        <Card className="group relative border-2 border-border bg-card p-4">
          <div className="absolute -left-5 top-4 flex h-10 w-10 items-center justify-center rounded-full border-2 border-border bg-background text-sm font-medium">
            {index + 1}
          </div>
          <div className="ml-8">
            <div className="space-y-2">
              <h4 className="font-medium text-foreground">Select an Action</h4>
              <Select
                value={step.action_id}
                onValueChange={value => {
                  onUpdate(index, {
                    action_id: value,
                    parameters: {}, // Reset parameters when changing action
                  });
                }}
              >
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Choose an action type" />
                </SelectTrigger>
                <SelectContent>
                  {availableActions.map(action => (
                    <SelectItem key={action.id} value={action.id}>
                      <div className="flex items-center gap-2">
                        {action.id.includes("extract") && <FileJson className="h-4 w-4 text-blue-500" />}
                        {action.id.includes("instruction") && <Edit2 className="h-4 w-4 text-purple-500" />}
                        {action.id.includes("save") && <Save className="h-4 w-4 text-green-500" />}
                        <span>{action.name}</span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="relative">
      {/* Connection Line */}
      {index > 0 && (
        <div className="absolute -top-4 left-8 h-4 w-0.5 bg-gradient-to-b from-transparent via-border to-border" />
      )}

      {/* Step Node */}
      <div className="space-y-0">
        <Card
          className={cn(
            "group relative border-2 bg-card transition-all duration-200",
            isConfigOpen ? "border-primary shadow-md" : "border-border hover:border-primary/50 hover:shadow-md"
          )}
        >
          {/* Step Number */}
          <div className="absolute -left-5 top-4 flex h-10 w-10 items-center justify-center rounded-full border-2 border-border bg-background text-sm font-medium group-hover:border-primary/50">
            {index + 1}
          </div>

          {/* Step Content */}
          <div className="p-4">
            <div className="ml-8">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3">
                    {action?.id.includes("extract") && (
                      <div className="rounded-lg bg-blue-500/10 p-2">
                        <FileJson className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                      </div>
                    )}
                    {action?.id.includes("instruction") && (
                      <div className="rounded-lg bg-purple-500/10 p-2">
                        <Edit2 className="h-5 w-5 text-purple-600 dark:text-purple-400" />
                      </div>
                    )}
                    {action?.id.includes("save") && (
                      <div className="rounded-lg bg-green-500/10 p-2">
                        <Save className="h-5 w-5 text-green-600 dark:text-green-400" />
                      </div>
                    )}
                    <div>
                      <h4 className="font-medium text-foreground">{action?.name || "Unknown Action"}</h4>
                      <p className="text-sm text-muted-foreground">{action?.description}</p>
                    </div>
                  </div>

                  {/* Quick Preview of Configuration */}
                  {!isConfigOpen && (
                    <>
                      {step.action_id === "morphik.actions.extract_structured" && step.parameters.schema && (
                        <div className="mt-2 text-sm text-muted-foreground">
                          Extracting:{" "}
                          {Object.keys(
                            typeof step.parameters.schema === "object" &&
                              step.parameters.schema !== null &&
                              "properties" in (step.parameters.schema as Record<string, unknown>)
                              ? (step.parameters.schema as { properties: Record<string, unknown> }).properties
                              : {}
                          ).join(", ") || "No fields defined"}
                        </div>
                      )}
                      {step.action_id === "morphik.actions.apply_instruction" && step.parameters.prompt_template && (
                        <div className="mt-2 text-sm text-muted-foreground">
                          Instruction: {(step.parameters.prompt_template as string).substring(0, 50)}...
                        </div>
                      )}
                      {step.action_id === "morphik.actions.save_to_metadata" && step.parameters.metadata_key && (
                        <div className="mt-2 text-sm text-muted-foreground">
                          Saving to: {step.parameters.metadata_key as string}
                        </div>
                      )}
                    </>
                  )}
                </div>

                {/* Action Buttons */}
                <div className="flex items-center gap-2">
                  <Button variant="outline" size="sm" onClick={() => setIsConfigOpen(!isConfigOpen)} className="gap-1">
                    <Edit2 className="h-3 w-3" />
                    Configure
                    {isConfigOpen ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                  </Button>

                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={e => {
                      e.stopPropagation();
                      onRemove(index);
                    }}
                    className="opacity-0 transition-all duration-200 hover:bg-destructive/10 group-hover:opacity-100"
                  >
                    <Trash2 className="h-4 w-4 text-destructive" />
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </Card>

        {/* Configuration Panel */}
        {isConfigOpen && (
          <div className="transition-all duration-200">
            <Card className="rounded-t-none border-2 border-t-0 border-primary/50 bg-muted/30 p-6">
              <div className="space-y-4">
                <div>
                  <Label className="text-foreground">Action Type</Label>
                  <Select
                    value={step.action_id}
                    onValueChange={value => {
                      onUpdate(index, {
                        action_id: value,
                        parameters: {}, // Reset parameters when changing action
                      });
                    }}
                  >
                    <SelectTrigger className="mt-1">
                      <SelectValue placeholder="Select an action" />
                    </SelectTrigger>
                    <SelectContent>
                      {availableActions.map(action => (
                        <SelectItem key={action.id} value={action.id}>
                          <div className="flex items-center gap-2">
                            {action.id.includes("extract") && <FileJson className="h-4 w-4 text-blue-500" />}
                            {action.id.includes("instruction") && <Edit2 className="h-4 w-4 text-purple-500" />}
                            {action.id.includes("save") && <Save className="h-4 w-4 text-green-500" />}
                            <span>{action.name}</span>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <Separator />

                {/* Action-specific Parameters */}
                {step.action_id === "morphik.actions.extract_structured" && (
                  <ExtractStructuredParams
                    parameters={step.parameters}
                    onChange={params => onUpdate(index, { parameters: params })}
                  />
                )}

                {step.action_id === "morphik.actions.apply_instruction" && (
                  <div>
                    <Label>Instruction Template</Label>
                    <Textarea
                      value={(step.parameters.prompt_template as string) || ""}
                      onChange={e =>
                        onUpdate(index, {
                          parameters: {
                            ...step.parameters,
                            prompt_template: e.target.value,
                          },
                        })
                      }
                      placeholder="Enter your instruction. Use {input_text} to reference the document content."
                      className="mt-1 min-h-32"
                    />
                    <p className="mt-1 text-xs text-muted-foreground">
                      Use {"{input_text}"} to reference the document content in your instruction.
                    </p>
                  </div>
                )}

                {step.action_id === "morphik.actions.save_to_metadata" && (
                  <div className="space-y-4">
                    <div>
                      <Label>Metadata Key</Label>
                      <Input
                        value={(step.parameters.metadata_key as string) || ""}
                        onChange={e =>
                          onUpdate(index, {
                            parameters: {
                              ...step.parameters,
                              metadata_key: e.target.value,
                            },
                          })
                        }
                        placeholder="e.g., document_info"
                        className="mt-1"
                      />
                      <p className="mt-1 text-xs text-muted-foreground">
                        The key under which the output will be saved (leave empty to merge fields at top level).
                      </p>
                    </div>
                    <div>
                      <Label>Data Source</Label>
                      <Select
                        value={(step.parameters.source as string) || "previous_step"}
                        onValueChange={value =>
                          onUpdate(index, {
                            parameters: {
                              ...step.parameters,
                              source: value,
                            },
                          })
                        }
                      >
                        <SelectTrigger className="mt-1">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="previous_step">Previous Step Output</SelectItem>
                          <SelectItem value="all_steps">All Steps Output</SelectItem>
                        </SelectContent>
                      </Select>
                      <p className="mt-1 text-xs text-muted-foreground">
                        Choose whether to save only the previous step&apos;s output or all steps&apos; outputs.
                      </p>
                    </div>
                  </div>
                )}

                <div className="flex justify-end pt-2">
                  <Button size="sm" onClick={() => setIsConfigOpen(false)}>
                    Done
                  </Button>
                </div>
              </div>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
};
