import React from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Plus, Layers } from "lucide-react";
import { WorkflowStepInlineConfig } from "./WorkflowStepInlineConfig";
import { Card, CardContent } from "@/components/ui/card";

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
  parameters_schema: Record<string, unknown>;
  output_schema: Record<string, unknown>;
}

interface WorkflowEditDialogProps {
  isOpen: boolean;
  onClose: () => void;
  workflowForm: {
    name: string;
    description: string;
    steps: ConfiguredAction[];
  };
  setWorkflowForm: React.Dispatch<
    React.SetStateAction<{
      name: string;
      description: string;
      steps: ConfiguredAction[];
    }>
  >;
  availableActions: ActionDefinition[];
  onUpdateWorkflow: () => void;
  ExtractStructuredParams: React.ComponentType<{
    parameters: Record<string, unknown>;
    onChange: (params: Record<string, unknown>) => void;
  }>;
}

export const WorkflowEditDialog: React.FC<WorkflowEditDialogProps> = ({
  isOpen,
  onClose,
  workflowForm,
  setWorkflowForm,
  availableActions,
  onUpdateWorkflow,
  ExtractStructuredParams,
}) => {
  const addStep = () => {
    setWorkflowForm(prev => ({
      ...prev,
      steps: [...prev.steps, { action_id: "", parameters: {} }],
    }));
  };

  const updateStep = (index: number, updates: Partial<ConfiguredAction>) => {
    setWorkflowForm(prev => ({
      ...prev,
      steps: prev.steps.map((step, i) => (i === index ? { ...step, ...updates } : step)),
    }));
  };

  const removeStep = (index: number) => {
    setWorkflowForm(prev => ({
      ...prev,
      steps: prev.steps.filter((_, i) => i !== index),
    }));
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="flex h-[85vh] max-w-3xl flex-col overflow-hidden p-0">
        <DialogHeader className="flex-shrink-0 p-6 pb-4">
          <DialogTitle className="text-xl">Edit Workflow</DialogTitle>
        </DialogHeader>

        <ScrollArea className="flex-1 overflow-y-auto px-6">
          <div className="space-y-6 pb-6">
            {/* Form Fields */}
            <div className="space-y-4">
              <div>
                <Label className="text-foreground">Workflow Name</Label>
                <Input
                  value={workflowForm.name}
                  onChange={e => setWorkflowForm(prev => ({ ...prev, name: e.target.value }))}
                  placeholder="Enter workflow name..."
                  className="mt-1 border-input bg-background"
                />
              </div>
              <div>
                <Label className="text-foreground">Description</Label>
                <Textarea
                  value={workflowForm.description}
                  onChange={e => setWorkflowForm(prev => ({ ...prev, description: e.target.value }))}
                  placeholder="Describe what this workflow does..."
                  className="mt-1 h-20 resize-none border-input bg-background"
                />
              </div>
            </div>

            {/* Workflow Steps */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-foreground">Workflow Steps</h3>
                {workflowForm.steps.length > 0 && (
                  <Button variant="outline" size="sm" onClick={addStep} className="border-dashed">
                    <Plus className="mr-2 h-4 w-4" />
                    Add Step
                  </Button>
                )}
              </div>

              {workflowForm.steps.length === 0 ? (
                <Card className="border-2 border-dashed border-muted-foreground/25 bg-card/50">
                  <CardContent className="flex flex-col items-center justify-center py-12">
                    <div className="rounded-full bg-primary/10 p-3">
                      <Layers className="h-8 w-8 text-primary" />
                    </div>
                    <h4 className="mt-3 text-base font-semibold text-foreground">Start Building Your Workflow</h4>
                    <p className="mt-1 text-center text-sm text-muted-foreground">
                      Add actions to process your documents automatically
                    </p>
                    <Button
                      onClick={addStep}
                      className="mt-4 bg-primary transition-colors hover:bg-primary/90"
                      size="sm"
                    >
                      <Plus className="mr-2 h-4 w-4" />
                      Add First Step
                    </Button>
                  </CardContent>
                </Card>
              ) : (
                <div className="space-y-4">
                  {workflowForm.steps.map((step, index) => (
                    <WorkflowStepInlineConfig
                      key={index}
                      step={step}
                      index={index}
                      availableActions={availableActions}
                      onUpdate={updateStep}
                      onRemove={removeStep}
                      ExtractStructuredParams={ExtractStructuredParams}
                    />
                  ))}

                  {/* Add Step Button */}
                  {workflowForm.steps.length > 0 && (
                    <div className="relative pl-5">
                      <div className="absolute left-8 top-0 h-4 w-0.5 bg-gradient-to-b from-border via-border to-transparent" />
                      <Button
                        variant="outline"
                        onClick={addStep}
                        className="ml-3 border-2 border-dashed transition-all duration-200 hover:border-primary hover:bg-primary/5"
                      >
                        <Plus className="mr-2 h-4 w-4" />
                        Add Step
                      </Button>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </ScrollArea>

        {/* Actions */}
        <div className="flex flex-shrink-0 justify-end gap-2 border-t p-6 pt-4">
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button
            onClick={onUpdateWorkflow}
            disabled={!workflowForm.name.trim() || workflowForm.steps.length === 0}
            className="bg-primary transition-colors hover:bg-primary/90"
          >
            Save Changes
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};
