"use client";

export const dynamic = "force-dynamic";

import { useEffect } from "react";
import WorkflowSection from "@/components/workflows/WorkflowSection";
import { useMorphik } from "@/contexts/morphik-context";
import { useHeader } from "@/contexts/header-context";
import { Button } from "@/components/ui/button";
import { Plus, RefreshCcw } from "lucide-react";

export default function WorkflowsPage() {
  const { apiBaseUrl, authToken } = useMorphik();
  const { setRightContent, setCustomBreadcrumbs } = useHeader();

  useEffect(() => {
    // Set breadcrumbs
    const breadcrumbs = [{ label: "Home", href: "/" }, { label: "Workflows" }];
    setCustomBreadcrumbs(breadcrumbs);

    // Set header controls
    const rightContent = (
      <>
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            const event = new CustomEvent("refreshWorkflows");
            window.dispatchEvent(event);
          }}
          title="Refresh workflows"
        >
          <RefreshCcw className="h-4 w-4" />
          <span className="ml-1">Refresh</span>
        </Button>
        <Button
          variant="default"
          size="sm"
          onClick={() => {
            const event = new CustomEvent("openNewWorkflowDialog");
            window.dispatchEvent(event);
          }}
        >
          <Plus className="mr-2 h-4 w-4" />
          New Workflow
        </Button>
      </>
    );

    setRightContent(rightContent);

    return () => {
      setRightContent(null);
      setCustomBreadcrumbs(null);
    };
  }, [setRightContent, setCustomBreadcrumbs]);

  return (
    <div className="flex flex-1 flex-col">
      <WorkflowSection apiBaseUrl={apiBaseUrl} authToken={authToken} />
    </div>
  );
}
