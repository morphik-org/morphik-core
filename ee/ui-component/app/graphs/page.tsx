"use client";

export const dynamic = "force-dynamic";

import { useEffect, useState } from "react";
import GraphSection from "@/components/GraphSection";
import { useMorphik } from "@/contexts/morphik-context";
import { useHeader } from "@/contexts/header-context";
import { Button } from "@/components/ui/button";
import { Plus } from "lucide-react";

export default function GraphsPage() {
  const { apiBaseUrl, authToken } = useMorphik();
  const { setRightContent, setCustomBreadcrumbs } = useHeader();
  const [showCreateDialog, setShowCreateDialog] = useState(false);

  // Set up breadcrumbs
  useEffect(() => {
    setCustomBreadcrumbs([{ label: "Home", href: "/" }, { label: "Graphs" }]);

    return () => {
      setCustomBreadcrumbs(null);
    };
  }, [setCustomBreadcrumbs]);

  // Set up header controls
  useEffect(() => {
    const rightContent = (
      <Button
        variant="default"
        size="sm"
        onClick={() => {
          const event = new CustomEvent("openCreateGraphDialog");
          window.dispatchEvent(event);
        }}
      >
        <Plus className="mr-2 h-4 w-4" />
        New Graph
      </Button>
    );

    setRightContent(rightContent);

    return () => {
      setRightContent(null);
    };
  }, [setRightContent]);

  // Listen for events
  useEffect(() => {
    const handleOpenCreateDialog = () => {
      setShowCreateDialog(true);
    };

    window.addEventListener("openCreateGraphDialog", handleOpenCreateDialog);

    return () => {
      window.removeEventListener("openCreateGraphDialog", handleOpenCreateDialog);
    };
  }, []);

  return (
    <GraphSection
      apiBaseUrl={apiBaseUrl}
      authToken={authToken}
      onSelectGraph={undefined}
      onGraphCreate={undefined}
      onGraphUpdate={undefined}
      showCreateDialog={showCreateDialog}
      setShowCreateDialog={setShowCreateDialog}
    />
  );
}
