"use client";

import { useEffect, useRef } from "react";
import LogsSection, { LogsSectionRef } from "@/components/logs/LogsSection";
import { useMorphik } from "@/contexts/morphik-context";
import { useHeader } from "@/contexts/header-context";
import { Button } from "@/components/ui/button";
import { RefreshCw } from "lucide-react";

export const dynamic = "force-dynamic";

export default function LogsPage() {
  const { apiBaseUrl, authToken } = useMorphik();
  const { setRightContent, setCustomBreadcrumbs } = useHeader();
  const logsSectionRef = useRef<LogsSectionRef>(null);

  // Set up breadcrumbs
  useEffect(() => {
    setCustomBreadcrumbs([{ label: "Home", href: "/" }, { label: "Logs" }]);

    return () => {
      setCustomBreadcrumbs(null);
    };
  }, [setCustomBreadcrumbs]);

  // Set up header controls
  useEffect(() => {
    const rightContent = (
      <Button
        variant="outline"
        size="sm"
        onClick={() => {
          logsSectionRef.current?.handleRefresh();
        }}
        title="Refresh logs"
      >
        <RefreshCw className="h-4 w-4" />
        <span className="ml-1">Refresh</span>
      </Button>
    );

    setRightContent(rightContent);

    return () => {
      setRightContent(null);
    };
  }, [setRightContent]);

  return <LogsSection ref={logsSectionRef} apiBaseUrl={apiBaseUrl} authToken={authToken} />;
}
