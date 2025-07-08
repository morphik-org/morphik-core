"use client";

export const dynamic = "force-dynamic";

import { useEffect } from "react";
import { ConnectorList } from "@/components/connectors/ConnectorList";
import { useMorphik } from "@/contexts/morphik-context";
import { useHeader } from "@/contexts/header-context";

export default function ConnectionsPage() {
  const { apiBaseUrl, authToken } = useMorphik();
  const { setCustomBreadcrumbs } = useHeader();

  // Set up breadcrumbs
  useEffect(() => {
    setCustomBreadcrumbs([{ label: "Home", href: "/" }, { label: "Connections" }]);

    return () => {
      setCustomBreadcrumbs(null);
    };
  }, [setCustomBreadcrumbs]);

  return <ConnectorList apiBaseUrl={apiBaseUrl} authToken={authToken} />;
}
