"use client";

export const dynamic = "force-dynamic";

import { useEffect } from "react";
import { SettingsSection } from "@/components/settings/SettingsSection";
import { useMorphik } from "@/contexts/morphik-context";
import { useHeader } from "@/contexts/header-context";

export default function SettingsPage() {
  const { authToken } = useMorphik();
  const { setCustomBreadcrumbs } = useHeader();

  // Set up breadcrumbs
  useEffect(() => {
    setCustomBreadcrumbs([{ label: "Home", href: "/" }, { label: "Settings" }]);

    return () => {
      setCustomBreadcrumbs(null);
    };
  }, [setCustomBreadcrumbs]);

  return <SettingsSection authToken={authToken} />;
}
