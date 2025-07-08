"use client";

export const dynamic = "force-dynamic";

import { useEffect } from "react";
import SearchSection from "@/components/search/SearchSection";
import { useMorphik } from "@/contexts/morphik-context";
import { useHeader } from "@/contexts/header-context";

export default function SearchPage() {
  const { apiBaseUrl, authToken } = useMorphik();
  const { setCustomBreadcrumbs } = useHeader();

  useEffect(() => {
    const breadcrumbs = [{ label: "Home", href: "/" }, { label: "Search" }];
    setCustomBreadcrumbs(breadcrumbs);

    return () => {
      setCustomBreadcrumbs(null);
    };
  }, [setCustomBreadcrumbs]);

  return <SearchSection apiBaseUrl={apiBaseUrl} authToken={authToken} onSearchSubmit={undefined} />;
}
