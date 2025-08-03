"use client";

export const dynamic = "force-dynamic";

import { useEffect } from "react";
import SearchV2Section from "@/components/search/SearchV2Section";
import { useMorphik } from "@/contexts/morphik-context";
import { useHeader } from "@/contexts/header-context";

export default function SearchV2Page() {
  const { apiBaseUrl, authToken } = useMorphik();
  const { setCustomBreadcrumbs } = useHeader();

  useEffect(() => {
    const breadcrumbs = [
      { label: "Home", href: "/" },
      { label: "Search", href: "/search" },
      { label: "Search V2" }
    ];
    setCustomBreadcrumbs(breadcrumbs);

    return () => {
      setCustomBreadcrumbs(null);
    };
  }, [setCustomBreadcrumbs]);

  return (
    <div className="-m-4 md:-m-6">
      <SearchV2Section apiBaseUrl={apiBaseUrl} authToken={authToken} />
    </div>
  );
}
