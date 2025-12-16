"use client";

export const dynamic = "force-dynamic";

import { useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { useMorphik } from "@/contexts/morphik-context";
import { useHeader } from "@/contexts/header-context";
import { PDFViewer } from "@/components/pdf/PDFViewer";

function PDFViewerContent() {
  const { apiBaseUrl, authToken } = useMorphik();
  const searchParams = useSearchParams();
  const { setCustomBreadcrumbs } = useHeader();

  const documentId = searchParams?.get("document") || null;

  // Update breadcrumbs
  useEffect(() => {
    const breadcrumbs = [
      { label: "Home", href: "/" },
      { label: "Documents", href: "/documents" },
      { label: "PDF Viewer" },
    ];

    setCustomBreadcrumbs(breadcrumbs);

    return () => {
      setCustomBreadcrumbs(null);
    };
  }, [setCustomBreadcrumbs]);

  return (
    <div className="h-full">
      <PDFViewer apiBaseUrl={apiBaseUrl} authToken={authToken} initialDocumentId={documentId || undefined} />
    </div>
  );
}

export default function PDFViewerPage() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <PDFViewerContent />
    </Suspense>
  );
}
