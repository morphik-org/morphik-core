"use client";

export const dynamic = "force-dynamic";

import { useEffect, Suspense, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useMorphik } from "@/contexts/morphik-context";
import { useHeader } from "@/contexts/header-context";
import { PDFViewer } from "@/components/pdf/PDFViewer";
import { PDFAPIService } from "@/components/pdf/PDFAPIService";
import { Button } from "@/components/ui/button";
import { MessageSquare } from "lucide-react";
import { cn } from "@/lib/utils";

function PDFViewerContent() {
  const { apiBaseUrl, authToken } = useMorphik();
  const searchParams = useSearchParams();
  const { setCustomBreadcrumbs, setRightContent } = useHeader();
  const [isChatOpen, setIsChatOpen] = useState(false);

  const documentId = searchParams?.get("document") || null;

  // Update breadcrumbs and header controls
  useEffect(() => {
    const breadcrumbs = [
      { label: "Home", href: "/" },
      { label: "Documents", href: "/documents" },
      { label: "PDF Viewer" },
    ];

    setCustomBreadcrumbs(breadcrumbs);

    // Set right content with Chat button
    const rightContent = (
      <Button
        variant="outline"
        size="sm"
        onClick={() => setIsChatOpen(!isChatOpen)}
        className={cn(isChatOpen && "bg-accent")}
      >
        <MessageSquare className="mr-2 h-4 w-4" />
        Chat
      </Button>
    );

    setRightContent(rightContent);

    return () => {
      setCustomBreadcrumbs(null);
      setRightContent(null);
    };
  }, [setCustomBreadcrumbs, setRightContent, isChatOpen]);

  return (
    <div className="h-full">
      <PDFAPIService>
        <PDFViewer
          apiBaseUrl={apiBaseUrl}
          authToken={authToken}
          initialDocumentId={documentId || undefined}
          chatOpen={isChatOpen}
          onChatToggle={setIsChatOpen}
        />
      </PDFAPIService>
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
