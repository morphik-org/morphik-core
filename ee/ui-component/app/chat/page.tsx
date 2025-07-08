"use client";

import { useEffect } from "react";
import ChatSection from "@/components/chat/ChatSection";
import { useMorphik } from "@/contexts/morphik-context";
import { useHeader } from "@/contexts/header-context";

export const dynamic = "force-dynamic";

export default function ChatPage() {
  const { apiBaseUrl, authToken } = useMorphik();
  const { setCustomBreadcrumbs } = useHeader();

  useEffect(() => {
    const breadcrumbs = [{ label: "Home", href: "/" }, { label: "Chat" }];
    setCustomBreadcrumbs(breadcrumbs);

    return () => {
      setCustomBreadcrumbs(null);
    };
  }, [setCustomBreadcrumbs]);

  return <ChatSection apiBaseUrl={apiBaseUrl} authToken={authToken} onChatSubmit={undefined} />;
}
