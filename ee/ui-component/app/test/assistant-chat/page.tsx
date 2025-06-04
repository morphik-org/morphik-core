"use client";

import { useState } from "react";
import AssistantChatSection from "@/components/chat/AssistantChatSection";
import { ThemeProvider } from "@/components/theme-provider";

export default function AssistantChatTestPage() {
  const [apiBaseUrl] = useState("http://localhost:8000");
  const [authToken] = useState<string | null>(null);

  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
      <div className="h-screen w-full bg-background">
        <AssistantChatSection
          apiBaseUrl={apiBaseUrl}
          authToken={authToken}
          onChatSubmit={(query, options) => {
            console.log("Chat submitted:", { query, options });
          }}
        />
      </div>
    </ThemeProvider>
  );
}
