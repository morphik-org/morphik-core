"use client";

import React, { createContext, useContext, useCallback, useState } from "react";

// Create a context for chat and settings state sharing
interface ChatContextType {
  activeChatId?: string;
  setActiveChatId: (id: string | undefined) => void;
  showChatView: boolean;
  setShowChatView: (show: boolean) => void;
  showSettingsView: boolean;
  setShowSettingsView: (show: boolean) => void;
  activeSettingsTab: string;
  setActiveSettingsTab: (tab: string) => void;
}

const ChatContext = createContext<ChatContextType | null>(null);

export function useChatContext() {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error("useChatContext must be used within a ChatProvider");
  }
  return context;
}

export function ChatProvider({ children }: { children: React.ReactNode }) {
  const [showChatView, setShowChatView] = useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    const onChatPage = window.location.pathname === "/chat";
    const manuallyHidden = sessionStorage.getItem("chatViewManuallyHidden") === "true";
    return onChatPage && !manuallyHidden;
  });
  const [activeChatId, setActiveChatId] = useState<string | undefined>();
  const [showSettingsView, setShowSettingsView] = useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    const onSettingsPage = window.location.pathname === "/settings";
    const manuallyHidden = sessionStorage.getItem("settingsViewManuallyHidden") === "true";
    return onSettingsPage && !manuallyHidden;
  });
  const [activeSettingsTab, setActiveSettingsTab] = useState("api-keys");

  const setActiveChatIdMemo = useCallback((id: string | undefined) => {
    setActiveChatId(prev => (prev !== id ? id : prev));
  }, []);

  const setShowChatViewMemo = useCallback((show: boolean) => {
    setShowChatView(prev => (prev !== show ? show : prev));
  }, []);

  const setShowSettingsViewMemo = useCallback((show: boolean) => {
    setShowSettingsView(prev => (prev !== show ? show : prev));
  }, []);

  const setActiveSettingsTabMemo = useCallback((tab: string) => {
    setActiveSettingsTab(prev => (prev !== tab ? tab : prev));
  }, []);

  const contextValue = React.useMemo(
    () => ({
      activeChatId,
      setActiveChatId: setActiveChatIdMemo,
      showChatView,
      setShowChatView: setShowChatViewMemo,
      showSettingsView,
      setShowSettingsView: setShowSettingsViewMemo,
      activeSettingsTab,
      setActiveSettingsTab: setActiveSettingsTabMemo,
    }),
    [
      activeChatId,
      showChatView,
      showSettingsView,
      activeSettingsTab,
      setActiveChatIdMemo,
      setShowChatViewMemo,
      setShowSettingsViewMemo,
      setActiveSettingsTabMemo,
    ]
  );

  return <ChatContext.Provider value={contextValue}>{children}</ChatContext.Provider>;
}
