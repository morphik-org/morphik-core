"use client";

import React, { useState, createContext, useContext, useCallback } from "react";
import { useMorphik } from "@/contexts/morphik-context";
import { MorphikSidebar } from "@/components/morphik-sidebar";

// Create a context for chat state sharing
interface ChatContextType {
  activeChatId?: string;
  setActiveChatId: (id: string | undefined) => void;
  showChatView: boolean;
  setShowChatView: (show: boolean) => void;
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
  const [showChatView, setShowChatView] = useState(false);
  const [activeChatId, setActiveChatId] = useState<string | undefined>();

  // Memoize the setter functions to prevent unnecessary re-renders
  const setActiveChatIdMemo = useCallback((id: string | undefined) => {
    setActiveChatId(prev => prev !== id ? id : prev);
  }, []);

  const setShowChatViewMemo = useCallback((show: boolean) => {
    setShowChatView(prev => prev !== show ? show : prev);
  }, []);

  // Memoize the context value to prevent unnecessary re-renders
  const contextValue = React.useMemo(() => ({
    activeChatId,
    setActiveChatId: setActiveChatIdMemo,
    showChatView,
    setShowChatView: setShowChatViewMemo
  }), [activeChatId, showChatView, setActiveChatIdMemo, setShowChatViewMemo]);

  return (
    <ChatContext.Provider value={contextValue}>
      {children}
    </ChatContext.Provider>
  );
}

export function ConnectedSidebar() {
  const { connectionUri, updateConnectionUri, userProfile, onLogout, onProfileNavigate, onUpgradeClick } = useMorphik();
  const { showChatView, setShowChatView, activeChatId, setActiveChatId } = useChatContext();

  return (
    <MorphikSidebar
      variant="inset"
      showEditableUri={true}
      connectionUri={connectionUri}
      onUriChange={updateConnectionUri}
      userProfile={userProfile}
      onLogout={onLogout}
      onProfileNavigate={onProfileNavigate}
      onUpgradeClick={onUpgradeClick}
      showChatView={showChatView}
      onChatViewChange={setShowChatView}
      activeChatId={activeChatId}
      onChatSelect={setActiveChatId}
    />
  );
}
