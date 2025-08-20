"use client";

import React, { createContext, useContext, useState } from "react";
import { Breadcrumb } from "@/components/types";

interface HeaderContextType {
  customBreadcrumbs: Breadcrumb[] | null;
  rightContent: React.ReactNode | null;
  setCustomBreadcrumbs: (breadcrumbs: Breadcrumb[] | null) => void;
  setRightContent: (content: React.ReactNode | null) => void;
}

const HeaderContext = createContext<HeaderContextType | undefined>(undefined);

export function HeaderProvider({ children }: { children: React.ReactNode }) {
  const [customBreadcrumbs, setCustomBreadcrumbs] = useState<Breadcrumb[] | null>(null);
  const [rightContent, setRightContent] = useState<React.ReactNode | null>(null);

  return (
    <HeaderContext.Provider
      value={{
        customBreadcrumbs,
        rightContent,
        setCustomBreadcrumbs,
        setRightContent,
      }}
    >
      {children}
    </HeaderContext.Provider>
  );
}

export function useHeader() {
  const context = useContext(HeaderContext);
  if (context === undefined) {
    throw new Error("useHeader must be used within a HeaderProvider");
  }
  return context;
}
