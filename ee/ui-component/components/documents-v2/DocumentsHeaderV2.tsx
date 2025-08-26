"use client";

import React from "react";
import { useDocumentsV2 } from "./DocumentsProviderV2";
import { Input } from "../ui/input";
import { Search } from "lucide-react";

export function DocumentsHeaderV2() {
  const { searchQuery, setSearchQuery } = useDocumentsV2();

  return (
    <div className="border-b px-4 py-3">
      {/* Search bar */}
      <div className="relative max-w-md">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 transform text-muted-foreground" />
        <Input
          type="text"
          placeholder="Search documents..."
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
          className="pl-10"
        />
      </div>
    </div>
  );
}
