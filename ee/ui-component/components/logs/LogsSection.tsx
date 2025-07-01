"use client";

import React, { useState } from "react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { useLogs } from "@/hooks/useLogs";
import { Button } from "@/components/ui/button";
import { RotateCw, Loader2 } from "lucide-react";

interface LogsSectionProps {
  apiBaseUrl: string;
  authToken: string | null;
}

export default function LogsSection({ apiBaseUrl, authToken }: LogsSectionProps) {
  const { logs, loading, error, refresh } = useLogs({ apiBaseUrl, authToken });
  const [expanded, setExpanded] = useState<Set<number>>(new Set());
  const [refreshing, setRefreshing] = useState(false);

  const toggle = (idx: number) => {
    setExpanded(prev => {
      const n = new Set(prev);
      if (n.has(idx)) n.delete(idx);
      else n.add(idx);
      return n;
    });
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await refresh();
    setRefreshing(false);
  };

  if (loading) return <p className="p-4 text-sm">Loading logs…</p>;
  if (error) return <p className="p-4 text-sm text-red-600">{error.message}</p>;

  return (
    <div className="h-full overflow-auto p-4 md:p-6">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-xl font-semibold">Logs</h2>
        <Button variant="ghost" size="icon" onClick={handleRefresh} disabled={refreshing} title="Refresh logs">
          {refreshing ? <Loader2 className="h-4 w-4 animate-spin" /> : <RotateCw className="h-4 w-4" />}
        </Button>
      </div>
      <Table className="text-xs md:text-sm">
        <TableHeader>
          <TableRow>
            <TableHead className="w-40">Time</TableHead>
            <TableHead>Operation</TableHead>
            <TableHead>Status</TableHead>
            <TableHead className="text-right">Duration&nbsp;(ms)</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {logs.length === 0 && (
            <TableRow>
              <TableCell colSpan={4} className="text-center text-muted-foreground">
                No logs found.
              </TableCell>
            </TableRow>
          )}
          {logs.map((log, idx) => (
            <>
              <TableRow
                key={`${log.timestamp}-${log.operation_type}`}
                className="cursor-pointer whitespace-nowrap hover:bg-muted/40 [&>td]:py-1"
                onClick={() => toggle(idx)}
              >
                <TableCell>{new Date(log.timestamp).toLocaleString()}</TableCell>
                <TableCell>{log.operation_type}</TableCell>
                <TableCell>
                  {log.status === "success" ? (
                    <Badge
                      variant="secondary"
                      className="bg-green-100 capitalize text-green-800 dark:bg-green-900/30 dark:text-green-300"
                    >
                      Success
                    </Badge>
                  ) : (
                    <Badge variant="destructive" className="capitalize">
                      Error
                    </Badge>
                  )}
                </TableCell>
                <TableCell className="text-right">{log.duration_ms.toFixed(2)}</TableCell>
              </TableRow>
              {expanded.has(idx) && (
                <TableRow className="bg-muted/20">
                  <TableCell colSpan={4} className="space-y-2 p-2 font-mono text-[10px] md:text-xs">
                    {log.error && <div className="text-red-600 dark:text-red-400">Error: {log.error}</div>}
                    {log.metadata && (
                      <pre className="whitespace-pre-wrap break-all">{JSON.stringify(log.metadata, null, 2)}</pre>
                    )}
                  </TableCell>
                </TableRow>
              )}
            </>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
