"use client";

import React, { useState, useEffect } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { IconLink, IconX, IconLock } from "@tabler/icons-react";

interface ConnectionInputProps {
  value?: string;
  onChange?: (uri: string) => void;
  onClear?: () => void;
  placeholder?: string;
  showProtocolSelector?: boolean;
}

export function ConnectionInput({
  value = "",
  onChange,
  onClear,
  placeholder = "localhost:8000 or morphik://...",
  showProtocolSelector = true,
}: ConnectionInputProps) {
  const [useSSL, setUseSSL] = useState(false);
  const [hostInput, setHostInput] = useState("");
  const [isEditing, setIsEditing] = useState(false);

  // Parse the current value to determine SSL and host
  useEffect(() => {
    if (value && !isEditing) {
      // Check if it's a morphik:// URI
      if (value.startsWith("morphik://")) {
        // For morphik:// URIs, extract the base URI without protocol modifier
        let baseUri = value;

        // Check if it has explicit protocol and strip it for display
        if (value.includes("@https:")) {
          setUseSSL(true);
          baseUri = value.replace("@https:", "@");
        } else if (value.includes("@http:")) {
          setUseSSL(false);
          baseUri = value.replace("@http:", "@");
        } else {
          // No protocol specified - default to HTTP (not HTTPS)
          setUseSSL(false);
        }

        setHostInput(baseUri);
      } else if (value.startsWith("https://")) {
        setUseSSL(true);
        setHostInput(value.replace("https://", ""));
      } else if (value.startsWith("http://")) {
        setUseSSL(false);
        setHostInput(value.replace("http://", ""));
      } else {
        // Plain host
        setHostInput(value);
        setUseSSL(false);
      }
    }
  }, [value, isEditing]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (hostInput.trim() && onChange) {
      const input = hostInput.trim();

      // If it's a morphik:// URI, add protocol based on SSL checkbox
      if (input.startsWith("morphik://")) {
        // Parse the URI to add protocol modifier
        const match = input.match(/^(morphik:\/\/[^@]+)@(.+)$/);
        if (match) {
          const [, prefix, host] = match;
          // Add protocol modifier based on SSL checkbox
          const protocol = useSSL ? "https:" : "http:";
          onChange(`${prefix}@${protocol}${host}`);
        } else {
          // Malformed morphik URI, pass through as-is
          onChange(input);
        }
      } else {
        // Strip any protocol the user might have added
        let cleanHost = input;
        if (input.startsWith("https://")) {
          cleanHost = input.substring(8);
        } else if (input.startsWith("http://")) {
          cleanHost = input.substring(7);
        }

        // Construct URL with selected protocol from checkbox
        const protocol = useSSL ? "https" : "http";
        onChange(`${protocol}://${cleanHost}`);
      }
      setIsEditing(false);
    }
  };

  const handleClear = () => {
    setHostInput("");
    setIsEditing(false);
    onClear?.();
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-2">
      <div className="flex items-center gap-2">
        <Input
          type="text"
          placeholder={placeholder}
          value={hostInput}
          onChange={e => {
            const value = e.target.value;
            setHostInput(value);
            setIsEditing(true);

            // Auto-detect SSL preference if user types a URL with protocol
            if (!value.startsWith("morphik://")) {
              if (value.startsWith("https://")) {
                setUseSSL(true);
              } else if (value.startsWith("http://")) {
                setUseSSL(false);
              }
            }
          }}
          className="h-7 flex-1 text-xs"
        />

        <div className="flex items-center gap-1">
          <Button
            type="submit"
            size="sm"
            variant="ghost"
            className="h-7 w-7 p-0"
            title="Connect"
            disabled={!hostInput.trim()}
          >
            <IconLink className="h-3 w-3" />
          </Button>

          {value && (
            <Button
              type="button"
              size="sm"
              variant="ghost"
              className="h-7 w-7 p-0"
              title="Clear connection"
              onClick={handleClear}
            >
              <IconX className="h-3 w-3" />
            </Button>
          )}
        </div>
      </div>

      {showProtocolSelector && (
        <div className="flex items-center gap-2">
          <Checkbox
            id="ssl-mode"
            checked={useSSL}
            onCheckedChange={checked => setUseSSL(checked as boolean)}
            className="h-3 w-3"
          />
          <label htmlFor="ssl-mode" className="flex cursor-pointer select-none items-center gap-1 text-[10px]">
            <IconLock className="h-3 w-3" />
            <span>Use HTTPS (SSL)</span>
          </label>
        </div>
      )}
    </form>
  );
}
