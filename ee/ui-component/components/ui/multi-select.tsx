"use client";

import * as React from "react";
import { X, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export interface MultiSelectOption {
  label: string;
  value: string;
}

interface MultiSelectProps {
  options: MultiSelectOption[];
  selected: string[];
  onChange: (selected: string[]) => void;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
}

export function MultiSelect({
  options,
  selected,
  onChange,
  placeholder = "Select items...",
  className,
  disabled = false,
}: MultiSelectProps) {
  const handleUnselect = (item: string) => {
    onChange(selected.filter(i => i !== item));
  };

  const handleCheckboxChange = (checked: boolean | "indeterminate", value: string) => {
    if (checked === true) {
      onChange([...selected, value]);
    } else {
      onChange(selected.filter(item => item !== value));
    }
  };

  const clearAll = () => {
    onChange([]);
  };

  return (
    <div className={cn("grid gap-2", className)}>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="outline"
            className={cn(
              "h-8 justify-between text-left font-normal",
              selected.length === 0 && "text-muted-foreground",
              disabled && "cursor-not-allowed opacity-50"
            )}
            disabled={disabled}
          >
            <div className="flex-1 truncate">
              {selected.length === 0 ? (
                placeholder
              ) : selected.length === 1 ? (
                <span
                  className="block max-w-[150px] truncate"
                  title={options.find(option => option.value === selected[0])?.label}
                >
                  {options.find(option => option.value === selected[0])?.label}
                </span>
              ) : (
                <span className="font-medium">{selected.length} selected</span>
              )}
            </div>
            <ChevronDown className="h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent className="max-h-64 w-56 overflow-auto" align="start">
          {/* Clear all option */}
          {selected.length > 0 && (
            <>
              <DropdownMenuItem className="cursor-pointer text-muted-foreground" onClick={clearAll}>
                <div className="flex w-full items-center space-x-2">
                  <X className="h-4 w-4" />
                  <span>Clear all ({selected.length})</span>
                </div>
              </DropdownMenuItem>
              <DropdownMenuSeparator />
            </>
          )}

          {/* Options */}
          {options.map(option => (
            <DropdownMenuItem key={option.value} className="cursor-pointer" onSelect={e => e.preventDefault()}>
              <div className="flex w-full items-center space-x-2">
                <Checkbox
                  checked={selected.includes(option.value)}
                  onCheckedChange={checked => handleCheckboxChange(checked, option.value)}
                />
                <span className="flex-1">{option.label}</span>
              </div>
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>

      {/* Selected items display - only show badges for single selection */}
      {selected.length === 1 && (
        <div className="flex flex-wrap gap-1">
          <Badge variant="secondary" className="max-w-[180px] text-xs">
            <span className="truncate" title={options.find(option => option.value === selected[0])?.label}>
              {options.find(option => option.value === selected[0])?.label}
            </span>
            <button
              className="ml-1 rounded-full outline-none ring-offset-background focus:ring-2 focus:ring-ring focus:ring-offset-2"
              onKeyDown={e => {
                if (e.key === "Enter") {
                  handleUnselect(selected[0]);
                }
              }}
              onMouseDown={e => {
                e.preventDefault();
                e.stopPropagation();
              }}
              onClick={() => handleUnselect(selected[0])}
            >
              <X className="h-3 w-3 text-muted-foreground hover:text-foreground" />
            </button>
          </Badge>
        </div>
      )}

      {/* For 2+ selections, show compact summary with clear button */}
      {selected.length >= 2 && (
        <div className="flex items-center justify-between">
          <Badge variant="outline" className="text-xs">
            {selected.length} items selected
          </Badge>
          <Button
            variant="ghost"
            size="sm"
            onClick={clearAll}
            className="h-6 px-2 text-xs text-muted-foreground hover:text-foreground"
          >
            Clear all
          </Button>
        </div>
      )}
    </div>
  );
}
