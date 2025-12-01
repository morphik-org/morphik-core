"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

interface ResizablePanelGroupProps {
  children: React.ReactNode;
  direction?: "horizontal" | "vertical";
  className?: string;
}

interface ResizablePanelProps {
  children: React.ReactNode;
  defaultSize?: number;
  minSize?: number;
  maxSize?: number;
  className?: string;
}

interface ResizableHandleProps {
  className?: string;
  withHandle?: boolean;
}

interface PanelContextValue {
  sizes: number[];
  setSizes: React.Dispatch<React.SetStateAction<number[]>>;
  direction: "horizontal" | "vertical";
  panelCount: number;
  registerPanel: () => number;
}

const PanelContext = React.createContext<PanelContextValue | null>(null);

export function ResizablePanelGroup({ children, direction = "horizontal", className }: ResizablePanelGroupProps) {
  const [sizes, setSizes] = React.useState<number[]>([]);
  const panelCountRef = React.useRef(0);
  const registeredRef = React.useRef(false);

  const registerPanel = React.useCallback(() => {
    return panelCountRef.current++;
  }, []);

  // Initialize sizes once we know how many panels we have
  React.useEffect(() => {
    if (!registeredRef.current) {
      registeredRef.current = true;
      // Count Panel children
      const panelChildren = React.Children.toArray(children).filter(
        child =>
          React.isValidElement(child) && (child.type as { displayName?: string })?.displayName === "ResizablePanel"
      );
      const count = panelChildren.length;
      if (count > 0 && sizes.length === 0) {
        // Check for defaultSize props
        const defaultSizes = panelChildren.map(child => {
          if (React.isValidElement(child)) {
            return (child.props as ResizablePanelProps).defaultSize ?? 100 / count;
          }
          return 100 / count;
        });
        setSizes(defaultSizes);
      }
    }
  }, [children, sizes.length]);

  return (
    <PanelContext.Provider value={{ sizes, setSizes, direction, panelCount: panelCountRef.current, registerPanel }}>
      <div className={cn("flex h-full w-full", direction === "horizontal" ? "flex-row" : "flex-col", className)}>
        {children}
      </div>
    </PanelContext.Provider>
  );
}

export function ResizablePanel({ children, defaultSize, minSize = 10, maxSize = 90, className }: ResizablePanelProps) {
  const context = React.useContext(PanelContext);
  const indexRef = React.useRef<number | null>(null);

  if (indexRef.current === null && context) {
    indexRef.current = context.registerPanel();
  }

  const index = indexRef.current ?? 0;
  const size = context?.sizes[index] ?? defaultSize ?? 50;

  return (
    <div
      className={cn("overflow-hidden", className)}
      style={{
        [context?.direction === "horizontal" ? "width" : "height"]: `${size}%`,
        flexShrink: 0,
      }}
      data-min-size={minSize}
      data-max-size={maxSize}
    >
      {children}
    </div>
  );
}

ResizablePanel.displayName = "ResizablePanel";

export function ResizableHandle({ className, withHandle = true }: ResizableHandleProps) {
  const context = React.useContext(PanelContext);
  const handleRef = React.useRef<HTMLDivElement>(null);
  const isDragging = React.useRef(false);

  // Determine which panels this handle is between based on DOM position
  const getAdjacentPanelIndices = React.useCallback(() => {
    if (!handleRef.current) return { leftIndex: 0, rightIndex: 1 };

    const container = handleRef.current.parentElement;
    if (!container) return { leftIndex: 0, rightIndex: 1 };

    const children = Array.from(container.children);
    const handlePosition = children.indexOf(handleRef.current);

    // Count panels before and after the handle
    let leftIndex = 0;
    let rightIndex = 1;
    let panelCount = 0;

    for (let i = 0; i < children.length; i++) {
      const child = children[i] as HTMLElement;
      if (child.hasAttribute("data-min-size")) {
        if (i < handlePosition) {
          leftIndex = panelCount;
        } else if (i > handlePosition) {
          rightIndex = panelCount;
          break;
        }
        panelCount++;
      }
    }

    return { leftIndex, rightIndex };
  }, []);

  const handleMouseDown = React.useCallback(() => {
    if (!context) return;
    isDragging.current = true;
    document.body.style.cursor = context.direction === "horizontal" ? "col-resize" : "row-resize";
    document.body.style.userSelect = "none";
  }, [context]);

  React.useEffect(() => {
    if (!context) return;

    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging.current || !handleRef.current) return;

      const container = handleRef.current.parentElement;
      if (!container) return;

      const { leftIndex, rightIndex } = getAdjacentPanelIndices();

      // Get panel elements to read their min/max constraints
      const panels = Array.from(container.children).filter(child =>
        (child as HTMLElement).hasAttribute("data-min-size")
      ) as HTMLElement[];

      const leftPanel = panels[leftIndex];
      const rightPanel = panels[rightIndex];

      if (!leftPanel || !rightPanel) return;

      const leftMinSize = parseFloat(leftPanel.getAttribute("data-min-size") || "10");
      const leftMaxSize = parseFloat(leftPanel.getAttribute("data-max-size") || "90");
      const rightMinSize = parseFloat(rightPanel.getAttribute("data-min-size") || "10");
      const rightMaxSize = parseFloat(rightPanel.getAttribute("data-max-size") || "90");

      const containerRect = container.getBoundingClientRect();
      const containerSize = context.direction === "horizontal" ? containerRect.width : containerRect.height;
      const currentPos = context.direction === "horizontal" ? e.clientX : e.clientY;
      const startContainerPos = context.direction === "horizontal" ? containerRect.left : containerRect.top;

      // Calculate sum of sizes of panels before the left panel
      let sizeBeforeLeft = 0;
      for (let i = 0; i < leftIndex; i++) {
        sizeBeforeLeft += context.sizes[i] || 0;
      }

      // Calculate position as percentage and adjust for panels before
      const posPercent = ((currentPos - startContainerPos) / containerSize) * 100;

      // Calculate new sizes for left and right panels
      const totalForBothPanels = (context.sizes[leftIndex] || 50) + (context.sizes[rightIndex] || 50);
      let newLeftSize = posPercent - sizeBeforeLeft;
      let newRightSize =
        totalForBothPanels - newLeftSize + sizeBeforeLeft + (context.sizes[leftIndex] || 50) - posPercent;

      // Simplified: just split the position between the two adjacent panels
      newLeftSize = posPercent - sizeBeforeLeft;
      newRightSize = totalForBothPanels - newLeftSize;

      // Apply constraints from both panels
      newLeftSize = Math.max(leftMinSize, Math.min(leftMaxSize, newLeftSize));
      newRightSize = Math.max(rightMinSize, Math.min(rightMaxSize, newRightSize));

      // Ensure they still sum to totalForBothPanels
      const currentTotal = newLeftSize + newRightSize;
      if (Math.abs(currentTotal - totalForBothPanels) > 0.1) {
        // Adjust to maintain total
        if (newLeftSize === leftMinSize || newLeftSize === leftMaxSize) {
          newRightSize = totalForBothPanels - newLeftSize;
          newRightSize = Math.max(rightMinSize, Math.min(rightMaxSize, newRightSize));
        } else {
          newLeftSize = totalForBothPanels - newRightSize;
          newLeftSize = Math.max(leftMinSize, Math.min(leftMaxSize, newLeftSize));
        }
      }

      const newSizes = [...context.sizes];
      newSizes[leftIndex] = newLeftSize;
      newSizes[rightIndex] = newRightSize;

      context.setSizes(newSizes);
    };

    const handleMouseUp = () => {
      isDragging.current = false;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [context, getAdjacentPanelIndices]);

  const isHorizontal = context?.direction === "horizontal";

  return (
    <div
      ref={handleRef}
      onMouseDown={handleMouseDown}
      className={cn(
        "relative flex items-center justify-center bg-border",
        isHorizontal
          ? "w-px cursor-col-resize hover:w-1 hover:bg-primary/50 active:bg-primary"
          : "h-px cursor-row-resize hover:h-1 hover:bg-primary/50 active:bg-primary",
        "transition-all duration-150",
        className
      )}
    >
      {withHandle && (
        <div
          className={cn(
            "z-10 flex items-center justify-center rounded-sm border bg-border",
            isHorizontal ? "h-8 w-3" : "h-3 w-8"
          )}
        >
          <div className={cn("rounded-full bg-muted-foreground/50", isHorizontal ? "h-4 w-0.5" : "h-0.5 w-4")} />
        </div>
      )}
    </div>
  );
}
