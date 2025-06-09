"use client";

import React, { useState, useCallback, useRef, useEffect, useMemo } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Upload,
  ZoomIn,
  ZoomOut,
  RotateCw,
  ChevronLeft,
  ChevronRight,
  FileText,
  Download,
  Maximize2,
  User,
  Cpu,
} from "lucide-react";
import { cn } from "@/lib/utils";

// Configure PDF.js worker - use CDN for reliability
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

interface PDFViewerProps {
  apiBaseUrl?: string;
  authToken?: string | null;
}

interface PDFState {
  file: File | null;
  currentPage: number;
  totalPages: number;
  scale: number;
  rotation: number;
  pdfDataUrl: string | null;
  controlMode: "manual" | "api"; // New mode toggle
}

interface ZoomBounds {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
export function PDFViewer({ apiBaseUrl, authToken }: PDFViewerProps) {
  const [pdfState, setPdfState] = useState<PDFState>({
    file: null,
    currentPage: 1,
    totalPages: 0,
    scale: 1.0,
    rotation: 0,
    pdfDataUrl: null,
    controlMode: "manual", // Default to manual control
  });

  const [isLoading, setIsLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [zoomBounds, setZoomBounds] = useState<ZoomBounds>({});
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pdfContainerRef = useRef<HTMLDivElement>(null);

  // Memoize PDF options to prevent unnecessary reloads
  const pdfOptions = useMemo(
    () => ({
      cMapUrl: `//unpkg.com/pdfjs-dist@${pdfjs.version}/cmaps/`,
      cMapPacked: true,
      standardFontDataUrl: `//unpkg.com/pdfjs-dist@${pdfjs.version}/standard_fonts/`,
    }),
    []
  );

  // Handle PDF load success
  const onDocumentLoadSuccess = useCallback(({ numPages }: { numPages: number }) => {
    setPdfState(prev => ({
      ...prev,
      totalPages: numPages,
      currentPage: 1,
    }));
    setIsLoading(false);
  }, []);

  // Handle PDF load error
  const onDocumentLoadError = useCallback(
    (error: Error) => {
      console.error("Error loading PDF:", error);
      console.error("PDF.js worker src:", pdfjs.GlobalWorkerOptions.workerSrc);
      console.error("PDF file URL:", pdfState.pdfDataUrl);
      setIsLoading(false);
    },
    [pdfState.pdfDataUrl]
  );

  // Handle file upload
  const handleFileUpload = useCallback(async (file: File) => {
    if (!file || file.type !== "application/pdf") {
      console.error("Please select a valid PDF file");
      return;
    }

    setIsLoading(true);
    try {
      // Create object URL for the PDF
      const pdfDataUrl = URL.createObjectURL(file);

      setPdfState(prev => ({
        ...prev,
        file,
        pdfDataUrl,
        currentPage: 1,
        totalPages: 0, // Will be set in onDocumentLoadSuccess
        scale: 1.0,
        rotation: 0,
      }));
    } catch (error) {
      console.error("Error loading PDF:", error);
      setIsLoading(false);
    }
  }, []);

  // Handle drag and drop
  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(false);

      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    },
    [handleFileUpload]
  );

  // PDF Controls
  const goToPage = useCallback(
    (page: number) => {
      if (page >= 1 && page <= pdfState.totalPages) {
        setPdfState(prev => ({ ...prev, currentPage: page }));
      }
    },
    [pdfState.totalPages]
  );

  const nextPage = useCallback(() => {
    goToPage(pdfState.currentPage + 1);
  }, [pdfState.currentPage, goToPage]);

  const prevPage = useCallback(() => {
    goToPage(pdfState.currentPage - 1);
  }, [pdfState.currentPage, goToPage]);

  const zoomIn = useCallback(() => {
    setPdfState(prev => ({ ...prev, scale: Math.min(prev.scale * 1.2, 3.0) }));
  }, []);

  const zoomOut = useCallback(() => {
    setPdfState(prev => ({ ...prev, scale: Math.max(prev.scale / 1.2, 0.5) }));
  }, []);

  const rotate = useCallback(() => {
    setPdfState(prev => ({ ...prev, rotation: (prev.rotation + 90) % 360 }));
  }, []);

  const resetZoom = useCallback(() => {
    setPdfState(prev => ({ ...prev, scale: 1.0 }));
  }, []);

  // Mode toggle functions
  const toggleControlMode = useCallback(() => {
    setPdfState(prev => ({
      ...prev,
      controlMode: prev.controlMode === "manual" ? "api" : "manual",
    }));
  }, []);

  // Zoom to specific bounds (0-1000 relative coordinates)
  const zoomToY = useCallback((bounds: { top: number; bottom: number }) => {
    const container = pdfContainerRef.current;
    if (!container) return;

    console.log("zoomToY called with bounds:", bounds);

    // Convert 0-1000 bounds to relative (0-1) coordinates
    const relativeTop = bounds.top / 1000;
    const relativeBottom = bounds.bottom / 1000;
    const relativeHeight = relativeBottom - relativeTop;
    const relativeCenter = (relativeTop + relativeBottom) / 2;

    console.log("Relative coords:", { relativeTop, relativeBottom, relativeHeight, relativeCenter });

    // Calculate scale to fit the bounds height in the container
    const containerHeight = container.clientHeight;
    const pdfPageHeight = 842; // Standard PDF page height (A4 aspect ratio)
    const boundsHeightPixels = relativeHeight * pdfPageHeight;
    const newScale = Math.max(0.1, containerHeight / boundsHeightPixels);

    console.log("Scale calculation:", { containerHeight, pdfPageHeight, boundsHeightPixels, newScale });

    setPdfState(prev => ({ ...prev, scale: newScale }));

    // Find the scroll container - try multiple selectors
    setTimeout(() => {
      const scrollContainers = [
        container.closest("[data-radix-scroll-area-viewport]"),
        container.closest(".scroll-area-viewport"),
        container.closest('[role="region"]'),
        document.querySelector("[data-radix-scroll-area-viewport]"),
      ].filter(Boolean);

      console.log("Found scroll containers:", scrollContainers.length);

      if (scrollContainers.length > 0) {
        const scrollArea = scrollContainers[0] as HTMLElement;

        // Calculate scroll position to center the bounds
        const scaledPageHeight = pdfPageHeight * newScale;
        const centerPositionPixels = relativeCenter * scaledPageHeight;
        const targetScrollTop = centerPositionPixels - containerHeight / 2;

        console.log("Scroll calculation:", { scaledPageHeight, centerPositionPixels, targetScrollTop });
        console.log("Current scroll top:", scrollArea.scrollTop);

        scrollArea.scrollTop = Math.max(0, targetScrollTop);

        console.log("New scroll top:", scrollArea.scrollTop);
      } else {
        console.warn("No scroll container found");
      }
    }, 100);

    setZoomBounds(prev => ({
      ...prev,
      y: relativeTop * pdfPageHeight,
      height: relativeHeight * pdfPageHeight,
    }));
  }, []);

  const zoomToX = useCallback((bounds: { left: number; right: number }) => {
    const container = pdfContainerRef.current;
    if (!container) return;

    console.log("zoomToX called with bounds:", bounds);

    // Convert 0-1000 bounds to relative (0-1) coordinates
    const relativeLeft = bounds.left / 1000;
    const relativeRight = bounds.right / 1000;
    const relativeWidth = relativeRight - relativeLeft;
    const relativeCenter = (relativeLeft + relativeRight) / 2;

    console.log("Relative coords:", { relativeLeft, relativeRight, relativeWidth, relativeCenter });

    // Calculate scale to fit the bounds width in the container
    const containerWidth = container.clientWidth;
    const pdfPageWidth = 595; // Standard PDF page width (A4 aspect ratio)
    const boundsWidthPixels = relativeWidth * pdfPageWidth;
    const newScale = Math.max(0.1, containerWidth / boundsWidthPixels);

    console.log("Scale calculation:", { containerWidth, pdfPageWidth, boundsWidthPixels, newScale });

    setPdfState(prev => ({ ...prev, scale: newScale }));

    // Find the scroll container - try multiple selectors
    setTimeout(() => {
      const scrollContainers = [
        container.closest("[data-radix-scroll-area-viewport]"),
        container.closest(".scroll-area-viewport"),
        container.closest('[role="region"]'),
        document.querySelector("[data-radix-scroll-area-viewport]"),
      ].filter(Boolean);

      console.log("Found scroll containers:", scrollContainers.length);

      if (scrollContainers.length > 0) {
        const scrollArea = scrollContainers[0] as HTMLElement;

        // Calculate scroll position to center the bounds
        const scaledPageWidth = pdfPageWidth * newScale;
        const centerPositionPixels = relativeCenter * scaledPageWidth;
        const targetScrollLeft = centerPositionPixels - containerWidth / 2;

        console.log("Scroll calculation:", { scaledPageWidth, centerPositionPixels, targetScrollLeft });
        console.log("Current scroll left:", scrollArea.scrollLeft);

        scrollArea.scrollLeft = Math.max(0, targetScrollLeft);

        console.log("New scroll left:", scrollArea.scrollLeft);
      } else {
        console.warn("No scroll container found");
      }
    }, 100);

    setZoomBounds(prev => ({
      ...prev,
      x: relativeLeft * pdfPageWidth,
      width: relativeWidth * pdfPageWidth,
    }));
  }, []);

  // API endpoint handlers (these will be called by external API requests)
  useEffect(() => {
    if (pdfState.file && pdfState.controlMode === "api") {
      console.log("Registering PDF viewer controls in API mode...");
      // Register global PDF viewer control functions
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (window as any).pdfViewerControls = {
        changePage: (page: number) => {
          console.log("PDF viewer changePage called with:", page);
          goToPage(page);
        },
        zoomToY: (bounds: { top: number; bottom: number }) => {
          console.log("PDF viewer zoomToY called with:", bounds);
          zoomToY(bounds);
        },
        zoomToX: (bounds: { left: number; right: number }) => {
          console.log("PDF viewer zoomToX called with:", bounds);
          zoomToX(bounds);
        },
        getCurrentState: () => {
          console.log("PDF viewer getCurrentState called");
          return pdfState;
        },
        getMode: () => pdfState.controlMode,
      };
      console.log("PDF viewer controls registered successfully");
    } else if (pdfState.controlMode === "manual") {
      console.log("Unregistering PDF viewer controls (manual mode)");
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      delete (window as any).pdfViewerControls;
    }

    return () => {
      if (pdfState.controlMode === "api") {
        console.log("Unregistering PDF viewer controls");
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (window as any).pdfViewerControls;
      }
    };
  }, [goToPage, zoomToY, zoomToX, pdfState.file, pdfState.controlMode, pdfState]);

  if (!pdfState.file) {
    return (
      <div className="flex h-full flex-col bg-white dark:bg-slate-900">
        {/* Clean Header */}
        <div className="border-b border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-900">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <FileText className="h-5 w-5 text-slate-600 dark:text-slate-400" />
              <h2 className="text-lg font-medium text-slate-900 dark:text-slate-100">PDF Viewer</h2>
            </div>
          </div>
        </div>

        {/* Clean Upload Area */}
        <div className="flex flex-1 items-center justify-center p-8">
          <div className="w-full max-w-md">
            <Card
              className={cn(
                "border-2 border-dashed p-8 text-center transition-colors",
                dragActive
                  ? "border-slate-400 bg-slate-50 dark:border-slate-500 dark:bg-slate-800"
                  : "border-slate-300 hover:border-slate-400 dark:border-slate-600 dark:hover:border-slate-500"
              )}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <div className="flex flex-col items-center gap-4">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-slate-100 dark:bg-slate-800">
                  <Upload className="h-6 w-6 text-slate-600 dark:text-slate-400" />
                </div>
                <div>
                  <h3 className="text-lg font-medium text-slate-900 dark:text-slate-100">
                    {dragActive ? "Drop your PDF here" : "Upload PDF"}
                  </h3>
                  <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">Drag and drop or click to browse</p>
                </div>
                <Button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isLoading}
                  className="bg-slate-900 text-white hover:bg-slate-800 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-slate-200"
                >
                  {isLoading ? "Loading..." : "Choose File"}
                </Button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="application/pdf"
                  onChange={e => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
                  className="hidden"
                />
              </div>
            </Card>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col bg-white dark:bg-slate-900">
      {/* Clean Header */}
      <div className="border-b border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-900">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FileText className="h-5 w-5 text-slate-600 dark:text-slate-400" />
            <h2 className="text-lg font-medium text-slate-900 dark:text-slate-100">{pdfState.file.name}</h2>
          </div>

          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={() => fileInputRef.current?.click()}>
              <Upload className="mr-2 h-4 w-4" />
              New PDF
            </Button>
            <Button variant="outline" size="sm">
              <Download className="mr-2 h-4 w-4" />
              Download
            </Button>
          </div>
        </div>
      </div>

      {/* PDF Display Area */}
      <div className="relative flex-1 overflow-hidden">
        <ScrollArea className="h-full w-full">
          <div
            ref={pdfContainerRef}
            className="flex justify-center p-4 pb-24"
            style={{
              transform: `scale(${pdfState.scale}) rotate(${pdfState.rotation}deg)`,
              transformOrigin: "center top",
            }}
          >
            {pdfState.pdfDataUrl && (
              <div className="border border-slate-200 bg-white shadow-lg dark:border-slate-700 dark:bg-slate-800">
                <Document
                  file={pdfState.pdfDataUrl}
                  onLoadSuccess={onDocumentLoadSuccess}
                  onLoadError={onDocumentLoadError}
                  options={pdfOptions}
                  loading={
                    <div className="flex h-[800px] w-[600px] items-center justify-center bg-white p-8 text-slate-500 dark:bg-slate-800 dark:text-slate-400">
                      <div className="text-center">
                        <FileText className="mx-auto mb-4 h-16 w-16 animate-pulse" />
                        <p>Loading PDF...</p>
                      </div>
                    </div>
                  }
                  error={
                    <div className="flex h-[800px] w-[600px] items-center justify-center bg-white p-8 text-red-500 dark:bg-slate-800 dark:text-red-400">
                      <div className="text-center">
                        <FileText className="mx-auto mb-4 h-16 w-16" />
                        <p>Error loading PDF</p>
                        <p className="mt-2 text-sm">Please try uploading a different file</p>
                      </div>
                    </div>
                  }
                >
                  <Page
                    pageNumber={pdfState.currentPage}
                    loading={
                      <div className="flex h-[800px] w-[600px] items-center justify-center bg-slate-100 dark:bg-slate-700">
                        <div className="text-slate-500 dark:text-slate-400">Loading page...</div>
                      </div>
                    }
                    error={
                      <div className="flex h-[800px] w-[600px] items-center justify-center bg-slate-100 dark:bg-slate-700">
                        <div className="text-red-500 dark:text-red-400">Error loading page</div>
                      </div>
                    }
                    width={600}
                    renderTextLayer={true}
                    renderAnnotationLayer={true}
                  />
                </Document>
              </div>
            )}
          </div>
        </ScrollArea>

        {/* Bottom Floating Control Bar */}
        <div className="absolute bottom-4 left-1/2 z-10 -translate-x-1/2 transform">
          <div className="flex items-center gap-4 border border-slate-200 bg-white px-4 py-2 shadow-lg dark:border-slate-700 dark:bg-slate-900">
            {/* Control Mode Toggle */}
            <div
              onClick={toggleControlMode}
              className={cn(
                "flex cursor-pointer items-center gap-2 rounded-full px-3 py-1.5 text-sm font-medium transition-colors",
                pdfState.controlMode === "manual"
                  ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900"
                  : "bg-blue-600 text-white"
              )}
            >
              {pdfState.controlMode === "manual" ? <User className="h-4 w-4" /> : <Cpu className="h-4 w-4" />}
              {pdfState.controlMode === "manual" ? "Manual" : "API"}
            </div>

            {/* Page Navigation */}
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={prevPage}
                disabled={pdfState.currentPage <= 1 || pdfState.controlMode === "api"}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>

              <div className="flex items-center gap-2">
                <Input
                  type="number"
                  value={pdfState.currentPage}
                  onChange={e => goToPage(parseInt(e.target.value) || 1)}
                  className="w-16 text-center"
                  min={1}
                  max={pdfState.totalPages}
                  disabled={pdfState.controlMode === "api"}
                />
                <span className="text-sm text-slate-500">of {pdfState.totalPages}</span>
              </div>

              <Button
                variant="outline"
                size="sm"
                onClick={nextPage}
                disabled={pdfState.currentPage >= pdfState.totalPages || pdfState.controlMode === "api"}
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>

            {/* Zoom Controls */}
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" onClick={zoomOut} disabled={pdfState.controlMode === "api"}>
                <ZoomOut className="h-4 w-4" />
              </Button>

              <Button
                variant="outline"
                size="sm"
                onClick={resetZoom}
                disabled={pdfState.controlMode === "api"}
                className="min-w-16"
              >
                {Math.round(pdfState.scale * 100)}%
              </Button>

              <Button variant="outline" size="sm" onClick={zoomIn} disabled={pdfState.controlMode === "api"}>
                <ZoomIn className="h-4 w-4" />
              </Button>
            </div>

            {/* Additional Controls */}
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" onClick={rotate} disabled={pdfState.controlMode === "api"}>
                <RotateCw className="h-4 w-4" />
              </Button>

              <Button variant="outline" size="sm" disabled={pdfState.controlMode === "api"}>
                <Maximize2 className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="application/pdf"
        onChange={e => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
        className="hidden"
      />
    </div>
  );
}
