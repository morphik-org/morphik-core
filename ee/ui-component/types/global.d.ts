export interface PDFViewerControls {
  changePage: (page: number) => void;
  zoomToY: (bounds: { top: number; bottom: number }) => void;
  zoomToX: (bounds: { left: number; right: number }) => void;
  getCurrentState: () => {
    file: File | null;
    currentPage: number;
    totalPages: number;
    scale: number;
    rotation: number;
    pdfDataUrl: string | null;
    controlMode: "manual" | "api";
  };
  getMode: () => "manual" | "api";
}

declare global {
  interface Window {
    pdfViewerControls?: PDFViewerControls;
  }

  var pdfClients: ReadableStreamDefaultController[];
  var pdfCommandQueue: any[];
}

export {};
