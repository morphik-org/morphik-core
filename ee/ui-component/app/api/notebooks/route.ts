import { NextRequest, NextResponse } from "next/server";
import { readFileSync, writeFileSync } from "fs";
import { resolve } from "path";

// Path to the notebooks storage file
const NOTEBOOKS_FILE_PATH = resolve(process.cwd(), "notebook-storage/notebooks.json");

// Default notebooks in case file is missing or corrupted
const DEFAULT_NOTEBOOKS = [
  {
    id: "nb_default_1",
    name: "Research Papers",
    description: "Collection of scientific papers and research documents",
    created_at: "2023-01-15T12:00:00Z",
  },
  {
    id: "nb_default_2",
    name: "Project Documentation",
    description: "Technical specifications and project documents",
    created_at: "2023-01-20T14:30:00Z",
  },
];

// Helper functions to read and write notebooks
function readNotebooks() {
  try {
    const fileData = readFileSync(NOTEBOOKS_FILE_PATH, "utf8");
    const data = JSON.parse(fileData);
    return data.notebooks || DEFAULT_NOTEBOOKS;
  } catch (error) {
    console.error("Error reading notebooks:", error);
    return DEFAULT_NOTEBOOKS;
  }
}

interface Notebook {
  id: string;
  name: string;
  description: string;
  created_at: string;
}

function writeNotebooks(notebooks: Notebook[]) {
  try {
    const data = {
      notebooks,
      lastUpdated: new Date().toISOString(),
    };
    writeFileSync(NOTEBOOKS_FILE_PATH, JSON.stringify(data, null, 2), "utf8");
    return true;
  } catch (error) {
    console.error("Error writing notebooks:", error);
    return false;
  }
}

// GET endpoint to retrieve notebooks
export async function GET() {
  const notebooks = readNotebooks();
  return NextResponse.json({ notebooks });
}

// POST endpoint to save notebooks
export async function POST(request: NextRequest) {
  try {
    const data = await request.json();

    if (!data.notebooks || !Array.isArray(data.notebooks)) {
      return NextResponse.json({ error: "Invalid notebook data" }, { status: 400 });
    }

    const success = writeNotebooks(data.notebooks);

    if (success) {
      return NextResponse.json({ success: true });
    } else {
      return NextResponse.json({ error: "Failed to save notebooks" }, { status: 500 });
    }
  } catch (error) {
    console.error("Error in POST /api/notebooks:", error);
    return NextResponse.json({ error: "Server error" }, { status: 500 });
  }
}
