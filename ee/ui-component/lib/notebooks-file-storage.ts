/**
 * File-based notebook data store
 * This provides permanent storage for notebooks in a JSON file
 */

import { readFileSync, writeFileSync } from "fs";
import { resolve } from "path";

export interface Notebook {
  id: string;
  name: string;
  description: string;
  created_at: string;
}

interface NotebooksData {
  notebooks: Notebook[];
  lastUpdated: string;
}

// Path to the notebooks storage file
const NOTEBOOKS_FILE_PATH = resolve(process.cwd(), "ui-component/notebook-storage/notebooks.json");

// Default notebooks in case file is missing or corrupted
const DEFAULT_NOTEBOOKS: Notebook[] = [
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

/**
 * Load notebooks from file
 */
export const loadNotebooksFromFile = (): Notebook[] => {
  try {
    // Read the notebooks JSON file
    const fileData = readFileSync(NOTEBOOKS_FILE_PATH, "utf8");
    const data: NotebooksData = JSON.parse(fileData);

    return data.notebooks || DEFAULT_NOTEBOOKS;
  } catch (error) {
    console.error("Error loading notebooks from file:", error);
    return DEFAULT_NOTEBOOKS;
  }
};

/**
 * Save notebooks to file
 */
export const saveNotebooksToFile = (notebooks: Notebook[]): void => {
  try {
    const data: NotebooksData = {
      notebooks,
      lastUpdated: new Date().toISOString(),
    };

    writeFileSync(NOTEBOOKS_FILE_PATH, JSON.stringify(data, null, 2), "utf8");
  } catch (error) {
    console.error("Error saving notebooks to file:", error);
  }
};
