/**
 * Notebooks data API client
 * This module handles interaction with the notebook storage API
 */

export interface Notebook {
  id: string;
  name: string;
  description: string;
  created_at: string;
}

// Default notebooks in case API fails
const DEFAULT_NOTEBOOKS: Notebook[] = [
  {
    id: "nb_default_1",
    name: "Research Papers",
    description: "Collection of scientific papers and research documents",
    created_at: "2023-01-15T12:00:00Z"
  },
  {
    id: "nb_default_2",
    name: "Project Documentation",
    description: "Technical specifications and project documents",
    created_at: "2023-01-20T14:30:00Z"
  }
];

/**
 * Load notebooks from the API
 * 
 * Using cache:no-store to prevent React's automatic caching from causing issues
 */
export async function loadNotebooksFromAPI(): Promise<Notebook[]> {
  try {
    const response = await fetch('/api/notebooks', {
      method: 'GET',
      cache: 'no-store',
      headers: {
        'cache-control': 'no-cache'
      }
    });
    
    if (!response.ok) {
      console.error('Failed to load notebooks from API:', response.statusText);
      return DEFAULT_NOTEBOOKS;
    }
    
    const data = await response.json();
    return data.notebooks || DEFAULT_NOTEBOOKS;
  } catch (error) {
    console.error('Error loading notebooks from API:', error);
    return DEFAULT_NOTEBOOKS;
  }
}

/**
 * Save notebooks to the API
 */
export async function saveNotebooksToAPI(notebooks: Notebook[]): Promise<boolean> {
  try {
    const response = await fetch('/api/notebooks', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ notebooks }),
    });
    
    if (!response.ok) {
      console.error('Failed to save notebooks to API:', response.statusText);
      return false;
    }
    
    return true;
  } catch (error) {
    console.error('Error saving notebooks to API:', error);
    return false;
  }
}

/**
 * For backward compatibility - load notebooks from localStorage
 */
export function loadNotebooksFromLocalStorage(): Notebook[] {
  try {
    const storedNotebooks = localStorage.getItem('notebooks');
    if (storedNotebooks) {
      return JSON.parse(storedNotebooks);
    }
    return [];
  } catch (error) {
    console.error('Error loading notebooks from localStorage:', error);
    return [];
  }
}

/**
 * For backward compatibility - save notebooks to localStorage
 */
export function saveNotebooksToLocalStorage(notebooks: Notebook[]): void {
  try {
    localStorage.setItem('notebooks', JSON.stringify(notebooks));
  } catch (error) {
    console.error('Error saving notebooks to localStorage:', error);
  }
}