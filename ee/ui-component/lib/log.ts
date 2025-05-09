"use client";

export function setupLogging() {
  if (typeof window === "undefined") return;

  const isDebug = process.env.NEXT_PUBLIC_MORPHIK_DEBUG === "true";

  if (!isDebug) {
    // Preserve original console methods in case they are needed elsewhere
    const noop = () => {};
    console.log = noop;
    console.debug = noop;
  }
}
