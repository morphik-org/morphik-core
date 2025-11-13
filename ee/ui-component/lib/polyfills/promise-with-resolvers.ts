"use strict";

/**
 * Provides a minimal `Promise.withResolvers` implementation for runtimes
 * (like Node 18) that have not shipped the TC39 proposal yet.
 */
if (typeof Promise.withResolvers !== "function") {
  Promise.withResolvers = function withResolvers<T = unknown>() {
    let resolve!: (value: T | PromiseLike<T>) => void;
    let reject!: (reason?: unknown) => void;

    const promise = new Promise<T>((res, rej) => {
      resolve = res;
      reject = rej;
    });

    return { promise, resolve, reject };
  };
}

export {};
