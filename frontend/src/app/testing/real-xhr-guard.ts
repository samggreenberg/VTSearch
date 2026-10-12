import { expect } from 'vitest';

/**
 * Fails any spec that lets a real `XMLHttpRequest` out (#4808).
 *
 * Angular provides `HttpClient` in root, so a spec whose providers lack
 * `provideHttpTesting()` gets no "No provider for HttpClient" error. Its
 * requests go out through the real XHR backend to jsdom's default origin,
 * `http://localhost:3000`, and nothing fails: the run just prints
 * `connect ECONNREFUSED 127.0.0.1:3000`, credited to no file in particular.
 *
 * The guard stops every real `open()` before it reaches the network: the call
 * throws, and the request is recorded so `src/test-setup.ts` can fail the test
 * in an `afterEach`. A thrown `open()` alone is not enough, because the code
 * under test usually swallows its own request errors.
 */

const STATE = Symbol.for('vtsearch.realXhrGuard');

interface GuardState {
  requests: string[];
}

type GuardedGlobal = typeof globalThis & { [STATE]?: GuardState };

/**
 * Patch `XMLHttpRequest.prototype.open`, once per jsdom window.
 *
 * `vitest.config.ts` runs with `isolate: false`, so every spec file in a worker
 * shares one window while the setup file is re-evaluated for each of them. The
 * state lives on that window so a second call does not wrap `open()` again.
 */
export function installRealXhrGuard(): void {
  const g = globalThis as GuardedGlobal;
  if (g[STATE]) return;
  const state: GuardState = { requests: [] };
  g[STATE] = state;
  XMLHttpRequest.prototype.open = function (method: string, url: string | URL): void {
    const request = `${method} ${String(url)}`;
    state.requests.push(`${request} (opened in "${expect.getState().currentTestName ?? 'no test'}")`);
    throw new Error(
      `A spec sent a real XMLHttpRequest (${request}). Add ...provideHttpTesting() ` +
        'to its TestBed providers and handle the request with HttpTestingController.',
    );
  } as typeof XMLHttpRequest.prototype.open;
}

/** Return the real requests recorded since the last call, and forget them. */
export function takeRealXhrRequests(): string[] {
  const state = (globalThis as GuardedGlobal)[STATE];
  return state ? state.requests.splice(0) : [];
}
