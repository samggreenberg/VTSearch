/**
 * Vitest setup: polyfills for browser APIs that jsdom does not implement.
 *
 * The Angular component/service specs run headless in jsdom under the
 * `@angular/build:unit-test` Vitest runner. jsdom omits a handful of APIs the
 * app touches during component construction / lifecycle (SSE, media playback,
 * canvas drawing) and throws "Not implemented" when they are called, which
 * aborts the test before TestBed can clean up. These stubs are inert no-ops:
 * they keep jsdom quiet, they do not assert behavior.
 */

import { getTestBed } from '@angular/core/testing';
import { afterEach as vitestAfterEach, beforeEach as vitestBeforeEach } from 'vitest';
import { installRealXhrGuard, takeRealXhrRequests } from './app/testing/real-xhr-guard';

// --- TestBed cascade guard --------------------------------------------------
// The Angular TestBed is a module-level singleton reset by an afterEach hook.
// If a spec leaves the TestBed dirty in a way that makes teardown throw (e.g.
// an HttpTestingController with an unflushed request, so its verify() throws,
// or a component whose ngOnDestroy throws), that afterEach reset never
// completes and every following spec in the file fails with "test module
// already instantiated". Resetting defensively at the START of each test —
// before the spec's own beforeEach calls configureTestingModule — keeps one
// bad spec's fallout from cascading into the rest of the file. This runs after
// the builder's own cleanup beforeEach (setup files run first, in order) and
// before any spec-level beforeEach, so configureTestingModule always sees a
// fresh TestBed.
vitestBeforeEach(() => {
  try {
    getTestBed().resetTestingModule();
  } catch {
    // A throwing teardown from the previous spec is itself a reported failure;
    // swallow it here so it cannot mask the upcoming test.
  }
});

// --- Real XHR guard ---------------------------------------------------------
// A spec without provideHttpTesting() falls through to Angular's root
// HttpClient and its real XHR backend, which jsdom sends to localhost:3000
// (#4808). The guard makes that open() throw instead of reaching the network,
// and this hook fails the test that did it, naming the request.
installRealXhrGuard();
vitestAfterEach(() => {
  const requests = takeRealXhrRequests();
  if (requests.length) {
    throw new Error(
      `Real XMLHttpRequest(s) sent; add ...provideHttpTesting() to the spec's providers:\n  ${requests.join('\n  ')}`,
    );
  }
});

// --- EventSource (Server-Sent Events) --------------------------------------
// ProgressEventsService opens an EventSource on '/api/events'. jsdom has none.
class EventSourceMock {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSED = 2;
  readonly CONNECTING = 0;
  readonly OPEN = 1;
  readonly CLOSED = 2;
  url: string;
  readyState = 0;
  withCredentials = false;
  onopen: ((ev: Event) => void) | null = null;
  onmessage: ((ev: MessageEvent) => void) | null = null;
  onerror: ((ev: Event) => void) | null = null;
  constructor(url: string | URL) {
    this.url = String(url);
  }
  addEventListener(): void {}
  removeEventListener(): void {}
  dispatchEvent(): boolean {
    return false;
  }
  close(): void {
    this.readyState = 2;
  }
}
(globalThis as unknown as { EventSource: unknown }).EventSource = EventSourceMock;

// --- ResizeObserver ----------------------------------------------------------
// The virtualized grids (media-list, vote-grid) observe their viewport to
// derive column count and row stride. jsdom has no ResizeObserver — and no
// layout engine, so a real one would never fire anyway. An inert stub lets
// specs mount the virtualized branch.
class ResizeObserverMock {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}
(globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = ResizeObserverMock;

// --- Element.scrollIntoView -------------------------------------------------
// Several components scroll a focused/active row into view (context-pulldown,
// media-list, folder-browser, import-config). jsdom has no layout engine and
// omits scrollIntoView, throwing "not a function" — often from a deferred
// setTimeout callback that surfaces as an unhandled error rather than a test
// failure. An inert no-op keeps those paths quiet.
Object.defineProperty(Element.prototype, 'scrollIntoView', {
  configurable: true,
  writable: true,
  value: () => {},
});

// --- CSS.escape -------------------------------------------------------------
// import-config builds a `#${CSS.escape(id)}` selector to scroll the active
// combobox option into view. jsdom omits the `CSS` namespace object entirely,
// so the reference throws "Cannot read properties of undefined" from the
// deferred queueMicrotask callback (surfacing as an unhandled error, not a
// test failure). Provide a minimal, spec-correct escape() so the selector is
// built and the (stubbed) scrollIntoView above runs.
if (typeof (globalThis as { CSS?: unknown }).CSS === 'undefined') {
  (globalThis as unknown as { CSS: { escape(value: string): string } }).CSS = {
    escape: (value: string) => String(value).replace(/[^a-zA-Z0-9_-]/g, (ch) => `\\${ch}`),
  };
}

// --- HTMLDialogElement showModal/show/close ---------------------------------
// vt-modal opens a <dialog> with showModal() (#5297). jsdom has the element but
// none of its methods. These stand-ins only toggle the `open` attribute, and
// close() fires `close` as the platform does; jsdom has no top layer, inertness
// or focus steps, so what showModal() does for focus is checked in chromium,
// not here.
if (typeof HTMLDialogElement.prototype.showModal !== 'function') {
  Object.defineProperties(HTMLDialogElement.prototype, {
    showModal: {
      configurable: true,
      writable: true,
      value(this: HTMLDialogElement) {
        this.setAttribute('open', '');
      },
    },
    show: {
      configurable: true,
      writable: true,
      value(this: HTMLDialogElement) {
        this.setAttribute('open', '');
      },
    },
    close: {
      configurable: true,
      writable: true,
      value(this: HTMLDialogElement) {
        if (!this.hasAttribute('open')) return;
        this.removeAttribute('open');
        this.dispatchEvent(new Event('close'));
      },
    },
  });
}

// --- HTMLMediaElement play/pause/load --------------------------------------
// The audio/video players call these; jsdom throws "Not implemented".
Object.defineProperties(HTMLMediaElement.prototype, {
  play: { configurable: true, writable: true, value: () => Promise.resolve() },
  pause: { configurable: true, writable: true, value: () => {} },
  load: { configurable: true, writable: true, value: () => {} },
});

// --- HTMLCanvasElement.getContext ------------------------------------------
// The minimap / browse-canvas / charts code draws to a 2D context. jsdom
// (without the optional `canvas` package) throws on getContext. Return a stub
// whose drawing methods are no-ops and whose measureText reports zero width.
const canvasContextStub = new Proxy(
  {},
  {
    get: (_target, prop) => {
      if (prop === 'measureText') {
        return () => ({ width: 0 });
      }
      return () => undefined;
    },
    set: () => true,
  },
);
HTMLCanvasElement.prototype.getContext = (() =>
  canvasContextStub) as unknown as typeof HTMLCanvasElement.prototype.getContext;

// --- no zone.js at all -----------------------------------------------------
// The specs no longer use Angular's fakeAsync()/tick(); they drive time with
// native async + Vitest fake timers (vi.useFakeTimers / advanceTimersByTimeAsync)
// and real macrotask drains instead. Every fixture-creating spec runs under
// the zoneless `TestBed` (`provideZonelessChangeDetection()`), so no spec
// needs `NgZone` from a default zone-based TestBed. zone.js is therefore
// dropped entirely: the build:test polyfills array is empty and the
// package.json dependency is removed. Nothing in this setup file needs it.
