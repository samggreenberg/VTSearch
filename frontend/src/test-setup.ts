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

// --- fakeAsync / ProxyZone bootstrap ---------------------------------------
// Angular's fakeAsync()/tick() require each test to run inside a Zone forked
// with a ProxyZoneSpec. zone.js/testing only auto-wires that for Jasmine,
// Mocha and Jest runners — its jest patch bails with `typeof jest ===
// 'undefined'`, and Vitest is none of those — so without help every
// fakeAsync() spec throws "Expected to be running in 'ProxyZone'".
//
// zone.js/testing is already loaded (it ships in the builder's test polyfills,
// which run before this setup file), so Zone.ProxyZoneSpec is available. We
// replicate zone.js's jest patch against Vitest's global describe/it/hooks
// (which share the same names): describe bodies run in a sync-only zone and
// it/beforeEach/afterEach bodies run in the proxy zone, exactly as the jest
// patch does. The specs use no it.only/skip/each modifiers, so wrapping the
// bare globals is sufficient.
{
  type ZoneLike = {
    current: { fork(spec: unknown): { run(fn: unknown, ctx: unknown, args: unknown): unknown } };
    ProxyZoneSpec?: new () => unknown;
    SyncTestZoneSpec?: new (name: string) => unknown;
  };
  const g = globalThis as unknown as Record<string, unknown> & { Zone?: ZoneLike };
  const Zone = g.Zone;
  const PATCHED = '__vtsearchProxyZonePatched__';
  if (Zone?.ProxyZoneSpec && Zone.SyncTestZoneSpec && !g[PATCHED]) {
    g[PATCHED] = true;
    const rootZone = Zone.current;
    const syncZone = rootZone.fork(new Zone.SyncTestZoneSpec('vitest.describe'));
    const proxyZone = rootZone.fork(new Zone.ProxyZoneSpec());

    const runIn =
      (zone: { run(fn: unknown, ctx: unknown, args: unknown): unknown }, body: unknown) =>
      function (this: unknown, ...args: unknown[]): unknown {
        return zone.run(body, this, args);
      };

    const wrap = (name: string, bodyArgIndex: number, zone: typeof syncZone) => {
      const original = g[name] as ((...a: unknown[]) => unknown) | undefined;
      if (typeof original !== 'function') {
        return;
      }
      const wrapped = function (this: unknown, ...args: unknown[]): unknown {
        if (typeof args[bodyArgIndex] === 'function') {
          args[bodyArgIndex] = runIn(zone, args[bodyArgIndex]);
        }
        return original.apply(this, args);
      };
      // Preserve modifiers/helpers (.only, .skip, .each, …) as-is.
      Object.assign(wrapped, original);
      g[name] = wrapped;
    };

    for (const name of ['describe', 'fdescribe', 'xdescribe']) {
      wrap(name, 1, syncZone);
    }
    for (const name of ['it', 'fit', 'xit', 'test', 'xtest']) {
      wrap(name, 1, proxyZone);
    }
    for (const name of ['beforeEach', 'afterEach', 'beforeAll', 'afterAll']) {
      wrap(name, 0, proxyZone);
    }
  }
}
