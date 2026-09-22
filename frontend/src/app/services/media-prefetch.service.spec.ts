import { TestBed } from '@angular/core/testing';
import { MediaPrefetchService, PREFETCH_CAPACITY } from './media-prefetch.service';
import { configureZoneless } from '../testing/zoneless-testbed';

/**
 * Specs for the review-image prefetch store (#3896).
 *
 * The behaviours pinned here are the ones that make prefetching *safe* rather
 * than merely fast: an entry is only ever handed back for the URL it was
 * fetched from, it is handed back once, and every failure path degrades to the
 * network URL instead of to a blank panel or a wrong image.
 *
 * jsdom implements neither `fetch`'s blob path nor `URL.createObjectURL`, so
 * both are stubbed; `svc.canStore` reads them per call for exactly this reason.
 */
describe('MediaPrefetchService', () => {
  const A = '/api/medias/1/image?dataset_id=d';
  const B = '/api/medias/2/image?dataset_id=d';
  const C = '/api/medias/3/image?dataset_id=d';

  let svc: MediaPrefetchService;
  let created: string[];
  let revoked: string[];
  let pending: Map<string, { resolve: (ok: boolean) => void; aborted: boolean }>;

  /** Settle the microtasks the fetch chain queues (`.then` x2). */
  const flush = async (): Promise<void> => {
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
  };

  beforeEach(() => {
    configureZoneless();
    created = [];
    revoked = [];
    pending = new Map();

    let seq = 0;
    vi.stubGlobal('URL', {
      createObjectURL: (_blob: unknown) => {
        const url = `blob:mock/${++seq}`;
        created.push(url);
        return url;
      },
      revokeObjectURL: (url: string) => void revoked.push(url),
    });
    vi.stubGlobal('fetch', (url: string, init?: { signal?: AbortSignal }) => {
      return new Promise((resolvePromise, rejectPromise) => {
        const entry = {
          resolve: (ok: boolean) =>
            resolvePromise({ ok, blob: () => Promise.resolve({ size: 1 }) }),
          aborted: false,
        };
        pending.set(url, entry);
        init?.signal?.addEventListener('abort', () => {
          entry.aborted = true;
          rejectPromise(new Error('aborted'));
        });
      });
    });

    svc = TestBed.inject(MediaPrefetchService);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  /** Warm `url` and let its fetch succeed. */
  const warmed = async (url: string): Promise<void> => {
    svc.warm(url);
    pending.get(url)?.resolve(true);
    await flush();
  };

  it('resolves to the network URL when nothing is warm', () => {
    expect(svc.resolve(A)).toBe(A);
  });

  it('hands back the warmed bytes for that exact URL', async () => {
    await warmed(A);
    expect(svc.resolve(A)).toBe(created[0]);
  });

  it('never hands a warmed entry to a different URL', async () => {
    await warmed(A);
    expect(svc.resolve(B)).toBe(B);
  });

  it('hands an entry out once, then falls back', async () => {
    await warmed(A);
    expect(svc.resolve(A)).toBe(created[0]);
    expect(svc.resolve(A)).toBe(A);
  });

  it('keeps the displayed object URL alive and revokes it on the next hand-out', async () => {
    await warmed(A);
    await warmed(B);
    const first = svc.resolve(A);
    expect(revoked).not.toContain(first);
    const second = svc.resolve(B);
    expect(revoked).toContain(first);
    expect(revoked).not.toContain(second);
  });

  it('fetches a URL once while it is in flight or held', async () => {
    svc.warm(A);
    svc.warm(A);
    expect(pending.size).toBe(1);
    pending.get(A)?.resolve(true);
    await flush();
    svc.warm(A);
    expect(created.length).toBe(1);
  });

  it('drops the oldest entry past capacity and revokes it', async () => {
    await warmed(A);
    await warmed(B);
    await warmed(C);
    expect(created.length).toBe(PREFETCH_CAPACITY + 1);
    expect(revoked).toContain(created[0]);
    expect(svc.resolve(A)).toBe(A);
    expect(svc.resolve(C)).toBe(created[2]);
  });

  it('falls back when the fetch fails, and can be retried', async () => {
    svc.warm(A);
    pending.get(A)?.resolve(false);
    await flush();
    expect(svc.resolve(A)).toBe(A);
    expect(created.length).toBe(0);
    pending.clear();
    svc.warm(A);
    expect(pending.has(A)).toBe(true);
  });

  it('clear() aborts what is in flight and revokes what is held', async () => {
    await warmed(A);
    svc.warm(B);
    svc.clear();
    expect(pending.get(B)?.aborted).toBe(true);
    expect(revoked).toContain(created[0]);
    expect(svc.resolve(A)).toBe(A);
  });

  it('ignores bytes that land after a clear', async () => {
    svc.warm(A);
    svc.clear();
    pending.get(A)?.resolve(true);
    await flush();
    expect(svc.resolve(A)).toBe(A);
  });

  it('is inert where object URLs are unavailable', async () => {
    vi.stubGlobal('URL', {});
    svc.warm(A);
    await flush();
    expect(pending.size).toBe(0);
    expect(svc.resolve(A)).toBe(A);
  });
});
