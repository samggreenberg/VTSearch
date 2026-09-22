import { TestBed } from '@angular/core/testing';
import {
  MediaPrefetchService,
  PREFETCH_CAPACITY,
  imageUrlsWhileImages,
} from './media-prefetch.service';
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
    const urls = Array.from({ length: PREFETCH_CAPACITY + 1 }, (_, i) => `/api/medias/${i + 10}/image`);
    for (const url of urls) await warmed(url);
    expect(created.length).toBe(PREFETCH_CAPACITY + 1);
    expect(revoked).toContain(created[0]);
    expect(svc.resolve(urls[0])).toBe(urls[0]);
    expect(svc.resolve(urls[PREFETCH_CAPACITY])).toBe(created[PREFETCH_CAPACITY]);
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
  // --- prefetch(): the review flows' entry point -----------------------------

  describe('prefetch()', () => {
    it('fetches targets one at a time, in order', async () => {
      svc.prefetch([A, B]);
      // B waits: two parallel fetches would split the link and land A later.
      expect([...pending.keys()]).toEqual([A]);
      pending.get(A)?.resolve(true);
      await flush();
      await flush();
      expect([...pending.keys()]).toEqual([A, B]);
      pending.get(B)?.resolve(true);
      await flush();
      expect(svc.resolve(A)).toBe(created[0]);
      expect(svc.resolve(B)).toBe(created[1]);
    });

    it('aborts a stale in-flight target when the prediction changes', async () => {
      svc.prefetch([A, B]);
      // A re-sort lands with the same item on screen: the queue is now C, B.
      svc.prefetch([C, B]);
      expect(pending.get(A)?.aborted).toBe(true);
      expect(pending.has(C)).toBe(true);
      pending.get(C)?.resolve(true);
      await flush();
      await flush();
      expect(pending.has(B)).toBe(true);
    });

    it('revokes held bytes that are no longer predicted', async () => {
      svc.prefetch([A]);
      pending.get(A)?.resolve(true);
      await flush();
      svc.prefetch([B]);
      expect(revoked).toContain(created[0]);
      expect(svc.resolve(A)).toBe(A);
    });

    it('keeps the on-screen item held until the viewer takes it', async () => {
      svc.prefetch([A, B]);
      pending.get(A)?.resolve(true);
      await flush();
      // The vote advanced onto A. The selection effect retargets past it
      // *before* the viewer has resolved it — the two run in the same flush.
      svc.prefetch([B, C], [A]);
      expect(revoked).not.toContain(created[0]);
      expect(svc.resolve(A)).toBe(created[0]);
    });

    it('keeps the on-screen item in flight across a retarget', () => {
      svc.prefetch([A, B]);
      svc.prefetch([B, C], [A]);
      expect(pending.get(A)?.aborted).toBe(false);
    });

    it('does not refetch what it already holds', async () => {
      svc.prefetch([A]);
      pending.get(A)?.resolve(true);
      await flush();
      pending.clear();
      svc.prefetch([A]);
      expect(pending.size).toBe(0);
    });

    it('an empty prediction drops everything', async () => {
      svc.prefetch([A, B]);
      pending.get(A)?.resolve(true);
      await flush();
      await flush();
      expect(pending.has(B)).toBe(true);
      svc.prefetch([]);
      expect(pending.get(B)?.aborted).toBe(true);
      expect(revoked).toContain(created[0]);
    });
  });

  // --- claim(): the vote beat the prefetch -----------------------------------

  describe('claim()', () => {
    it('is null when nothing is in flight for the url', () => {
      expect(svc.claim(A)).toBeNull();
    });

    it('hands the in-flight bytes to the viewer instead of fetching twice', async () => {
      svc.prefetch([A, B]);
      const claimed = svc.claim(A);
      expect(claimed).not.toBeNull();
      pending.get(A)?.resolve(true);
      await flush();
      expect(await claimed).toBe(created[0]);
      // Handed out, so not also parked for a second resolve.
      expect(svc.resolve(A)).toBe(A);
      expect(pending.size).toBeLessThanOrEqual(2);
    });

    it('survives a retarget that no longer lists it', () => {
      svc.prefetch([A, B]);
      void svc.claim(A);
      svc.prefetch([B, C]);
      expect(pending.get(A)?.aborted).toBe(false);
    });

    it('resolves null when the fetch fails, so the viewer falls back', async () => {
      svc.prefetch([A]);
      const claimed = svc.claim(A);
      pending.get(A)?.resolve(false);
      await flush();
      expect(await claimed).toBeNull();
    });

    it('resolves null on clear()', async () => {
      svc.prefetch([A]);
      const claimed = svc.claim(A);
      svc.clear();
      expect(await claimed).toBeNull();
    });

    it('revokes the previously displayed object URL when claimed bytes arrive', async () => {
      await warmed(B);
      const shown = svc.resolve(B);
      svc.prefetch([A]);
      const claimed = svc.claim(A);
      pending.get(A)?.resolve(true);
      expect(await claimed).toBe(created[1]);
      expect(revoked).toContain(shown);
    });

    it('after release(), landed bytes are an ordinary warm again', async () => {
      svc.prefetch([A]);
      const claimed = svc.claim(A);
      svc.release(A);
      pending.get(A)?.resolve(true);
      expect(await claimed).toBeNull();
      expect(svc.resolve(A)).toBe(created[0]);
    });
  });
});

describe('imageUrlsWhileImages', () => {
  const types: Record<number, string> = { 1: 'image', 2: 'image', 3: 'audio', 4: 'image' };
  const url = (path: string) => `${path}?dataset_id=d`;

  it('maps image ids to their exact image URLs, in order', () => {
    expect(imageUrlsWhileImages([2, 1], (id) => types[id], url)).toEqual([
      '/api/medias/2/image?dataset_id=d',
      '/api/medias/1/image?dataset_id=d',
    ]);
  });

  it('cuts at the first non-image or unknown item instead of skipping it', () => {
    expect(imageUrlsWhileImages([1, 3, 4], (id) => types[id], url)).toEqual([
      '/api/medias/1/image?dataset_id=d',
    ]);
    expect(imageUrlsWhileImages([99, 1], (id) => types[id], url)).toEqual([]);
  });
});
