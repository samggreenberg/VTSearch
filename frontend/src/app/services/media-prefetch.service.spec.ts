import { TestBed } from '@angular/core/testing';
import { MediaPrefetchService, imageUrlsWhileImages } from './media-prefetch.service';
import { configureZoneless } from '../testing/zoneless-testbed';

/**
 * Specs for the review-image prefetch (#3896), which warms the browser's HTTP
 * cache now that `/api/medias/<id>/image` is cacheable (#5648).
 *
 * jsdom has no HTTP cache, so what is pinned here is the scheduling: which URL
 * is fetched, in what order, how many at once, what a retarget aborts, and that
 * a failing URL is not hammered. That the browser then answers the viewer's
 * `<img>` from the cache was measured in Chromium (see the service's doc).
 *
 * `fetch` is stubbed so each request can be settled or aborted by hand.
 */
describe('MediaPrefetchService', () => {
  const A = '/api/medias/1/image?dataset_id=d';
  const B = '/api/medias/2/image?dataset_id=d';
  const C = '/api/medias/3/image?dataset_id=d';

  let svc: MediaPrefetchService;
  let pending: Map<string, { resolve: (ok: boolean) => void; aborted: boolean }>;
  /** Every url fetched, in order, including repeats. */
  let requested: string[];
  /** Urls whose body was read to the end, i.e. would be in the cache. */
  let bodiesRead: string[];
  let credentials: (string | undefined)[];

  /** Settle the microtasks the fetch chain queues. */
  const flush = async (): Promise<void> => {
    for (let i = 0; i < 6; i++) await Promise.resolve();
  };

  beforeEach(() => {
    configureZoneless();
    pending = new Map();
    requested = [];
    bodiesRead = [];
    credentials = [];

    vi.stubGlobal('fetch', (url: string, init?: RequestInit) => {
      requested.push(url);
      credentials.push(init?.credentials);
      return new Promise((resolvePromise, rejectPromise) => {
        const entry = {
          resolve: (ok: boolean) =>
            resolvePromise({
              ok,
              blob: () => {
                bodiesRead.push(url);
                return Promise.resolve({ size: 1 });
              },
            }),
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

  it('fetches the exact URL with same-origin credentials, and reads the body so it is cached', async () => {
    svc.warm(A);
    expect(requested).toEqual([A]);
    // The viewer's `<img>` sends cookies too; matching it keeps one cache entry.
    expect(credentials).toEqual(['same-origin']);
    pending.get(A)?.resolve(true);
    await flush();
    expect(bodiesRead).toEqual([A]);
  });

  it('does not read the body of a failed response', async () => {
    svc.warm(A);
    pending.get(A)?.resolve(false);
    await flush();
    expect(bodiesRead).toEqual([]);
  });

  it('fetches a URL once while it is in flight or warm', async () => {
    svc.warm(A);
    svc.warm(A);
    expect(requested).toEqual([A]);
    pending.get(A)?.resolve(true);
    await flush();
    svc.warm(A);
    expect(requested).toEqual([A]);
  });

  it('clear() aborts what is in flight', () => {
    svc.warm(A);
    svc.clear();
    expect(pending.get(A)?.aborted).toBe(true);
  });

  it('forgets what was warm on clear(), so the next prediction fetches it again', async () => {
    svc.prefetch([A]);
    pending.get(A)?.resolve(true);
    await flush();
    svc.clear();
    svc.prefetch([A]);
    expect(requested).toEqual([A, A]);
  });

  it('ignores a fetch that lands after a clear', async () => {
    svc.prefetch([A, B]);
    svc.clear();
    pending.get(A)?.resolve(true);
    await flush();
    // Nothing moved on to B: the cleared prediction is gone.
    expect(requested).toEqual([A]);
  });

  it('is inert where fetch is unavailable', () => {
    vi.stubGlobal('fetch', undefined);
    expect(() => svc.prefetch([A, B])).not.toThrow();
  });

  it('treats a fetch that throws synchronously as a failure', async () => {
    vi.stubGlobal('fetch', () => {
      throw new TypeError('bad url');
    });
    svc.prefetch([A, B]);
    await flush();
    let retried = 0;
    vi.stubGlobal('fetch', () => {
      retried++;
      return new Promise(() => undefined);
    });
    svc.prefetch([A, B]);
    await flush();
    // Both failed once and stay failed while wanted; nothing is retried.
    expect(retried).toBe(0);
  });

  // --- prefetch(): the review flows' entry point -----------------------------

  describe('prefetch()', () => {
    it('fetches targets one at a time, in order', async () => {
      svc.prefetch([A, B]);
      // B waits: two parallel fetches would split the link and land A later.
      expect(requested).toEqual([A]);
      pending.get(A)?.resolve(true);
      await flush();
      expect(requested).toEqual([A, B]);
      pending.get(B)?.resolve(true);
      await flush();
      expect(bodiesRead).toEqual([A, B]);
    });

    it('aborts a stale in-flight target when the prediction changes', async () => {
      svc.prefetch([A, B]);
      // A re-sort lands with the same item on screen: the queue is now C, B.
      svc.prefetch([C, B]);
      expect(pending.get(A)?.aborted).toBe(true);
      expect(requested).toEqual([A, C]);
      pending.get(C)?.resolve(true);
      await flush();
      expect(requested).toEqual([A, C, B]);
    });

    it('keeps the on-screen item in flight across a retarget', () => {
      // The vote advanced onto A while its warm was still downloading; the
      // viewer's `<img>` is riding on that download, so it must not be cut.
      svc.prefetch([A, B]);
      svc.prefetch([B, C], [A]);
      expect(pending.get(A)?.aborted).toBe(false);
      // And the next target waits for it: still one fetch at a time.
      expect(requested).toEqual([A]);
    });

    it('does not refetch what it already warmed', async () => {
      svc.prefetch([A]);
      pending.get(A)?.resolve(true);
      await flush();
      svc.prefetch([A, B]);
      expect(requested).toEqual([A, B]);
    });

    it('does not refetch the on-screen item once it has moved from target to keep', async () => {
      svc.prefetch([A, B]);
      pending.get(A)?.resolve(true);
      await flush();
      svc.prefetch([B, C], [A]);
      pending.get(B)?.resolve(true);
      await flush();
      expect(requested).toEqual([A, B, C]);
    });

    it('warms again an image a retarget dropped and then wants back', async () => {
      // The HTTP cache still holds it, so the repeat fetch is a cache read;
      // forgetting it keeps this service's bookkeeping bounded by the prediction.
      svc.prefetch([A]);
      pending.get(A)?.resolve(true);
      await flush();
      svc.prefetch([B]);
      pending.get(B)?.resolve(true);
      await flush();
      svc.prefetch([A]);
      expect(requested).toEqual([A, B, A]);
    });

    it('does not fetch a failed target again while the prediction holds it (#5099)', async () => {
      svc.prefetch([A, B]);
      pending.get(A)?.resolve(false);
      await flush();
      // On to B, not back to A: a url that fails would otherwise be fetched
      // back to back for as long as it stays a target.
      expect(requested).toEqual([A, B]);
      // A retarget that still wants A leaves it alone too.
      svc.prefetch([A, B, C]);
      pending.get(B)?.resolve(true);
      await flush();
      expect(requested).toEqual([A, B, C]);
    });

    it('tries a failed target again once a retarget has dropped it', async () => {
      svc.prefetch([A]);
      pending.get(A)?.resolve(false);
      await flush();
      svc.prefetch([B]);
      svc.prefetch([A]);
      expect(requested).toEqual([A, B, A]);
    });

    it('does not count a target a retarget aborted as failed', async () => {
      svc.prefetch([A]);
      svc.prefetch([B]);
      expect(pending.get(A)?.aborted).toBe(true);
      await flush();
      pending.get(B)?.resolve(true);
      await flush();
      svc.prefetch([A]);
      expect(requested).toEqual([A, B, A]);
    });

    it('an empty prediction aborts what is in flight', async () => {
      svc.prefetch([A, B]);
      pending.get(A)?.resolve(true);
      await flush();
      expect(requested).toEqual([A, B]);
      svc.prefetch([]);
      expect(pending.get(B)?.aborted).toBe(true);
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
