import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  MAX_THUMBS,
  MAX_THUMBS_FULL_RES,
  THUMB_BATCH_SIZE,
  THUMB_NATIVE_MAX_DIM,
  THUMB_RETRY_BACKOFF_MS,
  ThumbStore,
  parseThumbBatch,
} from './thumb-store';

/**
 * Unit coverage for the browse canvas's representative-thumbnail store: the
 * insertion-ordered LRU, the capped/full-res resolution tier, the waveform tint
 * cache, and the retry backoff that keeps a transient load failure from blanking
 * a bin for the rest of the session.
 *
 * `Image` is stubbed so a "load" is just a call to the handler the store
 * attached — no network, no decoding. Everything the store promises is
 * observable through that.
 */

/** The stub standing in for a real `HTMLImageElement`. */
interface FakeImage {
  src: string;
  decoding: string;
  onload: (() => void) | null;
  onerror: (() => void) | null;
  complete: boolean;
  naturalWidth: number;
  naturalHeight: number;
  attrs: Record<string, string>;
  setAttribute(name: string, value: string): void;
  listeners: Record<string, (() => void)[]>;
  addEventListener(type: string, listener: () => void): void;
}

/** Every image the store has constructed this test, in construction order. */
let created: FakeImage[] = [];

/** Simulate a successful decode of the image fetched for `src`. */
function finishLoad(img: FakeImage, w = 64, h = 48): void {
  img.complete = true;
  img.naturalWidth = w;
  img.naturalHeight = h;
  img.onload?.();
  for (const listener of img.listeners['load'] ?? []) listener();
}

/** Simulate a failed fetch. */
function failLoad(img: FakeImage): void {
  img.onerror?.();
  for (const listener of img.listeners['error'] ?? []) listener();
}

/** The most recently constructed image (the one a `get`/`warm` just started). */
const last = () => created[created.length - 1];

beforeEach(() => {
  // Restore any spy/global a previous test installed: `vi.spyOn` on an
  // already-spied method reuses the existing mock, so without this the shared
  // `document.createElement` call counts would accumulate across tests.
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  created = [];
  vi.stubGlobal(
    'Image',
    class {
      src = '';
      decoding = '';
      onload: (() => void) | null = null;
      onerror: (() => void) | null = null;
      complete = false;
      naturalWidth = 0;
      naturalHeight = 0;
      attrs: Record<string, string> = {};
      setAttribute(name: string, value: string) {
        this.attrs[name] = value;
      }
      listeners: Record<string, (() => void)[]> = {};
      addEventListener(type: string, listener: () => void) {
        (this.listeners[type] ??= []).push(listener);
      }
      constructor() {
        created.push(this as unknown as FakeImage);
      }
    },
  );
});

/** A store wired to spies, with an injectable clock for the retry backoff. */
function makeStore(overrides: { accent?: () => string } = {}) {
  const onLoaded = vi.fn();
  const clock = { t: 1_000 };
  const store = new ThumbStore({
    mediaUrl: (path) => `https://host${path}?ds=7`,
    onLoaded,
    accent: overrides.accent ?? (() => '#ff0000'),
    now: () => clock.t,
  });
  return { store, onLoaded, clock };
}

describe('ThumbStore.wantsFullRes', () => {
  it('stays on the capped thumbnail while a cell fits within the native cap', () => {
    // Diameter 2*r*dpr must exceed THUMB_NATIVE_MAX_DIM to flip.
    expect(ThumbStore.wantsFullRes(THUMB_NATIVE_MAX_DIM / 2, 1)).toBe(false);
    expect(ThumbStore.wantsFullRes(28, 1)).toBe(false);
  });

  it('flips once a cell is drawn wider than the capped thumbnail', () => {
    expect(ThumbStore.wantsFullRes(THUMB_NATIVE_MAX_DIM / 2 + 1, 1)).toBe(true);
  });

  it('accounts for device pixel ratio, not just CSS size', () => {
    const radius = THUMB_NATIVE_MAX_DIM / 2 - 10;
    expect(ThumbStore.wantsFullRes(radius, 1)).toBe(false);
    expect(ThumbStore.wantsFullRes(radius, 2)).toBe(true);
  });
});

describe('ThumbStore fetching', () => {
  it('starts a fetch on first request and returns null until it decodes', () => {
    const { store } = makeStore();
    expect(store.get(1)).toBeNull();
    expect(created).toHaveLength(1);
    expect(last().src).toBe('https://host/api/medias/1/thumbnail?ds=7');
    // Still pending.
    expect(store.get(1)).toBeNull();
    expect(created).toHaveLength(1);
  });

  it('returns the image once it has decoded', () => {
    const { store } = makeStore();
    store.get(1);
    finishLoad(last());
    expect(store.get(1)).toBe(last());
  });

  it('repaints when a visible thumbnail lands', () => {
    const { store, onLoaded } = makeStore();
    store.get(1);
    expect(onLoaded).not.toHaveBeenCalled();
    finishLoad(last());
    expect(onLoaded).toHaveBeenCalledTimes(1);
  });

  it('treats a decoded-but-empty image as not ready', () => {
    const { store } = makeStore();
    store.get(1);
    last().complete = true;
    last().naturalWidth = 0;
    expect(store.get(1)).toBeNull();
  });

  it('fetches the full-res endpoint in the full-res tier', () => {
    const { store } = makeStore();
    store.setFullRes(true);
    store.get(1);
    expect(last().src).toBe('https://host/api/medias/1/image?ds=7');
  });

  it('peek reads without fetching or bumping recency', () => {
    const { store } = makeStore();
    expect(store.peek(1)).toBeUndefined();
    expect(created).toHaveLength(0);
    store.get(1);
    expect(store.peek(1)).toBe(last());
    expect(created).toHaveLength(1);
  });

  it('marks off-view warm-ups as low priority and silent', () => {
    const { store, onLoaded } = makeStore();
    expect(store.warm(1)).toBe(true);
    expect(last().attrs['fetchpriority']).toBe('low');
    finishLoad(last());
    // A warm-up paints nothing, so it must not schedule a repaint.
    expect(onLoaded).not.toHaveBeenCalled();
  });

  it('asks for a visible thumbnail at high priority', () => {
    const { store } = makeStore();
    store.get(1);
    expect(last().attrs['fetchpriority']).toBe('high');
  });

  it('repaints when a warm-up a visible cell has since asked for lands', () => {
    // #5636: a warm-up that came on screen while still in flight used to land
    // silently, so the bin stayed grey until something else redrew the frame.
    const { store, onLoaded } = makeStore();
    store.warm(1);
    expect(store.get(1)).toBeNull();
    expect(created).toHaveLength(1);
    finishLoad(last());
    expect(onLoaded).toHaveBeenCalledTimes(1);
    expect(store.get(1)).toBe(last());
  });

  it('repaints when a warm-up a visible cell has since asked for fails', () => {
    const { store, onLoaded } = makeStore();
    store.warm(1);
    store.get(1);
    failLoad(last());
    expect(onLoaded).toHaveBeenCalledTimes(1);
  });

  it('reports loaded and settled without fetching', () => {
    const { store } = makeStore();
    expect(store.loaded(1)).toBe(false);
    expect(store.settled(1)).toBe(false);
    expect(created).toHaveLength(0);
    store.get(1);
    expect(store.settled(1)).toBe(false);
    finishLoad(last());
    expect(store.loaded(1)).toBe(true);
    expect(store.settled(1)).toBe(true);
  });

  it('counts a failed id as settled, so nothing waits on it', () => {
    const { store } = makeStore();
    store.get(1);
    failLoad(last());
    expect(store.loaded(1)).toBe(false);
    expect(store.settled(1)).toBe(true);
  });

  it('does not let a stale failure drop the entry that replaced it', () => {
    const { store } = makeStore();
    store.get(1);
    const stale = last();
    store.clear();
    store.get(1);
    const fresh = last();
    failLoad(stale);
    expect(store.peek(1)).toBe(fresh);
  });

  it('does not re-fetch an id that is already cached', () => {
    const { store } = makeStore();
    store.get(1);
    expect(store.warm(1)).toBe(false);
    expect(created).toHaveLength(1);
  });

  it('reports whether an id is known, so a preload pass can skip it', () => {
    const { store } = makeStore();
    expect(store.known(1)).toBe(false);
    store.get(1);
    expect(store.known(1)).toBe(true);
  });
});

describe('ThumbStore full-res preload guard', () => {
  // Regression: the idle preloader shared the visible tier, so once the zoom
  // crossed into full-res every idle pass warmed up to 64 OFF-SCREEN cells with
  // uncapped originals — potentially gigabytes fetched for cells the user may
  // never see, since the LRU bound counts entries rather than bytes.
  it('refuses to warm off-view thumbnails while in the full-res tier', () => {
    const { store } = makeStore();
    store.setFullRes(true);
    expect(store.warm(1)).toBe(false);
    expect(created).toHaveLength(0);
  });

  it('still loads full-res on demand for a cell that is actually on screen', () => {
    const { store } = makeStore();
    store.setFullRes(true);
    expect(store.get(1)).toBeNull();
    expect(created).toHaveLength(1);
    expect(last().src).toContain('/image');
  });

  it('resumes warming once the zoom drops back to the capped tier', () => {
    const { store } = makeStore();
    store.setFullRes(true);
    expect(store.warm(1)).toBe(false);
    store.setFullRes(false);
    expect(store.warm(1)).toBe(true);
    expect(last().src).toContain('/thumbnail');
  });

  it('bounds retention far more tightly in the full-res tier', () => {
    const { store } = makeStore();
    expect(store.capacity).toBe(MAX_THUMBS);
    store.setFullRes(true);
    expect(store.capacity).toBe(MAX_THUMBS_FULL_RES);
    expect(MAX_THUMBS_FULL_RES).toBeLessThan(MAX_THUMBS);
  });
});

describe('ThumbStore resolution tier', () => {
  it('reports whether the tier actually changed', () => {
    const { store } = makeStore();
    expect(store.isFullRes).toBe(false);
    expect(store.setFullRes(false)).toBe(false);
    expect(store.setFullRes(true)).toBe(true);
    expect(store.isFullRes).toBe(true);
    expect(store.setFullRes(true)).toBe(false);
  });

  it('drops the cache when the tier flips so cells reload at the new resolution', () => {
    const { store } = makeStore();
    store.get(1);
    finishLoad(last());
    expect(store.size).toBe(1);
    store.setFullRes(true);
    expect(store.size).toBe(0);
    // The next request re-fetches, now at full resolution.
    store.get(1);
    expect(last().src).toContain('/image');
  });

  it('leaves the cache alone when the tier is unchanged', () => {
    const { store } = makeStore();
    store.get(1);
    finishLoad(last());
    store.setFullRes(false);
    expect(store.size).toBe(1);
  });
});

describe('ThumbStore LRU', () => {
  /** Fill the store with `n` decoded thumbnails, ids 0..n-1. */
  function fill(store: ThumbStore, n: number): void {
    for (let i = 0; i < n; i++) {
      store.get(i);
      finishLoad(last());
    }
  }

  it('evicts the oldest quarter once at capacity', () => {
    const { store } = makeStore();
    store.setFullRes(true); // the small cap, so the test stays fast
    fill(store, MAX_THUMBS_FULL_RES);
    expect(store.full).toBe(true);
    // The next insertion triggers one eviction pass down to 75%.
    store.get(9999);
    expect(store.size).toBe(Math.floor(MAX_THUMBS_FULL_RES * 0.75) + 1);
    // The oldest ids went; the newest stayed.
    expect(store.peek(0)).toBeUndefined();
    expect(store.peek(MAX_THUMBS_FULL_RES - 1)).toBeDefined();
  });

  it('bumps recency on a visible paint, so an on-screen thumb outlives an old one', () => {
    const { store } = makeStore();
    store.setFullRes(true);
    fill(store, MAX_THUMBS_FULL_RES);
    // Id 0 is the oldest — paint it, which re-inserts it as the newest.
    store.get(0);
    store.get(9999); // force the eviction pass
    expect(store.peek(0)).toBeDefined();
    // Id 1 is now the oldest and goes instead.
    expect(store.peek(1)).toBeUndefined();
  });

  it('does not bump recency on a peek, so a pure read cannot save a stale entry', () => {
    // Only a *painted* thumbnail earns its place; the first-view gate reads the
    // cache without painting, and must not thereby protect an off-view warm-up
    // from eviction.
    const { store } = makeStore();
    store.setFullRes(true);
    fill(store, MAX_THUMBS_FULL_RES);
    expect(store.peek(0)).toBeDefined();
    store.get(9999); // force the eviction pass
    expect(store.peek(0)).toBeUndefined();
  });

  it('reports fullness against the live tier capacity', () => {
    const { store } = makeStore();
    store.setFullRes(true);
    expect(store.full).toBe(false);
    fill(store, MAX_THUMBS_FULL_RES);
    expect(store.full).toBe(true);
  });
});

describe('ThumbStore retry backoff', () => {
  // Regression: a failed id went into a permanent set cleared only by a
  // projection switch or a tier crossing, so one transient 502 left that bin
  // rendered as flat density shading for the rest of the session.
  it('records a failure and does not refetch immediately', () => {
    const { store } = makeStore();
    store.get(1);
    failLoad(last());
    expect(store.failed(1)).toBe(true);
    expect(store.get(1)).toBeNull();
    expect(created).toHaveLength(1);
  });

  it('repaints on a visible failure so the frame stops waiting on that cell', () => {
    const { store, onLoaded } = makeStore();
    store.get(1);
    failLoad(last());
    expect(onLoaded).toHaveBeenCalledTimes(1);
  });

  it('stays silent when an off-view warm-up fails', () => {
    const { store, onLoaded } = makeStore();
    store.warm(1);
    failLoad(last());
    expect(onLoaded).not.toHaveBeenCalled();
  });

  it('retries once the backoff has expired', () => {
    const { store, clock } = makeStore();
    store.get(1);
    failLoad(last());
    clock.t += THUMB_RETRY_BACKOFF_MS[0] - 1;
    expect(store.get(1)).toBeNull();
    expect(created).toHaveLength(1);
    clock.t += 1;
    expect(store.get(1)).toBeNull();
    expect(created).toHaveLength(2);
  });

  it('backs off further after each successive failure', () => {
    const { store, clock } = makeStore();
    store.get(1);
    for (const backoff of THUMB_RETRY_BACKOFF_MS) {
      failLoad(last());
      // Just short of the window: no retry.
      clock.t += backoff - 1;
      store.get(1);
      const before = created.length;
      // Just past it: one retry.
      clock.t += 1;
      store.get(1);
      expect(created.length).toBe(before + 1);
    }
  });

  it('gives up after the attempts are spent rather than retrying forever', () => {
    const { store, clock } = makeStore();
    store.get(1);
    for (const backoff of THUMB_RETRY_BACKOFF_MS) {
      failLoad(last());
      clock.t += backoff;
      store.get(1);
    }
    // The last scheduled retry has now also failed.
    failLoad(last());
    const spent = created.length;
    clock.t += 1e9;
    store.get(1);
    expect(created).toHaveLength(spent);
    expect(store.failed(1)).toBe(true);
  });

  it('keeps reporting a failed id as failed through the backoff', () => {
    // The first-view gate uses this to decide whether to hold the cover: a cell
    // that failed once must never hold it across the retry schedule.
    const { store, clock } = makeStore();
    store.get(1);
    failLoad(last());
    clock.t += THUMB_RETRY_BACKOFF_MS[0] + 1;
    expect(store.failed(1)).toBe(true);
  });

  it('clears the failure history once a retry succeeds', () => {
    const { store, clock } = makeStore();
    store.get(1);
    failLoad(last());
    clock.t += THUMB_RETRY_BACKOFF_MS[0];
    store.get(1);
    finishLoad(last());
    expect(store.failed(1)).toBe(false);
    // A later blip gets the full schedule again, not a spent one.
    store.clear();
    store.get(1);
    failLoad(last());
    clock.t += THUMB_RETRY_BACKOFF_MS[0];
    const before = created.length;
    store.get(1);
    expect(created.length).toBe(before + 1);
  });

  it('does not warm an id whose backoff is still pending', () => {
    const { store } = makeStore();
    store.warm(1);
    failLoad(last());
    expect(store.warm(1)).toBe(false);
    expect(created).toHaveLength(1);
  });
});

describe('ThumbStore waveform tinting', () => {
  /** A minimal 2D-context stub recording the composite fill. */
  function stubCanvas() {
    const calls: string[] = [];
    const ctx = {
      globalCompositeOperation: '',
      fillStyle: '',
      drawImage: () => calls.push('drawImage'),
      fillRect: () => calls.push('fillRect'),
    };
    const canvas = {
      width: 0,
      height: 0,
      getContext: () => ctx,
    } as unknown as HTMLCanvasElement;
    vi.spyOn(document, 'createElement').mockReturnValue(canvas);
    return { canvas, ctx, calls };
  }

  it('composites the accent through the mask at the source size', () => {
    const { ctx, calls } = stubCanvas();
    const { store } = makeStore({ accent: () => '#00ff00' });
    const src = { naturalWidth: 32, naturalHeight: 16 } as HTMLImageElement;
    const tinted = store.tinted(1, src);
    expect(tinted.width).toBe(32);
    expect(tinted.height).toBe(16);
    expect(calls).toEqual(['drawImage', 'fillRect']);
    expect(ctx.globalCompositeOperation).toBe('source-in');
    expect(ctx.fillStyle).toBe('#00ff00');
  });

  it('builds the tint once per clip', () => {
    stubCanvas();
    const { store } = makeStore();
    const src = { naturalWidth: 8, naturalHeight: 8 } as HTMLImageElement;
    expect(store.tinted(1, src)).toBe(store.tinted(1, src));
    expect(document.createElement).toHaveBeenCalledTimes(1);
  });

  it('re-tints against the live accent after a theme flip', () => {
    stubCanvas();
    let accent = '#111111';
    const { store } = makeStore({ accent: () => accent });
    const src = { naturalWidth: 8, naturalHeight: 8 } as HTMLImageElement;
    store.tinted(1, src);
    accent = '#222222';
    store.clearTinted();
    const ctx = store.tinted(1, src).getContext('2d')!;
    expect(ctx.fillStyle).toBe('#222222');
  });

  it('keeps the raw masks when only the tints are dropped', () => {
    stubCanvas();
    const { store } = makeStore();
    store.get(1);
    finishLoad(last());
    store.clearTinted();
    // The mask itself is theme-agnostic, so nothing needs re-fetching.
    expect(store.size).toBe(1);
  });

  it('falls back to a blank canvas when no 2D context is available', () => {
    const canvas = { width: 0, height: 0, getContext: () => null } as unknown as HTMLCanvasElement;
    vi.spyOn(document, 'createElement').mockReturnValue(canvas);
    const { store } = makeStore();
    const src = { naturalWidth: 4, naturalHeight: 4 } as HTMLImageElement;
    expect(() => store.tinted(1, src)).not.toThrow();
  });

  it('handles a zero-sized mask without producing a zero-dimension canvas', () => {
    stubCanvas();
    const { store } = makeStore();
    const tinted = store.tinted(1, { naturalWidth: 0, naturalHeight: 0 } as HTMLImageElement);
    expect(tinted.width).toBe(1);
    expect(tinted.height).toBe(1);
  });
});

describe('ThumbStore trim', () => {
  it('keeps only the most recently painted thumbnails', () => {
    const { store } = makeStore();
    for (let id = 0; id < 5; id++) store.get(id);
    // A paint bumps 0 to the newest.
    finishLoad(created[0]);
    store.get(0);
    store.trim(2);
    expect(store.size).toBe(2);
    expect(store.peek(0)).toBeDefined();
    expect(store.peek(4)).toBeDefined();
    expect(store.peek(1)).toBeUndefined();
  });
});

describe('ThumbStore clear', () => {
  it('drops images, tints and failure history together', () => {
    stubDocumentCanvas();
    const { store } = makeStore();
    store.get(1);
    finishLoad(last());
    store.get(2);
    failLoad(last());
    expect(store.size).toBe(1);
    expect(store.failed(2)).toBe(true);

    store.clear();
    expect(store.size).toBe(0);
    expect(store.failed(2)).toBe(false);
    // A cleared id fetches again straight away.
    store.get(2);
    expect(store.known(2)).toBe(true);
  });

  /** Minimal canvas stub for the clear test, which never inspects the context. */
  function stubDocumentCanvas() {
    vi.spyOn(document, 'createElement').mockReturnValue({
      width: 0,
      height: 0,
      getContext: () => null,
    } as unknown as HTMLCanvasElement);
  }
});

/** Encode a ``GET /api/medias/thumbnails`` body: ``[mimetype, bytes]`` per id,
 *  ``null`` for an id with no thumbnail. */
function batchBody(entries: ([string, number[]] | null)[]): ArrayBuffer {
  const parts: number[] = [];
  for (const entry of entries) {
    const [type, bytes] = entry ?? ['', []];
    const n = bytes.length;
    parts.push((n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255, type.length);
    for (const ch of type) parts.push(ch.charCodeAt(0));
    parts.push(...bytes);
  }
  return new Uint8Array(parts).buffer;
}

describe('parseThumbBatch', () => {
  it('splits a body into its entries in order, empty ones as null', () => {
    const body = batchBody([['image/jpeg', [1, 2, 3]], null, ['image/png', [9]]]);
    const entries = parseThumbBatch(body, 3);
    expect(entries.map((e) => e && { type: e.type, bytes: [...e.bytes] })).toEqual([
      { type: 'image/jpeg', bytes: [1, 2, 3] },
      null,
      { type: 'image/png', bytes: [9] },
    ]);
  });

  it('answers null for every entry a short body fails to cover', () => {
    const whole = new Uint8Array(batchBody([['image/jpeg', [1, 2, 3]], ['image/png', [4, 5]]]));
    const cut = whole.slice(0, whole.length - 1).buffer;
    const entries = parseThumbBatch(cut, 3);
    expect(entries[0]?.type).toBe('image/jpeg');
    expect(entries.slice(1)).toEqual([null, null]);
  });
});

describe('ThumbStore batching (#5637)', () => {
  /** Answers to hand back, one per `fetch` call, and the calls made. */
  let fetchCalls: { url: string; priority: string | undefined }[];
  let answer: (url: string) => Promise<Response>;
  let revoked: string[];

  beforeEach(() => {
    fetchCalls = [];
    revoked = [];
    // Each id's thumbnail is its own id as one byte, typed by the id's parity.
    answer = async (url) => {
      const ids = new URL(url).searchParams.get('ids')!.split(',').map(Number);
      const body = batchBody(ids.map((id) => [id % 2 ? 'image/png' : 'image/jpeg', [id & 255]]));
      return { ok: true, status: 200, arrayBuffer: async () => body } as unknown as Response;
    };
    let n = 0;
    vi.spyOn(URL, 'createObjectURL').mockImplementation(() => `blob:thumb-${n++}`);
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation((url) => void revoked.push(url));
  });

  function makeBatchStore() {
    const onLoaded = vi.fn();
    const clock = { t: 1_000 };
    const store = new ThumbStore({
      mediaUrl: (path) => `https://host${path}?ds=7`,
      batchUrl: (ids) => `https://host/api/medias/thumbnails?ids=${ids.join(',')}&ds=7`,
      fetch: (url, init) => {
        fetchCalls.push({ url, priority: init.priority });
        return answer(url);
      },
      onLoaded,
      accent: () => '#ff0000',
      now: () => clock.t,
    });
    return { store, onLoaded, clock };
  }

  /** Let the end-of-task flush send, and the stubbed fetch chain land. */
  const settle = async () => {
    for (let i = 0; i < 6; i++) await Promise.resolve();
  };

  const idsOf = (call: { url: string }) => new URL(call.url).searchParams.get('ids');

  it('sends the thumbnails asked for in one task as one sorted, high-priority request', async () => {
    const { store } = makeBatchStore();
    for (const id of [30, 4, 17]) expect(store.get(id)).toBeNull();
    // Nothing is fetched until the task ends, and no image has a URL yet.
    expect(fetchCalls).toHaveLength(0);
    expect(created.every((img) => img.src === '')).toBe(true);
    await settle();
    expect(fetchCalls).toEqual([{ url: expect.any(String), priority: 'high' }]);
    expect(idsOf(fetchCalls[0])).toBe('4,17,30');
  });

  it('hands each image its bytes and repaints as each lands', async () => {
    const { store, onLoaded } = makeBatchStore();
    store.get(4);
    store.get(17);
    await settle();
    const [four, seventeen] = created;
    expect(four.src).toMatch(/^blob:/);
    expect(seventeen.src).toMatch(/^blob:/);
    finishLoad(four);
    expect(onLoaded).toHaveBeenCalledTimes(1);
    expect(store.get(4)).toBe(four);
    // The blob URL is let go once the image has decoded.
    expect(revoked).toEqual([four.src]);
  });

  it('splits a big ask into batches of the batch size', async () => {
    const { store } = makeBatchStore();
    const n = THUMB_BATCH_SIZE * 2 + 3;
    for (let id = 0; id < n; id++) store.get(id);
    await settle();
    expect(fetchCalls.map((c) => idsOf(c)!.split(',').length)).toEqual([THUMB_BATCH_SIZE, THUMB_BATCH_SIZE, 3]);
  });

  it('sends warm-ups in their own low-priority batch', async () => {
    const { store, onLoaded } = makeBatchStore();
    store.get(1);
    store.warm(2);
    await settle();
    expect(fetchCalls.map((c) => [idsOf(c), c.priority])).toEqual([
      ['1', 'high'],
      ['2', 'low'],
    ]);
    finishLoad(created[1]);
    expect(onLoaded).not.toHaveBeenCalled();
  });

  it('moves a queued warm-up into the visible batch when a cell asks for it first', async () => {
    const { store, onLoaded } = makeBatchStore();
    store.warm(2);
    store.get(2);
    await settle();
    expect(fetchCalls.map((c) => [idsOf(c), c.priority])).toEqual([['2', 'high']]);
    finishLoad(created[0]);
    expect(onLoaded).toHaveBeenCalledTimes(1);
  });

  it('fails an id the batch had no thumbnail for, so nothing waits on it', async () => {
    answer = async () =>
      ({ ok: true, status: 200, arrayBuffer: async () => batchBody([['image/png', [1]], null]) }) as unknown as Response;
    const { store, onLoaded } = makeBatchStore();
    store.get(1);
    store.get(2);
    await settle();
    expect(store.failed(2)).toBe(true);
    expect(store.settled(2)).toBe(true);
    expect(store.failed(1)).toBe(false);
    expect(onLoaded).toHaveBeenCalledTimes(1);
  });

  it('fails the whole batch when the request does', async () => {
    answer = async () => ({ ok: false, status: 500 }) as unknown as Response;
    const { store } = makeBatchStore();
    store.get(1);
    store.get(2);
    await settle();
    expect(store.failed(1) && store.failed(2)).toBe(true);
  });

  it('retries a failed id in a later batch once its backoff has passed', async () => {
    answer = async () => {
      throw new Error('offline');
    };
    const { store, clock } = makeBatchStore();
    store.get(1);
    await settle();
    expect(store.failed(1)).toBe(true);
    store.get(1);
    await settle();
    expect(fetchCalls).toHaveLength(1);
    clock.t += THUMB_RETRY_BACKOFF_MS[0];
    store.get(1);
    await settle();
    expect(fetchCalls).toHaveLength(2);
  });

  it('never batches the full-res tier', async () => {
    const { store } = makeBatchStore();
    store.setFullRes(true);
    store.get(1);
    expect(last().src).toBe('https://host/api/medias/1/image?ds=7');
    await settle();
    expect(fetchCalls).toHaveLength(0);
  });

  it('does not send an id cleared before its batch went out', async () => {
    const { store } = makeBatchStore();
    store.get(1);
    store.clear();
    await settle();
    expect(fetchCalls).toHaveLength(0);
  });

  it('ignores a batch entry whose image was cleared while it was out', async () => {
    let release!: (r: Response) => void;
    answer = () => new Promise<Response>((resolve) => (release = resolve));
    const { store, onLoaded } = makeBatchStore();
    store.get(1);
    await settle();
    const stale = created[0];
    store.clear();
    release({ ok: true, status: 200, arrayBuffer: async () => batchBody([['image/png', [1]]]) } as unknown as Response);
    await settle();
    expect(stale.src).toBe('');
    expect(store.failed(1)).toBe(false);
    expect(onLoaded).not.toHaveBeenCalled();
  });
});
