/**
 * Representative-thumbnail cache for the browse canvas.
 *
 * Every bin the canvas paints in thumbnail mode is drawn with its
 * representative clip's image, so a single frame can ask for hundreds of
 * thumbnails and a pan/zoom session for tens of thousands. This module owns
 * all of the state that makes that affordable:
 *
 * - **A count-bounded LRU** over the decoded {@link HTMLImageElement}s, keyed by
 *   media id. Insertion order *is* the recency order: a visible paint re-inserts
 *   its entry (see {@link ThumbStore.get}), so off-view warm-ups — which never
 *   re-insert — are always first in line for eviction.
 * - **A resolution tier.** Below a zoom threshold cells are painted from the
 *   downscaled ``/thumbnail`` route; past it a cell is drawn wider than that
 *   route's native cap, so the full-res ``/image`` is fetched instead. Crossing
 *   the threshold drops the cache so every cell reloads at the matching
 *   resolution — the two tiers are never mixed in one cache.
 * - **A tinted-mask cache.** Audio waveform thumbnails are theme-agnostic alpha
 *   masks (issue #2369); each is composited to the live accent colour once per
 *   (clip, theme) rather than once per frame.
 * - **Retryable failure records.** A load error is remembered with a backoff so
 *   a transient blip doesn't blank that bin for the rest of the session.
 *
 * It is deliberately free of Angular and of the tile pyramid: it knows how to
 * fetch, retain, evict and tint one image per media id, and nothing about which
 * ids are worth fetching. Deciding *that* is the caller's job (see
 * `prefetch-geometry.ts` for the tile-set half).
 */

/** Longest side (px) the ``/thumbnail`` route caps images at (mirrors
 * ``vtscore``'s ``DEFAULT_MAX_DIM``). Once a cell is drawn wider than this the
 * capped thumbnail would upscale, so at those zoom levels the canvas fetches
 * the full-res ``/image`` instead. */
export const THUMB_NATIVE_MAX_DIM = 384;

/** LRU capacity while the store holds downscaled ``/thumbnail`` images. Each is
 * at most {@link THUMB_NATIVE_MAX_DIM} on its longest side, so a full cache is
 * tens of megabytes — cheap enough to keep a long pan session warm. */
export const MAX_THUMBS = 2048;

/**
 * LRU capacity while the store holds full-res ``/image`` originals.
 *
 * Sharply lower than {@link MAX_THUMBS} because the entries are a different
 * order of magnitude: a photo dataset's originals run to megabytes each, so the
 * 2048-entry bound that keeps the capped tier at tens of megabytes would let
 * the full-res tier retain gigabytes (the cap counts entries, not bytes).
 *
 * The tier only engages once a cell is drawn wider than
 * {@link THUMB_NATIVE_MAX_DIM}, i.e. past ~384 device px across, so a 1080p
 * viewport holds on the order of fifteen cells. This leaves better than ten
 * screens of pan history warm while bounding retention to something a browser
 * tab can hold.
 */
export const MAX_THUMBS_FULL_RES = 192;

/** Fraction of the cache retained by one eviction pass (the oldest quarter is
 *  dropped), so eviction amortizes over many insertions instead of running on
 *  every load once the cache is full. */
const EVICT_RETAIN = 0.75;

/**
 * Backoff (ms) before a failed thumbnail is retried, indexed by how many
 * attempts have already failed. A transient failure — a server restart, a
 * brief network blip, one 502 during a burst of preload fetches — used to
 * blank that bin permanently: the id went into a failure set that was only
 * ever cleared by a projection switch or a tier crossing, so the cell rendered
 * as flat density shading among its thumbnail neighbours for the rest of the
 * session, with no user-visible way to recover short of leaving the view.
 *
 * The schedule grows so a genuinely broken id costs three requests rather than
 * one per frame, and stops entirely after the last entry: a media whose image
 * really is missing (deleted file, unsupported codec) should settle into the
 * flat-density fallback rather than retry forever.
 */
export const THUMB_RETRY_BACKOFF_MS = [2_000, 8_000, 32_000];

/** What the store needs from its host to fetch, repaint and tint. */
export interface ThumbStoreDeps {
  /** Build a dataset-scoped URL for a media endpoint (``/api/medias/…``). */
  mediaUrl(path: string): string;
  /** Schedule a repaint: a *visible* thumbnail has finished decoding, or has
   *  failed and the frame should stop waiting on it. Never called for an
   *  off-view warm-up, which has no cell on screen to repaint. */
  onLoaded(): void;
  /** The live theme's accent colour, used to tint audio waveform masks. Read
   *  per tint rather than captured, so a theme flip (which clears the tinted
   *  cache) re-tints against the new colour. */
  accent(): string;
  /** Clock, injectable so the retry backoff is testable without real timers. */
  now?(): number;
}

/** A remembered load failure and when it may be retried. */
interface FailureRecord {
  /** How many fetches for this id have failed. */
  attempts: number;
  /** Timestamp (ms, {@link ThumbStoreDeps.now}) before which no retry is made.
   *  ``Infinity`` once the attempts are spent — the id is given up on. */
  retryAt: number;
}

export class ThumbStore {
  /** Loaded representative thumbnails, keyed by media id (insertion-ordered LRU). */
  private readonly images = new Map<number, HTMLImageElement>();
  /** Media ids whose thumbnail failed, with their retry backoff, so we neither
   *  refetch every frame nor give up permanently on a transient blip. */
  private readonly failures = new Map<number, FailureRecord>();
  /** Waveform masks composited to the live accent, keyed like {@link images}
   *  and evicted alongside them so a tint can never outlive its source. */
  private readonly tintedImages = new Map<number, HTMLCanvasElement>();
  /** Whether the cache currently holds full-res originals rather than capped
   *  thumbnails. Flipped by {@link setFullRes}, which drops the cache. */
  private fullRes = false;

  constructor(private readonly deps: ThumbStoreDeps) {}

  private now(): number {
    return this.deps.now ? this.deps.now() : Date.now();
  }

  /** Entries retained before eviction, which depends on the live tier. */
  get capacity(): number {
    return this.fullRes ? MAX_THUMBS_FULL_RES : MAX_THUMBS;
  }

  /** How many images are cached (pending loads included). */
  get size(): number {
    return this.images.size;
  }

  /** Whether the cache is at capacity — a warm-up stops here so an off-view
   *  preload can never evict something that is on screen. */
  get full(): boolean {
    return this.images.size >= this.capacity;
  }

  /** Whether the store is filled with full-res originals. */
  get isFullRes(): boolean {
    return this.fullRes;
  }

  /**
   * Whether a cell is drawn wide enough that the capped ``/thumbnail`` would
   * upscale, so the full-res ``/image`` should be fetched instead. Pure, so the
   * threshold is testable without a canvas.
   */
  static wantsFullRes(targetRadius: number, dpr: number): boolean {
    return 2 * targetRadius * dpr > THUMB_NATIVE_MAX_DIM;
  }

  /**
   * Switch the resolution tier, dropping the cache when it actually changes so
   * cells reload at the matching resolution. Returns whether the tier flipped
   * (a cheap no-op otherwise), which the caller can use to decide whether a
   * repaint is owed.
   */
  setFullRes(fullRes: boolean): boolean {
    if (fullRes === this.fullRes) return false;
    this.fullRes = fullRes;
    this.clear();
    return true;
  }

  /**
   * The loaded thumbnail for a representative media id, or null while it loads
   * or if it failed. Kicks off the fetch on first request (and on a retry once
   * a failure's backoff has expired) and repaints when the image arrives.
   */
  get(id: number): HTMLImageElement | null {
    const cached = this.images.get(id);
    if (cached) {
      // Bump recency: re-insert so a thumbnail painted this frame becomes the
      // newest entry. The map is insertion-ordered (see {@link evict}), so this
      // keeps currently-visible thumbnails last in line for eviction — off-view
      // warm-ups (which never bump) are dropped first, so warming the ring can
      // never evict something that's on screen.
      this.images.delete(id);
      this.images.set(id, cached);
      return cached.complete && cached.naturalWidth > 0 ? cached : null;
    }
    this.startLoad(id, false);
    return null;
  }

  /**
   * The cached image for an id *without* bumping its recency or starting a
   * fetch — a pure read, for callers that only want to know whether a load has
   * landed (the first-view gate). Undefined when nothing is cached.
   */
  peek(id: number): HTMLImageElement | undefined {
    return this.images.get(id);
  }

  /** Whether an id is cached or already known-failed, i.e. whether asking for
   *  it again would start a fetch. */
  known(id: number): boolean {
    return this.images.has(id) || this.failures.has(id);
  }

  /**
   * Whether this id has a recorded load failure. True through the whole backoff
   * (not just while a retry is pending), because callers use it to decide
   * whether to *wait* on a thumbnail: a cell that has failed once must not hold
   * the first-view cover up across the retry schedule.
   */
  failed(id: number): boolean {
    return this.failures.has(id);
  }

  /**
   * Warm an off-view thumbnail at low priority, returning whether a fetch was
   * actually started (so callers can budget passes by real network cost).
   *
   * Refused while the store is in the full-res tier. The tier exists because a
   * *visible* giant cell would upscale a capped thumbnail, and the comment
   * justifying it — only a handful of such cells fit on screen — is exactly why
   * warming is wrong there: a ring warm-up fetches on the order of sixty
   * off-screen cells, which at full resolution is potentially gigabytes pulled
   * for cells the user may never pan to. On-demand loading is unaffected, so
   * panning to such a cell still fills it; only the speculative fetch is
   * dropped, and it is dropped where speculation is most expensive.
   */
  warm(id: number): boolean {
    if (this.fullRes) return false;
    return this.startLoad(id, true);
  }

  /**
   * Kick off the fetch for a representative id and stash the pending image in
   * the cache; returns whether a fetch was started. ``preload`` marks an
   * off-view warm-up: it loads at low network priority and does not repaint
   * when it lands (the cell isn't on screen), so it never competes with visible
   * thumbnails. A no-op when the id is already cached, or when a previous
   * failure's backoff has not yet expired.
   */
  private startLoad(id: number, preload: boolean): boolean {
    if (this.images.has(id)) return false;
    const failure = this.failures.get(id);
    if (failure && this.now() < failure.retryAt) return false;

    if (this.full) this.evict();

    const img = new Image();
    img.decoding = 'async';
    if (preload) {
      // Idle warm-up: let the browser schedule it behind visible-thumbnail and
      // tile requests, and skip the repaint — the cell is off-screen, so a later
      // draw picks it up from the cache once the user pans to it.
      img.setAttribute('fetchpriority', 'low');
    } else {
      img.onload = () => {
        // A load that lands clears the failure history: the id is good again,
        // so a later blip gets the full retry schedule rather than resuming a
        // spent one.
        this.failures.delete(id);
        this.deps.onLoaded();
      };
    }
    img.onerror = () => {
      this.images.delete(id);
      this.recordFailure(id, failure);
      // Repaint so the frame stops waiting on this cell (it now falls back to
      // flat density shading) instead of stranding the first-view cover until
      // its backstop timer fires. Warm-ups paint nothing, so they stay silent.
      if (!preload) this.deps.onLoaded();
    };
    // Downscaled /thumbnail by default: a browse projection can hold thousands
    // of points, so painting full-size bitmaps onto every bin would exhaust
    // memory. The /thumbnail route serves the frame for video via the same
    // image_response hook, then downscales it. At the largest zoom levels a cell
    // is drawn wider than the thumbnail's native resolution, so the full-res
    // /image is fetched instead — see {@link wantsFullRes}.
    const endpoint = this.fullRes ? 'image' : 'thumbnail';
    img.src = this.deps.mediaUrl(`/api/medias/${id}/${endpoint}`);
    this.images.set(id, img);
    return true;
  }

  /** Note a failed attempt and schedule (or give up on) the next retry. */
  private recordFailure(id: number, previous: FailureRecord | undefined): void {
    const attempts = (previous?.attempts ?? 0) + 1;
    const backoff = THUMB_RETRY_BACKOFF_MS[attempts - 1];
    this.failures.set(id, {
      attempts,
      // Out of attempts: give up rather than refetch a genuinely missing image
      // once per backoff window forever.
      retryAt: backoff === undefined ? Infinity : this.now() + backoff,
    });
  }

  /** Drop the oldest quarter of cached thumbnails (insertion-ordered LRU). */
  private evict(): void {
    const target = Math.floor(this.capacity * EVICT_RETAIN);
    const toRemove = this.images.size - target;
    let i = 0;
    for (const key of this.images.keys()) {
      if (i++ >= toRemove) break;
      this.images.delete(key);
      // Drop the matching tinted mask (issue #2369) so it can't outlive its raw
      // source and leak; it re-tints on demand if the clip is revisited.
      this.tintedImages.delete(key);
    }
  }

  /**
   * The audio waveform thumbnail tinted to the live theme's accent colour,
   * built once per (clip, theme) and cached (issue #2369).
   *
   * The raw thumbnail is a theme-agnostic alpha mask — a transparent PNG whose
   * only opaque pixels are the wave. Painting it to an offscreen canvas and
   * compositing the accent through ``source-in`` recolours just the wave,
   * leaving the background transparent so the tile's themed surface (filled by
   * the caller) shows through. The tinted cache is cleared on a theme flip and
   * whenever the raw cache is, so it never serves a stale colour.
   */
  tinted(id: number, src: HTMLImageElement): HTMLCanvasElement {
    const cached = this.tintedImages.get(id);
    if (cached) return cached;

    const w = src.naturalWidth || 1;
    const h = src.naturalHeight || 1;
    const off = document.createElement('canvas');
    off.width = w;
    off.height = h;
    const octx = off.getContext('2d');
    if (octx) {
      octx.drawImage(src, 0, 0);
      octx.globalCompositeOperation = 'source-in';
      octx.fillStyle = this.deps.accent();
      octx.fillRect(0, 0, w, h);
    }
    this.tintedImages.set(id, off);
    return off;
  }

  /** Drop the tinted masks only, keeping the raw ones — what a theme flip
   *  needs, since the masks themselves are theme-agnostic. */
  clearTinted(): void {
    this.tintedImages.clear();
  }

  /** Drop everything: images, tints and failure history. Used on a projection
   *  switch, a tier crossing and teardown. */
  clear(): void {
    this.images.clear();
    this.tintedImages.clear();
    this.failures.clear();
  }
}
