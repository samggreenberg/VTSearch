import { Injectable, OnDestroy } from '@angular/core';

/**
 * Holds the next few review images in memory so a vote swaps in bytes that are
 * already fetched and decoded (#3896).
 *
 * **Why this cannot be a plain `<link rel=prefetch>` or a throwaway `Image()`.**
 * `/api/medias/<id>/image` answers `Cache-Control: no-cache` **with no `ETag`
 * and no `Last-Modified`** (measured; pinned by `TestMediaImage` in
 * `tests/core/test_medias.py`). `no-cache` lets the browser keep the bytes but
 * forbids reusing them without revalidating first, and a response carrying no
 * validator cannot be revalidated conditionally — so the revalidation is a
 * fresh download of the whole body. Warming the HTTP cache therefore buys
 * nothing reliable, and this store holds the bytes itself, as an object URL the
 * `<img>` can be pointed at.
 *
 * That header is not an oversight to be optimised away: an item's bytes change
 * under a stable URL when it is cropped or cleaned, which is what makes a
 * long-lived cache entry wrong here. This store keeps the same property by
 * holding bytes only across a single advance (see below).
 *
 * **What it is scoped to.** Entries are keyed by the exact URL string, which
 * `ActiveContextService.mediaUrl()` builds with `dataset_id` / `detector_id` /
 * `variant` in the query. A cached blob can only ever be handed back for the
 * URL it was fetched from, so a prefetch can never paint another dataset's item
 * — the failure that would be much worse than the pause being fixed.
 *
 * **Entries are single-use.** `resolve()` hands an entry out once and forgets
 * it. That is what keeps `no-store`'s reason intact: the bytes live for the few
 * seconds between "the reviewer is looking at item N" and "item N+1 is on
 * screen", never across a second visit to the same id, so an item edited in
 * between is re-fetched like any other. The previous consumed object URL is
 * revoked when the next one is handed out rather than at hand-out time, because
 * the `<img>` showing it is still using it.
 *
 * Everything here degrades to "no prefetch": a failed fetch, an aborted one, or
 * an environment without `URL.createObjectURL` all leave {@link resolve}
 * returning the network URL it was given.
 */

/**
 * How many items ahead of the one on screen the review flows warm.
 *
 * Two, not one. With one, a reviewer who votes before item N+1's bytes land
 * (a fast voter, a large image) always meets an in-flight fetch, and the fetch
 * for N+2 cannot even start until N+1 is on screen. With two, N+2 starts a full
 * think-time earlier. Deeper buys little: every step further ahead is a guess
 * that one more vote leaves the ranking alone, which a learned re-sort after
 * each vote routinely breaks.
 */
export const PREFETCH_DEPTH = 2;

/** How many warmed images to retain: the {@link PREFETCH_DEPTH} upcoming items
 *  plus the one on screen, whose bytes must survive a retarget that lands
 *  before the viewer has claimed them. A safety bound, not a cache — the
 *  targets passed to {@link MediaPrefetchService.prefetch} already drop
 *  everything else. */
export const PREFETCH_CAPACITY = PREFETCH_DEPTH + 1;

interface InFlight {
  controller: AbortController;
  /** Resolves to the object URL once the bytes land, or `null` on any failure. */
  done: Promise<string | null>;
  /** Set by {@link MediaPrefetchService.claim}: the viewer is waiting on these
   *  bytes, so they go straight to it instead of into the ready map, and no
   *  retarget may abort the fetch. */
  claimed: boolean;
}

@Injectable({ providedIn: 'root' })
export class MediaPrefetchService implements OnDestroy {
  /** url -> object URL, insertion-ordered so the oldest evicts first. */
  private readonly ready = new Map<string, string>();
  private readonly inFlight = new Map<string, InFlight>();
  /** Handed out by the last {@link resolve}/{@link claim}; revoked when the
   *  next one is handed out. */
  private lastConsumed: string | null = null;
  /** The urls {@link prefetch} was last asked for, in priority order. Fetched
   *  one at a time, see {@link pump}. */
  private targets: string[] = [];

  /** True when this environment can hold bytes for us (jsdom, by default,
   *  cannot). Read once per call rather than cached, so a test that stubs
   *  `createObjectURL` after construction still gets the real path. */
  private get canStore(): boolean {
    return typeof URL !== 'undefined' && typeof URL.createObjectURL === 'function';
  }

  /**
   * Make `targets` the set of images being warmed, in priority order (the item
   * the next vote lands on first), and forget everything else except `keep`.
   *
   * This is what a review flow calls whenever its *prediction* of the next
   * items changes — a new selection, but equally a learned re-sort landing, a
   * vote reconciling, or the select mode or cut moving while the same item
   * stays on screen. A stale prediction is dropped rather than left to finish:
   * its fetch is aborted, because on an ~11 Mbps tunnel it would otherwise
   * compete for bandwidth with the image the reviewer is about to need, and its
   * held bytes are revoked.
   *
   * `keep` is the item on screen. Its bytes may still be held or in flight when
   * this runs — the selection effect that calls this and the viewer's effect
   * that consumes them run in the same flush, in either order — and dropping
   * them there would turn a hit into a fresh download.
   *
   * Targets are fetched **one at a time**, in order. Two parallel fetches split
   * the link, so N+1 would land later than it does alone, and N+1 is the one a
   * fast voter will need first.
   */
  prefetch(targets: readonly string[], keep: readonly string[] = []): void {
    const wanted = new Set<string>([...targets, ...keep].filter(Boolean));
    for (const [url, entry] of this.inFlight) {
      if (wanted.has(url) || entry.claimed) continue;
      entry.controller.abort();
      this.inFlight.delete(url);
    }
    for (const [url, objectUrl] of this.ready) {
      if (wanted.has(url)) continue;
      this.ready.delete(url);
      this.revoke(objectUrl);
    }
    this.targets = targets.filter(Boolean);
    this.pump();
  }

  /**
   * Start fetching `url` if it is not already held or in flight.
   *
   * Fire-and-forget by design: nothing awaits this, and a failure is silent.
   * The caller's next {@link resolve} simply falls back to the network URL,
   * which is exactly the behaviour that existed before this service. Review
   * flows go through {@link prefetch}, which also orders and retires targets;
   * this is the unordered primitive underneath it.
   */
  warm(url: string): void {
    if (!url || !this.canStore) return;
    if (this.ready.has(url) || this.inFlight.has(url)) return;
    const controller = new AbortController();
    const entry: InFlight = { controller, claimed: false, done: Promise.resolve(null) };
    this.inFlight.set(url, entry);
    entry.done = fetch(url, { signal: controller.signal, credentials: 'same-origin' })
      .then((response) => (response.ok ? response.blob() : null))
      .then((blob) => {
        // `clear()` or a retarget may have run while the bytes were in the air;
        // its abort does not reliably beat a resolved fetch. Leave that case
        // entirely alone: the entry in the map (if any) is somebody else's, and
        // forgetting it would strand their fetch.
        if (this.inFlight.get(url) !== entry) return null;
        // Ours, so retire it whatever the outcome. Forgetting to do so on the
        // failure path leaves the url permanently "in flight" and silently
        // unprefetchable for the rest of the session.
        this.inFlight.delete(url);
        if (!blob) return null;
        const objectUrl = URL.createObjectURL(blob);
        if (entry.claimed) return objectUrl;
        this.ready.set(url, objectUrl);
        this.predecode(objectUrl);
        this.evict();
        return objectUrl;
      })
      .catch(() => {
        if (this.inFlight.get(url) === entry) this.inFlight.delete(url);
        return null;
      })
      .finally(() => this.pump());
  }

  /**
   * The source an `<img>` should use for `url`: the warmed object URL when one
   * is held, otherwise `url` itself.
   */
  resolve(url: string): string {
    const held = this.ready.get(url);
    if (held === undefined) return url;
    this.ready.delete(url);
    this.handOut(held);
    return held;
  }

  /**
   * When `url` is still being prefetched, take over that fetch: the returned
   * promise resolves to its object URL (handed out exactly as {@link resolve}
   * would), or to `null` if it fails or is cleared, in which case the caller
   * should fall back to the network URL. Returns `null` when nothing is in
   * flight for `url`.
   *
   * This is the fast-voter case: the vote lands before the next image's bytes
   * do. Starting a second download of the same image then would throw away
   * everything already transferred and split the link with the first one
   * until it finished.
   */
  claim(url: string): Promise<string | null> | null {
    const entry = this.inFlight.get(url);
    if (!entry) return null;
    entry.claimed = true;
    return entry.done.then((objectUrl) => {
      if (!objectUrl) return null;
      if (!entry.claimed) {
        // Released while in flight. If the bytes landed after the release they
        // went to `ready` like any warm; if they landed before it, nobody holds
        // them any more.
        if (![...this.ready.values()].includes(objectUrl)) this.revoke(objectUrl);
        return null;
      }
      // Claimed bytes bypass `ready`, so hand them out here.
      this.handOut(objectUrl);
      return objectUrl;
    });
  }

  /**
   * Give up a {@link claim}: the viewer moved on before the bytes landed. The
   * fetch becomes an ordinary target again, so the next {@link prefetch} keeps
   * it only if it is still wanted.
   */
  release(url: string): void {
    const entry = this.inFlight.get(url);
    if (entry) entry.claimed = false;
  }

  /** Drop everything: in-flight fetches, warmed bytes, and the object URL the
   *  caller was last handed. For teardown. */
  clear(): void {
    this.targets = [];
    for (const entry of this.inFlight.values()) {
      entry.claimed = false;
      entry.controller.abort();
    }
    this.inFlight.clear();
    for (const objectUrl of this.ready.values()) this.revoke(objectUrl);
    this.ready.clear();
    this.revokeLastConsumed();
  }

  ngOnDestroy(): void {
    this.clear();
  }

  /**
   * Start the first wanted target that is neither held nor in flight, if
   * nothing is in flight. One fetch at a time; see {@link prefetch}.
   */
  private pump(): void {
    if (this.inFlight.size > 0) return;
    const next = this.targets.find((url) => !this.ready.has(url));
    if (next) this.warm(next);
  }

  private handOut(objectUrl: string): void {
    this.revokeLastConsumed();
    this.lastConsumed = objectUrl;
  }

  private evict(): void {
    while (this.ready.size > PREFETCH_CAPACITY) {
      const oldest = this.ready.keys().next();
      if (oldest.done) return;
      const objectUrl = this.ready.get(oldest.value);
      this.ready.delete(oldest.value);
      if (objectUrl) this.revoke(objectUrl);
    }
  }

  private revokeLastConsumed(): void {
    if (this.lastConsumed) this.revoke(this.lastConsumed);
    this.lastConsumed = null;
  }

  /**
   * Ask the browser to decode the bytes now rather than when the `<img>`
   * mounts.
   *
   * Best effort, and honestly so: the issue's traces put several hundred ms of
   * the felt pause after the bytes could have arrived — decode, layout, paint —
   * and this is the half of that we can move off the critical path. Whether the
   * decoded frame is actually reused when a *different* element later points at
   * the same object URL is a browser-internals question this has not measured;
   * it costs one detached image per warmed entry either way, which is why the
   * store stays small. A failure (a decode-less environment, an undecodable
   * payload) is ignored: the bytes are still warm, which was the main prize.
   */
  private predecode(objectUrl: string): void {
    if (typeof Image !== 'function') return;
    try {
      const img = new Image();
      img.src = objectUrl;
      void img.decode?.().catch(() => undefined);
    } catch {
      // ignored, see above
    }
  }

  private revoke(objectUrl: string): void {
    if (typeof URL !== 'undefined' && typeof URL.revokeObjectURL === 'function') {
      URL.revokeObjectURL(objectUrl);
    }
  }
}

/**
 * The image URLs for an upcoming-items queue, cut at the first item that is not
 * an image (or whose metadata is not loaded yet).
 *
 * Cut rather than filtered: the store fetches in queue order because that is
 * the order the reviewer needs them in, and an item past a non-image one is an
 * item two votes away presented as if it were one.
 */
export function imageUrlsWhileImages(
  ids: readonly number[],
  mediaType: (id: number) => string | undefined,
  mediaUrl: (path: string) => string,
): string[] {
  const urls: string[] = [];
  for (const id of ids) {
    if (mediaType(id) !== 'image') break;
    urls.push(mediaUrl(`/api/medias/${id}/image`));
  }
  return urls;
}
