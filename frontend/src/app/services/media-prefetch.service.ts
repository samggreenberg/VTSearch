import { Injectable, OnDestroy } from '@angular/core';

/**
 * Warms the browser's HTTP cache with the next few review images, so a vote
 * swaps in an image the browser already holds (#3896).
 *
 * **Why the HTTP cache is enough.** `/api/medias/<id>/image` answers
 * `Cache-Control: private, max-age=86400` with an `ETag` (#5648; pinned by
 * `TestMediaImage` in `tests/core/test_medias.py`). A warm is an ordinary
 * `fetch` of the exact URL the viewer's `<img>` will ask for, its body read to
 * the end so it lands in the cache, and the `<img>` is then answered from the
 * cache without a request. Measured in Chromium against a slow server sending
 * those headers, the `<img>` made no second request whether the warm had
 * finished or was still in flight when it mounted (the browser joins the
 * download already under way), so nothing here has to hand bytes to the viewer.
 *
 * Before #5648 the route answered `no-cache` with no validator, so a warmed
 * cache entry could not be reused, and this service held the bytes itself as
 * object URLs, handed them out once, and let the viewer claim a fetch still in
 * flight. A cacheable route makes all of that the browser's job.
 *
 * **What it is scoped to.** A warm is keyed by the exact URL string, which
 * `ActiveContextService.mediaUrl()` builds with `dataset_id` / `detector_id` /
 * `variant` in the query, so the cache entry it fills is the one the viewer
 * reads for that item of that dataset, and no other.
 *
 * **What is left here** is choosing what to fetch and when: the review flows
 * hand over their prediction of the next items, this fetches them one at a
 * time in that order, and it aborts a fetch the prediction has dropped so it
 * does not compete for bandwidth with the image the reviewer needs next.
 *
 * Everything degrades to "no prefetch": a failed or aborted fetch, or an
 * environment without `fetch`, leaves the viewer fetching the image itself,
 * which is what it did before this service existed.
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

@Injectable({ providedIn: 'root' })
export class MediaPrefetchService implements OnDestroy {
  /** url -> the controller of its fetch, while the fetch is in the air. */
  private readonly inFlight = new Map<string, AbortController>();
  /** Wanted urls whose fetch completed, so the cache holds them and
   *  {@link pump} moves past them. */
  private readonly warmed = new Set<string>();
  /** Wanted urls whose fetch failed, which {@link pump} does not try again
   *  until a retarget drops them (#5099). */
  private readonly failed = new Set<string>();
  /** The urls {@link prefetch} was last asked for, in priority order. Fetched
   *  one at a time, see {@link pump}. */
  private targets: string[] = [];

  /**
   * Make `targets` the set of images being warmed, in priority order (the item
   * the next vote lands on first), and forget everything else except `keep`.
   *
   * This is what a review flow calls whenever its *prediction* of the next
   * items changes: a new selection, but equally a learned re-sort landing, a
   * vote reconciling, or the select mode or cut moving while the same item
   * stays on screen. A stale prediction's fetch is aborted rather than left to
   * finish, because on an ~11 Mbps tunnel it would otherwise compete for
   * bandwidth with the image the reviewer is about to need. An image whose
   * fetch already completed stays in the HTTP cache, so a re-ranking that wants
   * it back costs a cache read, not a download.
   *
   * `keep` is the item on screen. Its fetch may still be in the air, with the
   * viewer's `<img>` riding on it (the selection effect that calls this and the
   * viewer's effect that sets the `<img>` run in the same flush, in either
   * order), and aborting it there would turn a hit into a fresh download.
   *
   * Targets are fetched **one at a time**, in order. Two parallel fetches split
   * the link, so N+1 would land later than it does alone, and N+1 is the one a
   * fast voter will need first.
   */
  prefetch(targets: readonly string[], keep: readonly string[] = []): void {
    const wanted = new Set<string>([...targets, ...keep].filter(Boolean));
    for (const [url, controller] of this.inFlight) {
      if (wanted.has(url)) continue;
      controller.abort();
      this.inFlight.delete(url);
    }
    this.targets = targets.filter(Boolean);
    for (const url of this.warmed) if (!wanted.has(url)) this.warmed.delete(url);
    for (const url of this.failed) if (!wanted.has(url)) this.failed.delete(url);
    this.pump();
  }

  /**
   * Fetch `url` into the HTTP cache, unless it is already warm or in flight.
   *
   * Fire-and-forget by design: nothing awaits this, and a failure is silent.
   * The viewer requests the image itself either way, and the cache answers it
   * when this got there first. Review flows go through {@link prefetch}, which
   * also orders and retires targets; this is the unordered primitive under it.
   */
  warm(url: string): void {
    if (!url || typeof fetch !== 'function') return;
    if (this.warmed.has(url) || this.inFlight.has(url)) return;
    this.failed.delete(url);
    const controller = new AbortController();
    this.inFlight.set(url, controller);
    let response: Promise<Response>;
    try {
      // `same-origin` credentials, like the `<img>`'s own request, so the two
      // share one cache entry.
      response = fetch(url, { signal: controller.signal, credentials: 'same-origin' });
    } catch (err) {
      response = Promise.reject(err);
    }
    void response
      // Read the body to the end: a response nobody reads is not cached whole.
      .then((r) => (r.ok ? r.blob().then(() => true) : false))
      .catch(() => false)
      .then((ok) => {
        // A retarget or `clear()` retired this fetch while it was in the air.
        // Its abort does not reliably beat a resolved fetch, so leave that case
        // alone: the entry under `url`, if any, is somebody else's.
        if (this.inFlight.get(url) !== controller) return;
        this.inFlight.delete(url);
        if (!ok) {
          this.failed.add(url);
          return;
        }
        this.warmed.add(url);
        this.predecode(url);
      })
      .finally(() => this.pump());
  }

  /** Forget everything and abort every fetch in flight. For teardown. */
  clear(): void {
    this.targets = [];
    this.warmed.clear();
    this.failed.clear();
    for (const controller of this.inFlight.values()) controller.abort();
    this.inFlight.clear();
  }

  ngOnDestroy(): void {
    this.clear();
  }

  /**
   * Start the first target that is neither warm nor failed, if nothing is in
   * flight. One fetch at a time; see {@link prefetch}.
   *
   * A target whose fetch failed is skipped while it stays wanted (#5099).
   * Picking it again refetched a url that 404s or errors back to back for as
   * long as the prediction held it: a request storm in the browser, and under
   * jsdom, where the fetch rejects at once, a microtask loop that starved every
   * timer. The viewer still fetches it itself when it comes on screen.
   */
  private pump(): void {
    if (this.inFlight.size > 0) return;
    const next = this.targets.find((url) => !this.warmed.has(url) && !this.failed.has(url));
    if (next) this.warm(next);
  }

  /**
   * Ask the browser to decode the image now rather than when the `<img>`
   * mounts.
   *
   * Best effort. The traces in #3896 put several hundred ms of the felt pause
   * after the bytes could have arrived (decode, layout, paint), and this is the
   * part of that which can move off the critical path. The image is loaded by
   * the same URL the viewer's `<img>` uses, so it is answered from the cache
   * the warm just filled; whether the browser then reuses the decoded frame for
   * the `<img>` is a browser-internals question this has not measured. A
   * failure (an environment with no decode, an undecodable payload) is
   * ignored: the bytes are still in the cache, which was the main prize.
   */
  private predecode(url: string): void {
    if (typeof Image !== 'function') return;
    try {
      const img = new Image();
      img.src = url;
      void img.decode?.().catch(() => undefined);
    } catch {
      // ignored, see above
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
