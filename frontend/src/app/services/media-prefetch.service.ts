import { Injectable, OnDestroy } from '@angular/core';

/**
 * Holds the *next* review image in memory so a vote swaps in bytes that are
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

/** How many warmed images to retain. Two, not one: a reviewer who votes faster
 *  than a fetch completes would otherwise evict the in-flight entry they are
 *  about to need. Entries are small in number by construction — the store
 *  holds what the *next* click needs, not a history. */
export const PREFETCH_CAPACITY = 2;

@Injectable({ providedIn: 'root' })
export class MediaPrefetchService implements OnDestroy {
  /** url -> object URL, insertion-ordered so the oldest evicts first. */
  private readonly ready = new Map<string, string>();
  private readonly inFlight = new Map<string, AbortController>();
  /** Handed out by the last {@link resolve}; revoked when the next one lands. */
  private lastConsumed: string | null = null;

  /** True when this environment can hold bytes for us (jsdom, by default,
   *  cannot). Read once per call rather than cached, so a test that stubs
   *  `createObjectURL` after construction still gets the real path. */
  private get canStore(): boolean {
    return typeof URL !== 'undefined' && typeof URL.createObjectURL === 'function';
  }

  /**
   * Start fetching `url` if it is not already held or in flight.
   *
   * Fire-and-forget by design: nothing awaits this, and a failure is silent.
   * The caller's next {@link resolve} simply falls back to the network URL,
   * which is exactly the behaviour that existed before this service.
   */
  warm(url: string): void {
    if (!url || !this.canStore) return;
    if (this.ready.has(url) || this.inFlight.has(url)) return;
    const controller = new AbortController();
    this.inFlight.set(url, controller);
    fetch(url, { signal: controller.signal, credentials: 'same-origin' })
      .then((response) => (response.ok ? response.blob() : null))
      .then((blob) => {
        // `clear()` (a dataset switch, a teardown) may have run while the bytes
        // were in the air; its abort does not reliably beat a resolved fetch.
        // Leave that case entirely alone: the controller in the map is somebody
        // else's, and forgetting it would strand their fetch.
        if (this.inFlight.get(url) !== controller) return;
        // Ours, so retire it whatever the outcome. Forgetting to do so on the
        // failure path leaves the url permanently "in flight" and silently
        // unprefetchable for the rest of the session.
        this.inFlight.delete(url);
        if (!blob) return;
        const objectUrl = URL.createObjectURL(blob);
        this.ready.set(url, objectUrl);
        this.predecode(objectUrl);
        this.evict();
      })
      .catch(() => {
        if (this.inFlight.get(url) === controller) this.inFlight.delete(url);
      });
  }

  /**
   * The source an `<img>` should use for `url`: the warmed object URL when one
   * is held, otherwise `url` itself.
   */
  resolve(url: string): string {
    const held = this.ready.get(url);
    if (held === undefined) return url;
    this.ready.delete(url);
    this.revokeLastConsumed();
    this.lastConsumed = held;
    return held;
  }

  /** Drop everything: in-flight fetches, warmed bytes, and the object URL the
   *  caller was last handed. For a dataset/detector switch and teardown. */
  clear(): void {
    for (const controller of this.inFlight.values()) controller.abort();
    this.inFlight.clear();
    for (const objectUrl of this.ready.values()) this.revoke(objectUrl);
    this.ready.clear();
    this.revokeLastConsumed();
  }

  ngOnDestroy(): void {
    this.clear();
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
