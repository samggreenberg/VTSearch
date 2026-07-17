import { EMPTY, Observable, defer, fromEvent, merge, of, timer } from 'rxjs';
import { catchError, distinctUntilChanged, filter, map, repeat, switchMap, tap } from 'rxjs/operators';

/** Sentinel a poll emits when its request errors, so the loop keeps scheduling
 *  instead of tearing down (see {@link adaptivePoll}). Filtered out of output. */
const POLL_SKIP = Symbol('poll-skip');
/** Marks "no response applied yet" so the first poll never counts as unchanged. */
const NO_SIG = Symbol('no-signature');

export interface AdaptivePollConfig<T> {
  /** Delay between polls while the response keeps changing (data is "fresh"). */
  fastMs: number;
  /** Delay between polls once the response has been stable for `rampAfter`
   *  consecutive polls (data is "stale" — a low-frequency heartbeat). */
  slowMs: number;
  /** Consecutive unchanged polls before the cadence eases from `fastMs` toward
   *  `slowMs`. Defaults to 3. */
  rampAfter?: number;
  /** Cheap comparable derived from a response; when it differs from the prior
   *  poll's the cadence snaps back to `fastMs`. Defaults to a JSON snapshot of
   *  the whole response. Use this to ignore fields that always change (e.g. a
   *  per-request sequence number) so they don't defeat the back-off. */
  signature?: (value: T) => unknown;
}

/**
 * A self-scheduling poll that replaces the `timer(0, n)` + `switchMap` pattern.
 *
 * Three differences from a fixed-interval `switchMap` poll, each fixing a real
 * problem (issue #2572):
 *
 * 1. **No overlap / no cancellation.** Each `work()` runs to completion before
 *    the next is scheduled. A `switchMap` timer aborts the in-flight request on
 *    the next tick, so a backend slower than the interval had *every* request
 *    cancelled before it could finish and the panel froze permanently. Here a
 *    slow backend simply degrades to its own response time.
 * 2. **Adaptive cadence.** Polls run at `fastMs` while the response keeps
 *    changing; after `rampAfter` unchanged polls the delay eases to `slowMs`, so
 *    an idle session stops hammering the server. Any change snaps back to fast.
 * 3. **Pauses while hidden.** Polling suspends entirely while the tab is hidden
 *    (Page Visibility API) and resumes with an immediate poll when it returns.
 *
 * Callers own teardown: pipe `takeUntil(stop$)` onto the result.
 */
export function adaptivePoll<T>(
  work: () => Observable<T>,
  config: AdaptivePollConfig<T>,
): Observable<T> {
  const rampAfter = config.rampAfter ?? 3;
  const signature = config.signature ?? ((v: T): unknown => JSON.stringify(v));

  return pageVisible$().pipe(switchMap((visible) => (visible ? loop() : EMPTY)));

  function loop(): Observable<T> {
    let idle = 0;
    let lastSig: unknown = NO_SIG;
    const nextDelay = (): number => (idle >= rampAfter ? config.slowMs : config.fastMs);

    return defer(() =>
      work().pipe(catchError(() => of(POLL_SKIP as typeof POLL_SKIP))),
    ).pipe(
      tap((value) => {
        if (value === POLL_SKIP) return;
        const sig = signature(value as T);
        idle = lastSig !== NO_SIG && sameSignature(sig, lastSig) ? idle + 1 : 0;
        lastSig = sig;
      }),
      filter((value): value is T => value !== POLL_SKIP),
      repeat({ delay: () => timer(nextDelay()) }),
    );
  }
}

/** Compare two signatures. Primitive signatures (the default JSON string) match
 *  by identity; structured signatures fall back to a JSON comparison. */
function sameSignature(a: unknown, b: unknown): boolean {
  return a === b || JSON.stringify(a) === JSON.stringify(b);
}

/** Emits `true` while the document is visible and `false` while it is hidden,
 *  starting with the current state. Emits only on genuine transitions. In a
 *  non-DOM context (SSR/tests without `document`) it emits a single `true`. */
function pageVisible$(): Observable<boolean> {
  if (typeof document === 'undefined') return of(true);
  return merge(of(null), fromEvent(document, 'visibilitychange')).pipe(
    map(() => !document.hidden),
    distinctUntilChanged(),
  );
}
