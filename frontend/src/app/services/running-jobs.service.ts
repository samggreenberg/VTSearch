import { DestroyRef, Injectable, OnDestroy, inject } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, EMPTY, Observable, Subject, of, timer } from 'rxjs';
import { catchError, distinctUntilChanged, exhaustMap, map, takeUntil, tap, timeout } from 'rxjs/operators';

/**
 * One entry returned by ``GET /api/jobs/active``.
 *
 * Mirrors {@link vtsearch.schemas.jobs.ActiveJobPairSchema}.
 */
export interface ActiveJobPair {
  dataset_id: string;
  detector_id: string;
  /** Logical job-type names (``"learned-sort"``, ``"eval"``, …). Stable across releases. */
  job_types: string[];
}

interface ActiveJobsResponse {
  busy_pairs: ActiveJobPair[];
}

/** Encode a pair as a stable string key for set membership. */
export function pairKey(datasetId: string, detectorId: string): string {
  return `${datasetId}::${detectorId}`;
}

/**
 * Polls ``GET /api/jobs/active`` and exposes the set of
 * ``(dataset_id, detector_id)`` pairs that currently have a running or
 * pending background job.
 *
 * Consumed by the top-bar pulldown to render a spinner glyph on rows
 * whose pair has work in flight; the spinner is **per-pair**, not
 * per-half.
 *
 * The service starts polling lazily on the first observer of
 * {@link busyPairs$} and stops when the last observer unsubscribes,
 * so a process with no pulldown visible (e.g. spec runs) never makes
 * the request.
 */
@Injectable({ providedIn: 'root' })
export class RunningJobsService implements OnDestroy {
  private http = inject(HttpClient);

  private readonly intervalMs = 5000;
  // Bound a single poll so a hung backend request cannot stall the
  // pipeline forever; a timeout counts as a failed poll and the next
  // tick retries.
  private readonly requestTimeoutMs = 10000;
  // A failed poll (error or timeout) holds the last known map: a backend
  // too busy to answer in time is exactly when jobs are running, so one
  // dropped request must not blank every spinner. Only this many
  // failures in a row (about 15 s at the 5 s interval, longer when each
  // one times out) clear the map, so an unreachable backend cannot pin
  // spinners forever. Any successful poll resets the count.
  private readonly maxConsecutiveFailures = 3;
  private consecutiveFailures = 0;
  private readonly busyPairsSubject = new BehaviorSubject<Map<string, string[]>>(new Map());
  private readonly stopPolling$ = new Subject<void>();
  private readonly destroyRef = inject(DestroyRef);
  private observerCount = 0;
  private polling = false;

  ngOnDestroy(): void {
    this.stopPolling$.next();
    this.stopPolling$.complete();
  }

  /**
   * Live map from ``"${datasetId}::${detectorId}"`` → list of job-type
   * names currently running on that pair. Missing keys mean "no jobs".
   *
   * Subscribing kicks off polling; unsubscribing stops it.
   */
  readonly busyPairs$: Observable<Map<string, string[]>> = new Observable<Map<string, string[]>>(
    (observer) => {
      this.observerCount += 1;
      if (this.observerCount === 1) this.startPolling();
      const sub = this.busyPairsSubject.subscribe(observer);
      return () => {
        sub.unsubscribe();
        this.observerCount -= 1;
        if (this.observerCount === 0) this.stopPolling();
      };
    },
  );

  /**
   * Returns ``true`` while at least one job is running on ``(datasetId,
   * detectorId)``. Emits ``false`` for the empty-half case (a real pair
   * needs both ids).
   */
  isBusy(datasetId: string, detectorId: string): Observable<boolean> {
    if (!datasetId || !detectorId) {
      return new BehaviorSubject(false).asObservable();
    }
    const key = pairKey(datasetId, detectorId);
    return this.busyPairs$.pipe(
      map((m) => m.has(key)),
      distinctUntilChanged(),
    );
  }

  private startPolling(): void {
    if (this.polling) return;
    this.polling = true;
    timer(0, this.intervalMs)
      .pipe(
        takeUntil(this.stopPolling$),
        takeUntilDestroyed(this.destroyRef),
        // exhaustMap (not switchMap): while one /api/jobs/active request
        // is in flight, ignore further ticks instead of aborting and
        // re-issuing. A slow or wedged backend then sees at most one poll
        // at a time, not a growing pile of aborted requests that exhaust
        // the worker's threads. timeout() bounds a hung request.
        exhaustMap(() =>
          this.http.get<ActiveJobsResponse>('/api/jobs/active').pipe(
            timeout(this.requestTimeoutMs),
            tap(() => (this.consecutiveFailures = 0)),
            catchError(() => {
              // A failure (or the timeout above) must not tear the
              // polling pipeline down; the next tick retries. Emit
              // nothing, keeping the last known map, until enough
              // failures in a row say the backend is really gone.
              this.consecutiveFailures += 1;
              return this.consecutiveFailures >= this.maxConsecutiveFailures
                ? of<ActiveJobsResponse>({ busy_pairs: [] })
                : EMPTY;
            }),
          ),
        ),
      )
      .subscribe((res) => {
        const next = new Map<string, string[]>();
        for (const p of res.busy_pairs || []) {
          next.set(pairKey(p.dataset_id, p.detector_id), p.job_types || []);
        }
        this.busyPairsSubject.next(next);
      });
  }

  private stopPolling(): void {
    if (!this.polling) return;
    this.polling = false;
    this.stopPolling$.next();
    // Clear the cache so a later resubscriber doesn't see stale data
    // before the first poll lands.
    this.busyPairsSubject.next(new Map());
  }
}
