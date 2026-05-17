import { Injectable, OnDestroy } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable, Subject, timer } from 'rxjs';
import { catchError, distinctUntilChanged, map, switchMap, takeUntil } from 'rxjs/operators';

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
 * whose pair has work in flight — the spinner is **per-pair**, not
 * per-half (see ``docs/plans/active-context-switcher.md`` § Phase 3).
 *
 * The service starts polling lazily on the first observer of
 * {@link busyPairs$} and stops when the last observer unsubscribes,
 * so a process with no pulldown visible (e.g. spec runs) never makes
 * the request.
 */
@Injectable({ providedIn: 'root' })
export class RunningJobsService implements OnDestroy {
  private readonly intervalMs = 3000;
  private readonly busyPairsSubject = new BehaviorSubject<Map<string, string[]>>(new Map());
  private readonly stopPolling$ = new Subject<void>();
  private readonly destroy$ = new Subject<void>();
  private observerCount = 0;
  private polling = false;

  constructor(private http: HttpClient) {}

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
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
        takeUntil(this.destroy$),
        switchMap(() =>
          this.http.get<ActiveJobsResponse>('/api/jobs/active').pipe(
            catchError(() => {
              // A transient registry-fetch failure should not tear the
              // polling pipeline down — the next tick will retry. Emit
              // an empty payload so the UI clears any stale spinners.
              return [{ busy_pairs: [] } as ActiveJobsResponse];
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
