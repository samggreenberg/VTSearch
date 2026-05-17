import { Injectable } from '@angular/core';
import { BehaviorSubject, EMPTY, Observable, ReplaySubject, Subject } from 'rxjs';
import { catchError, filter, take, takeUntil } from 'rxjs/operators';
import { Router } from '@angular/router';
import { ActiveContextService } from './active-context.service';
import { DatasetStateService } from './dataset-state.service';
import { DatasetsApiService } from './datasets-api.service';
import { DetectorsApiService } from './detectors-api.service';
import { ProgressEventsService } from './progress-events.service';
import { LoadingTask } from '../models/api.models';

interface ActiveSwitch {
  requestId: number;
  datasetId: string;
  detectorId: string;
  cancelled: boolean;
  cancellable: Subject<void>;
  /** ReplaySubject(1) so late subscribers (e.g. the route guard
   *  attaching after a synchronous fast-path completion) still see the
   *  completion signal. */
  completion: ReplaySubject<void>;
}

/**
 * Drives a pulldown-initiated active-pair change end-to-end: sets the
 * pair atomically on `ActiveContextService`, kicks off any dataset /
 * detector loads that are needed, and exposes a `switching$` observable
 * so view code (and the top-bar pulldowns) can render a loading state.
 *
 * Cancel-and-replace semantics: rapidly clicking a different row tags
 * each invocation with a monotonic request id. When a prep step
 * completes, the captured id is compared with the current id and the
 * result is discarded if they differ. The previous load's HTTP request
 * is also cancelled when possible. See
 * `docs/plans/active-context-switcher.md` § "Cancel-and-replace on
 * rapid re-click".
 *
 * Phase 2 added two entry points:
 *
 *  - `switchTo(ds, det)` — called by the top-bar pulldowns. When the
 *    user is on `/label/:ds/:det` or `/find/:ds/:det`, it navigates to
 *    the new URL (the route guard then calls `applyActivePair`). On
 *    other routes (e.g. `/dashboard`) it flips the pair imperatively.
 *  - `applyActivePair(ds, det)` — called by the active-context route
 *    guard. Flips the pair imperatively (no navigation) and returns an
 *    Observable that completes when any required loads complete, so
 *    the guard can hold the route until prep is done.
 */
@Injectable({ providedIn: 'root' })
export class ContextSwitchService {
  private readonly switchingSubject = new BehaviorSubject<boolean>(false);
  readonly switching$ = this.switchingSubject.asObservable();

  private active: ActiveSwitch | null = null;

  constructor(
    private activeContext: ActiveContextService,
    private datasetState: DatasetStateService,
    private datasetsApi: DatasetsApiService,
    private detectorsApi: DetectorsApiService,
    private progressEvents: ProgressEventsService,
    private router: Router,
  ) {}

  get switching(): boolean {
    return this.switchingSubject.value;
  }

  /**
   * Pulldown-initiated switch. On `/label` or `/find`, navigates to the
   * matching `/<view>/:ds/:det` URL — the route guard then drives the
   * active-pair flip via `applyActivePair`. On other routes, flips the
   * pair imperatively (no navigation).
   *
   * Passing `''` for either half clears that half — no load is fired
   * for an empty id, and we never navigate to a partial URL.
   */
  switchTo(datasetId: string, detectorId: string): void {
    const currentUrl = this.router.url.split('?')[0];
    const onLabel = currentUrl.startsWith('/label');
    const onFind = currentUrl.startsWith('/find');
    if ((onLabel || onFind) && datasetId && detectorId) {
      const seg = onFind ? 'find' : 'label';
      this.router.navigate(['/', seg, datasetId, detectorId]);
      return;
    }
    this.flipAndLoad(datasetId, detectorId).subscribe();
  }

  /**
   * Called by `activeContextGuard`. Atomically flips the active pair
   * and kicks off any dataset / detector loads, returning an Observable
   * that completes when all loads finish (or immediately if nothing was
   * needed). The guard holds the route activation until this completes
   * so the view doesn't render against half-loaded state.
   */
  applyActivePair(datasetId: string, detectorId: string): Observable<void> {
    return this.flipAndLoad(datasetId, detectorId);
  }

  private flipAndLoad(datasetId: string, detectorId: string): Observable<void> {
    const previous = this.active;
    if (
      previous &&
      previous.datasetId === datasetId &&
      previous.detectorId === detectorId &&
      !previous.cancelled
    ) {
      // Same pair, same in-flight switch — share the completion signal.
      return previous.completion.asObservable();
    }

    // Cancel any in-flight prep — best effort. The request-id check on
    // completion is what actually guards against stale results.
    if (previous && !previous.cancelled) {
      previous.cancelled = true;
      previous.cancellable.next();
      previous.cancellable.complete();
      previous.completion.complete();
      this.cancelInFlightLoads();
    }

    const requestId = this.activeContext.nextRequestId();
    const current: ActiveSwitch = {
      requestId,
      datasetId,
      detectorId,
      cancelled: false,
      cancellable: new Subject<void>(),
      completion: new ReplaySubject<void>(1),
    };
    this.active = current;

    // Flip the pair atomically so the HTTP interceptor immediately
    // routes against the new ids.
    this.activeContext.setActivePair(datasetId, detectorId);

    const datasets = this.datasetState.datasets;
    const detectors = this.datasetState.detectors;
    const dataset = datasetId ? datasets.find((d) => d.id === datasetId) : null;
    const detector = detectorId ? detectors.find((d) => d.id === detectorId) : null;

    const needsDatasetLoad = !!dataset && !dataset.loaded;
    const needsDetectorLoad = !!detector && !detector.detector_loaded;

    if (!needsDatasetLoad && !needsDetectorLoad) {
      this.finishIfCurrent(current);
      return current.completion.asObservable();
    }

    this.switchingSubject.next(true);
    this.datasetState.setLoading(true);

    let pending = 0;
    const tick = (): void => {
      pending -= 1;
      if (pending <= 0) this.finishIfCurrent(current);
    };

    if (needsDatasetLoad && dataset) {
      pending += 1;
      this.runDatasetLoad(current, dataset.id).subscribe({
        next: tick,
      });
    }
    if (needsDetectorLoad && detector) {
      pending += 1;
      this.runDetectorLoad(current, detector.id).subscribe({
        next: tick,
      });
    }

    return current.completion.asObservable();
  }

  private runDatasetLoad(current: ActiveSwitch, datasetId: string): Observable<void> {
    return new Observable<void>((sub) => {
      this.datasetsApi
        .loadRegistered(datasetId)
        .pipe(
          takeUntil(current.cancellable),
          catchError(() => {
            sub.next();
            sub.complete();
            return EMPTY;
          }),
        )
        .subscribe({
          next: () => {
            this.waitForDatasetLoad(current, datasetId).subscribe({
              next: () => {
                sub.next();
                sub.complete();
              },
            });
          },
        });
    });
  }

  private runDetectorLoad(current: ActiveSwitch, detectorId: string): Observable<void> {
    return new Observable<void>((sub) => {
      this.detectorsApi
        .loadDetector(detectorId)
        .pipe(
          takeUntil(current.cancellable),
          catchError(() => {
            sub.next();
            sub.complete();
            return EMPTY;
          }),
        )
        .subscribe({
          next: () => {
            this.waitForDetectorLoad(current, detectorId).subscribe({
              next: () => {
                sub.next();
                sub.complete();
              },
            });
          },
        });
    });
  }

  private waitForDatasetLoad(current: ActiveSwitch, datasetId: string): Observable<void> {
    return new Observable<void>((sub) => {
      this.progressEvents.loadingTasks$
        .pipe(
          takeUntil(current.cancellable),
          filter(
            (tasks: LoadingTask[]) =>
              !tasks.some((t) => t.dataset_id === datasetId && t.status !== 'idle'),
          ),
          take(1),
        )
        .subscribe({
          next: () => {
            this.datasetState.refresh();
            sub.next();
            sub.complete();
          },
        });
    });
  }

  private waitForDetectorLoad(current: ActiveSwitch, detectorId: string): Observable<void> {
    return new Observable<void>((sub) => {
      this.progressEvents.detectorLoadingTasks$
        .pipe(
          takeUntil(current.cancellable),
          filter(
            (tasks: LoadingTask[]) =>
              !tasks.some((t) => t.detector_id === detectorId && t.status !== 'idle'),
          ),
          take(1),
        )
        .subscribe({
          next: () => {
            this.datasetState.refresh();
            sub.next();
            sub.complete();
          },
        });
    });
  }

  private finishIfCurrent(current: ActiveSwitch): void {
    if (this.active !== current || current.cancelled) return;
    if (current.requestId !== this.activeContext.currentRequestId) return;
    this.switchingSubject.next(false);
    this.datasetState.setLoading(false);
    this.active = null;
    current.completion.next();
    current.completion.complete();
  }

  private cancelInFlightLoads(): void {
    // Best-effort: cancel any active dataset/detector loading tasks so
    // we don't waste CPU on a load whose result will be discarded. The
    // request-id check is the actual correctness guarantee.
    const datasetTasks = this.progressEvents.loadingTasks.filter((t) => t.status !== 'idle');
    for (const t of datasetTasks) {
      this.datasetsApi.cancelTask(t.task_id).subscribe({
        error: () => {
          /* tolerate races: the task may have completed already */
        },
      });
    }
    const detectorTasks = this.progressEvents.detectorLoadingTasks.filter((t) => t.status !== 'idle');
    for (const t of detectorTasks) {
      this.detectorsApi.cancelDetectorLoadingTask(t.task_id).subscribe({
        error: () => {
          /* tolerate races */
        },
      });
    }
  }
}
