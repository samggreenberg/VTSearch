import { Injectable } from '@angular/core';
import { BehaviorSubject, EMPTY, Observable, ReplaySubject, Subject } from 'rxjs';
import { catchError, filter, take, takeUntil } from 'rxjs/operators';
import { Router } from '@angular/router';
import { ActiveContextService } from './active-context.service';
import { DatasetStateService } from './dataset-state.service';
import { DatasetsRegistryApiService } from './datasets-registry-api.service';
import { DetectorsRegistryApiService } from './detectors-registry-api.service';
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
 * Drives a pulldown-initiated active-pair change end-to-end: tags the
 * user's intent on `ActiveContextService` immediately, kicks off any
 * dataset / detector loads that are needed, and promotes the pair to
 * *active* (what the HTTP interceptor reads) only once those loads have
 * settled. Exposes a `switching$` observable so view code (and the
 * top-bar pulldowns) can render a loading state.
 *
 * The intent/active split (see `ActiveContextService`) is what fixes
 * H25: without it, the interceptor would tag outgoing requests with
 * the new ids the moment the pulldown was clicked, causing a cascade
 * of 409 `dataset_not_loaded` until the load finished.
 *
 * Cancel-and-replace semantics: rapidly clicking a different row tags
 * each invocation with a monotonic request id. When a prep step
 * completes, the captured id is compared with the current id and the
 * result is discarded if they differ. The previous load's HTTP request
 * is also cancelled when possible.
 *
 * Two entry points:
 *
 *  - `switchTo(ds, det)`: called by the top-bar pulldowns. When the
 *    user is on `/label/:ds/:det` or `/find/:ds/:det`, it navigates to
 *    the new URL (the route guard then calls `applyActivePair`). On
 *    other routes (e.g. `/dashboard`) it flips the pair imperatively.
 *  - `applyActivePair(ds, det)`: called by the active-context route
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
    private datasetsRegistryApi: DatasetsRegistryApiService,
    private detectorsRegistryApi: DetectorsRegistryApiService,
    private progressEvents: ProgressEventsService,
    private router: Router,
  ) {}

  get switching(): boolean {
    return this.switchingSubject.value;
  }

  /**
   * Pulldown-initiated switch. On `/label` or `/find`, navigates to the
   * matching `/<view>/:ds/:det` URL; the route guard then drives the
   * active-pair flip via `applyActivePair`. On other routes, flips the
   * pair imperatively (no navigation).
   *
   * Passing `''` for either half clears that half; no load is fired
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
   * Called by `activeContextGuard`. Tags the user's intent, kicks off
   * any dataset / detector loads, and returns an Observable that
   * completes when all loads finish (or immediately if nothing was
   * needed). At completion the pair is promoted to *active* so the HTTP
   * interceptor starts tagging requests with the new ids. The guard
   * holds the route activation until this completes so the view never
   * renders against a half-loaded backend.
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
      // Same pair, same in-flight switch; share the completion signal.
      return previous.completion.asObservable();
    }

    // Cancel any in-flight prep (best effort). The request-id check on
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

    // Tag the user's intent so UI affordances (pulldown highlight) update
    // immediately. The active pair (what the HTTP interceptor reads)
    // stays pinned to the currently-loaded backend state and is promoted
    // in `finishIfCurrent` once any required load completes. This avoids
    // the H25 race where the interceptor would otherwise tag requests
    // with an id the backend hasn't loaded yet.
    this.activeContext.setIntent(datasetId, detectorId);

    const datasets = this.datasetState.datasets;
    const detectors = this.datasetState.detectors;
    const dataset = datasetId ? datasets.find((d) => d.id === datasetId) : null;
    const detector = detectorId ? detectors.find((d) => d.id === detectorId) : null;

    const needsDatasetLoad = !!dataset && !dataset.loaded;
    // Re-embed labels when the active dataset's embedder differs from the
    // one the detector's cached label vectors were built with. Both sides
    // are reported by the registry endpoints; missing values (unloaded
    // halves, legacy data) skip this trigger and fall through to the
    // normal load / no-op path.
    const needsLabelReembed =
      !!dataset &&
      !!detector &&
      !!detector.detector_loaded &&
      !!dataset.embedder &&
      !!detector.embedder &&
      dataset.embedder !== detector.embedder;
    const needsDetectorLoad = !!detector && (!detector.detector_loaded || needsLabelReembed);

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
      this.datasetsRegistryApi
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
      this.detectorsRegistryApi
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
    // Loads have settled; promote intent to active now so the HTTP
    // interceptor starts tagging requests with the new ids (and not
    // before, per H25).
    this.activeContext.setActive(current.datasetId, current.detectorId);
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
      this.datasetsRegistryApi.cancelTask(t.task_id).subscribe({
        error: () => {
          /* tolerate races: the task may have completed already */
        },
      });
    }
    const detectorTasks = this.progressEvents.detectorLoadingTasks.filter((t) => t.status !== 'idle');
    for (const t of detectorTasks) {
      this.detectorsRegistryApi.cancelDetectorLoadingTask(t.task_id).subscribe({
        error: () => {
          /* tolerate races */
        },
      });
    }
  }
}
