import { Injectable, inject } from '@angular/core';
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
  /** Set when a required load's kick-off POST failed or its background
   *  task errored.  A failed switch must NOT promote the pair to active:
   *  the backend hasn't loaded it, so promotion would re-open the H25
   *  cascade of 409 `dataset_not_loaded` this service exists to prevent. */
  failed: boolean;
  cancellable: Subject<void>;
  /** ReplaySubject(1) so late subscribers (e.g. the route guard
   *  attaching after a synchronous fast-path completion) still see the
   *  completion signal.  Completes WITHOUT emitting when the switch is
   *  superseded or fails; consumers must handle the empty completion
   *  (the guard uses `defaultIfEmpty`, browse-prep a `complete` handler). */
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
  private activeContext = inject(ActiveContextService);
  private datasetState = inject(DatasetStateService);
  private datasetsRegistryApi = inject(DatasetsRegistryApiService);
  private detectorsRegistryApi = inject(DetectorsRegistryApiService);
  private progressEvents = inject(ProgressEventsService);
  private router = inject(Router);

  private readonly switchingSubject = new BehaviorSubject<boolean>(false);
  readonly switching$ = this.switchingSubject.asObservable();

  private active: ActiveSwitch | null = null;

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
    const onBrowse = currentUrl.startsWith('/browse');
    if (onBrowse && datasetId) {
      this.router.navigate(['/', 'browse', datasetId]);
      return;
    }
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
      this.cancelInFlightLoads(previous);
    }

    const requestId = this.activeContext.nextRequestId();
    const current: ActiveSwitch = {
      requestId,
      datasetId,
      detectorId,
      cancelled: false,
      failed: false,
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

    const dataset = datasetId ? this.datasetState.datasetById().get(datasetId) ?? null : null;
    const detector = detectorId ? this.datasetState.detectorById().get(detectorId) ?? null : null;

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
            // The load kick-off failed: still tick so the switch settles,
            // but mark it failed so finishIfCurrent doesn't promote an
            // unloaded pair (which would 409 every subsequent request).
            current.failed = true;
            sub.next();
            sub.complete();
            return EMPTY;
          }),
        )
        .subscribe({
          next: (resp) => {
            this.waitForDatasetLoad(current, datasetId, resp?.task_id ?? '').subscribe({
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
            // See runDatasetLoad: settle the switch but don't promote.
            current.failed = true;
            sub.next();
            sub.complete();
            return EMPTY;
          }),
        )
        .subscribe({
          next: (resp) => {
            this.waitForDetectorLoad(current, detectorId, resp?.task_id ?? '').subscribe({
              next: () => {
                sub.next();
                sub.complete();
              },
            });
          },
        });
    });
  }

  /**
   * Resolve once the dataset load identified by *taskId* has settled.
   *
   * Keying on the load's own ``task_id`` (not just ``dataset_id``) closes
   * the SSE-lag race: ``loadRegistered`` resolves the moment the backend
   * *kicks off* the background load, which is typically before the
   * ``loading-tasks`` channel has reported the new task. A plain
   * "no non-idle task for this dataset" check reads that empty window as
   * "already finished" and promotes active early, so the next request
   * (e.g. the projection build the Browse button fires) hits a dataset
   * that isn't loaded yet and 409s. Instead we wait until the specific
   * task has been *observed* in the stream and is no longer running —
   * mirroring `DashboardLoadingTasksService`'s `awaitedTaskIds` guard.
   *
   * An empty ``taskId`` (the already-loaded / synchronous fast path that
   * starts no tracked background task) falls back to the prior id-based
   * "channel is quiet for this dataset" check, preserving that behavior.
   */
  private waitForDatasetLoad(
    current: ActiveSwitch,
    datasetId: string,
    taskId: string,
  ): Observable<void> {
    return new Observable<void>((sub) => {
      let seen = false;
      this.progressEvents.loadingTasks$
        .pipe(
          takeUntil(current.cancellable),
          filter((tasks: LoadingTask[]) => {
            if (!taskId) {
              return !tasks.some((t) => t.dataset_id === datasetId && t.status !== 'idle');
            }
            const task = tasks.find((t) => t.task_id === taskId);
            if (task) {
              seen = true;
              // A load that settled WITH an error must not promote the
              // pair; the dataset never loaded.
              if (task.status === 'idle' && task.error) {
                current.failed = true;
              }
            }
            // Settled only once we've seen the task and it's no longer
            // running (gone idle, or dropped from the stream entirely).
            return seen && !(task && task.status !== 'idle');
          }),
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

  /** Detector counterpart of {@link waitForDatasetLoad}; same SSE-lag race,
   *  same task-id guard, same id-based fallback when no task is tracked. */
  private waitForDetectorLoad(
    current: ActiveSwitch,
    detectorId: string,
    taskId: string,
  ): Observable<void> {
    return new Observable<void>((sub) => {
      let seen = false;
      this.progressEvents.detectorLoadingTasks$
        .pipe(
          takeUntil(current.cancellable),
          filter((tasks: LoadingTask[]) => {
            if (!taskId) {
              return !tasks.some((t) => t.detector_id === detectorId && t.status !== 'idle');
            }
            const task = tasks.find((t) => t.task_id === taskId);
            if (task) {
              seen = true;
              if (task.status === 'idle' && task.error) {
                current.failed = true;
              }
            }
            return seen && !(task && task.status !== 'idle');
          }),
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
    if (current.failed) {
      // A required load failed: the backend does NOT have the pair loaded,
      // so promoting it would tag every request with ids that 409. Roll
      // the intent back to the still-active pair (pulldown highlight) and
      // complete without emitting - the guard's `defaultIfEmpty` turns
      // that into a clean navigation denial, and the SSE error row /
      // global toast already surface the failure itself.
      this.activeContext.setIntent(this.activeContext.datasetId, this.activeContext.modelId);
      current.completion.complete();
      return;
    }
    // Loads have settled; promote intent to active now so the HTTP
    // interceptor starts tagging requests with the new ids (and not
    // before, per H25).
    this.activeContext.setActive(current.datasetId, current.detectorId);
    current.completion.next();
    current.completion.complete();
  }

  private cancelInFlightLoads(previous: ActiveSwitch): void {
    // Best-effort: cancel the superseded switch's own dataset/detector
    // loading tasks so we don't waste CPU on a load whose result will be
    // discarded. The request-id check is the actual correctness guarantee.
    // Scoped to the previous switch's ids: an unrelated import or a
    // dashboard-initiated load of a third dataset must not be blown away
    // by a rapid pulldown double-click.
    const datasetTasks = this.progressEvents
      .loadingTasks()
      .filter((t) => t.status !== 'idle' && !!t.dataset_id && t.dataset_id === previous.datasetId);
    for (const t of datasetTasks) {
      this.datasetsRegistryApi.cancelTask(t.task_id).subscribe({
        error: () => {
          /* tolerate races: the task may have completed already */
        },
      });
    }
    const detectorTasks = this.progressEvents
      .detectorLoadingTasks()
      .filter((t) => t.status !== 'idle' && !!t.detector_id && t.detector_id === previous.detectorId);
    for (const t of detectorTasks) {
      this.detectorsRegistryApi.cancelDetectorLoadingTask(t.task_id).subscribe({
        error: () => {
          /* tolerate races */
        },
      });
    }
  }
}
