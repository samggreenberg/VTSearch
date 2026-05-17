import { Injectable } from '@angular/core';
import { BehaviorSubject, EMPTY, Observable, Subject } from 'rxjs';
import { catchError, filter, take, takeUntil } from 'rxjs/operators';
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
  ) {}

  get switching(): boolean {
    return this.switchingSubject.value;
  }

  /**
   * Switch the active pair, kicking off any needed dataset / detector
   * loads. Returns nothing — callers observe completion via
   * `activeContext.pair$` (set as soon as the pair flips) plus the
   * `datasetState.loading$` signal.
   *
   * Passing `''` for either half clears that half — no load is fired
   * for an empty id.
   */
  switchTo(datasetId: string, detectorId: string): void {
    const previous = this.active;
    if (previous && previous.datasetId === datasetId && previous.detectorId === detectorId) {
      return;
    }

    // Cancel any in-flight prep — best effort. The request-id check on
    // completion is what actually guards against stale results.
    if (previous && !previous.cancelled) {
      previous.cancelled = true;
      previous.cancellable.next();
      previous.cancellable.complete();
      this.cancelInFlightLoads();
    }

    const requestId = this.activeContext.nextRequestId();
    const current: ActiveSwitch = {
      requestId,
      datasetId,
      detectorId,
      cancelled: false,
      cancellable: new Subject<void>(),
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
      return;
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
