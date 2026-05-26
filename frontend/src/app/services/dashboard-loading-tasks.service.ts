import { Injectable } from '@angular/core';
import { BehaviorSubject, Subject } from 'rxjs';
import { filter, takeUntil } from 'rxjs/operators';
import { LoadingTask } from '../models/api.models';
import { AchievementsService } from './achievements.service';
import { DatasetStateService } from './dataset-state.service';
import { DatasetsRegistryApiService } from './datasets-registry-api.service';
import { DetectorsRegistryApiService } from './detectors-registry-api.service';
import { ProgressEventsService } from './progress-events.service';

/**
 * Owns the Dashboard's per-task loading-row state and the bookkeeping
 * that drives the "poll until everything settles, then refresh"
 * behaviour. Lifted out of `DashboardComponent` so the dashboard is a
 * thinner layout/wiring shell - it reads the published lists, listens
 * for completion side effects, and forwards user clicks (cancel /
 * dismiss) here.
 *
 * Bookkeeping responsibilities, all of which used to live inline on the
 * Dashboard component:
 *  - Demux the raw SSE stream into "currently active" + "errored but
 *    not yet dismissed" rows, so the dashboard table can keep failed
 *    rows visible with the dashed loading bar + Dismiss button.
 *  - Track `awaitedTaskIds` for tasks whose HTTP response landed
 *    before the SSE stream caught up, so we don't bail out of polling
 *    before they're observed.
 *  - Track `completedTaskIds` / `completedModelTaskIds` so we can fire
 *    `datasetState.refresh()` + `achievements.refresh()` the moment a
 *    task finishes (not only when the whole batch is idle).
 *  - Drive `datasetState.setLoading()` based on active dataset tasks.
 */
@Injectable({ providedIn: 'root' })
export class DashboardLoadingTasksService {
  private readonly loadingTasksSubject = new BehaviorSubject<LoadingTask[]>([]);
  private readonly detectorLoadingTasksSubject = new BehaviorSubject<LoadingTask[]>([]);

  readonly loadingTasks$ = this.loadingTasksSubject.asObservable();
  readonly detectorLoadingTasks$ = this.detectorLoadingTasksSubject.asObservable();

  private polling$ = new Subject<void>();
  private detectorPolling$ = new Subject<void>();

  private awaitedTaskIds = new Set<string>();
  private completedTaskIds = new Set<string>();
  private completedModelTaskIds = new Set<string>();
  private datasetPollingActive = false;
  private detectorPollingActive = false;

  constructor(
    private progressEvents: ProgressEventsService,
    private datasetsRegistryApi: DatasetsRegistryApiService,
    private detectorsRegistryApi: DetectorsRegistryApiService,
    private datasetState: DatasetStateService,
    private achievements: AchievementsService,
  ) {
    // SSE pushes the initial snapshot on connect, so the first event on
    // each channel tells us whether there's anything in flight - auto-
    // resume polling without the dashboard having to coordinate it.
    this.progressEvents.loadingTasks$
      .pipe(filter((tasks) => tasks.some((t) => t.status !== 'idle')))
      .subscribe(() => this.startProgressPolling());
    this.progressEvents.detectorLoadingTasks$
      .pipe(filter((tasks) => tasks.some((t) => t.status !== 'idle')))
      .subscribe(() => this.startDetectorProgressPolling());
  }

  get loadingTasks(): LoadingTask[] {
    return this.loadingTasksSubject.value;
  }

  get detectorLoadingTasks(): LoadingTask[] {
    return this.detectorLoadingTasksSubject.value;
  }

  /** Map dataset_id → LoadingTask for tasks that match an existing dataset row. */
  get inlineTaskMap(): Map<string, LoadingTask> {
    const map = new Map<string, LoadingTask>();
    const datasetIds = new Set(this.datasetState.datasets.map((d) => d.id));
    for (const task of this.loadingTasks) {
      if (task.dataset_id && datasetIds.has(task.dataset_id)) {
        map.set(task.dataset_id, task);
      }
    }
    return map;
  }

  /** Loading tasks that have no matching dataset row (new imports, etc.). */
  get orphanLoadingTasks(): LoadingTask[] {
    const datasetIds = new Set(this.datasetState.datasets.map((d) => d.id));
    return this.loadingTasks.filter((t) => !t.dataset_id || !datasetIds.has(t.dataset_id));
  }

  getInlineTask(datasetId: string): LoadingTask | undefined {
    return this.inlineTaskMap.get(datasetId);
  }

  getInlineDetectorTask(modelId: string): LoadingTask | undefined {
    return this.detectorLoadingTasks.find((t) => t.detector_id === modelId);
  }

  startProgressPolling(awaitTaskId?: string, onComplete?: () => void): void {
    // Register any task we've been told to expect.  See the field
    // comment on `awaitedTaskIds` for why this is needed.
    if (awaitTaskId) {
      this.awaitedTaskIds.add(awaitTaskId);
    }
    // If polling is already active, don't restart - the existing loop
    // already covers all tasks.  This avoids clearing completedTaskIds
    // and losing track of tasks that just finished.
    if (this.datasetPollingActive) {
      return;
    }
    this.datasetPollingActive = true;
    this.completedTaskIds.clear();

    this.progressEvents.loadingTasks$
      .pipe(takeUntil(this.polling$))
      .subscribe({
        next: (tasks: LoadingTask[]) => {
          // Any task we were waiting for has now shown up in the SSE
          // stream - drop it from the awaited set so the bail-out check
          // below can fire as soon as the stream goes quiet.
          for (const t of tasks) {
            this.awaitedTaskIds.delete(t.task_id);
          }

          // Separate active from finished. Failed tasks are surfaced
          // globally by SseErrorRouterService → ToastService; we just
          // keep them in the inline list so the row still shows the
          // dashed loading bar with the error text.
          const active = tasks.filter((t) => t.status !== 'idle');
          const errored = tasks.filter((t) => t.status === 'idle' && !!t.error);
          const failed = errored.filter((t) => t.error !== 'Cancelled');

          this.loadingTasksSubject.next([...active, ...failed]);

          // Detect tasks that just completed successfully so we can
          // refresh the registry immediately (not only when ALL finish).
          const justFinished = tasks.filter(
            (t) => t.status === 'idle' && !t.error && !this.completedTaskIds.has(t.task_id),
          );
          for (const t of justFinished) {
            this.completedTaskIds.add(t.task_id);
          }
          if (justFinished.length > 0) {
            this.datasetState.refresh();
            this.achievements.refresh();
          }

          this.datasetState.setLoading(active.length > 0);

          if (active.length === 0 && this.awaitedTaskIds.size === 0) {
            // No more active tasks and no awaited task pending - stop
            // polling.
            this.polling$.next();
            this.datasetPollingActive = false;
            // Refresh unless we just did (justFinished already triggered it)
            if (justFinished.length === 0) {
              this.datasetState.refresh();
            }
            if (onComplete && failed.length === 0) {
              onComplete();
            }
          }
        },
      });
  }

  startDetectorProgressPolling(onComplete?: () => void): void {
    if (this.detectorPollingActive) {
      return;
    }
    this.detectorPollingActive = true;
    this.completedModelTaskIds.clear();

    this.progressEvents.detectorLoadingTasks$
      .pipe(takeUntil(this.detectorPolling$))
      .subscribe({
        next: (tasks: LoadingTask[]) => {
          const active = tasks.filter((t) => t.status !== 'idle');
          const errored = tasks.filter((t) => t.status === 'idle' && !!t.error);
          const failed = errored.filter((t) => t.error !== 'Cancelled');

          this.detectorLoadingTasksSubject.next([...active, ...failed]);

          // Detect tasks that just completed successfully
          const justFinished = tasks.filter(
            (t) => t.status === 'idle' && !t.error && !this.completedModelTaskIds.has(t.task_id),
          );
          for (const t of justFinished) {
            this.completedModelTaskIds.add(t.task_id);
          }
          if (justFinished.length > 0) {
            this.datasetState.refresh();
            this.achievements.refresh();
          }

          if (active.length === 0) {
            this.detectorPolling$.next();
            this.detectorPollingActive = false;
            if (justFinished.length === 0) {
              this.datasetState.refresh();
            }
            if (onComplete && failed.length === 0) {
              onComplete();
            }
          }
        },
      });
  }

  cancelLoadingTask(taskId: string): void {
    this.datasetsRegistryApi.cancelTask(taskId).subscribe();
  }

  dismissLoadingTask(taskId: string): void {
    this.loadingTasksSubject.next(this.loadingTasks.filter((t) => t.task_id !== taskId));
  }

  cancelDetectorLoadingTask(taskId: string): void {
    this.detectorsRegistryApi.cancelDetectorLoadingTask(taskId).subscribe();
  }

  dismissDetectorLoadingTask(taskId: string): void {
    this.detectorLoadingTasksSubject.next(
      this.detectorLoadingTasks.filter((t) => t.task_id !== taskId),
    );
  }
}
