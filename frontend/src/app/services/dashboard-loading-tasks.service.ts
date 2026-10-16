import { HttpContext } from '@angular/common/http';
import { Injectable, inject, signal } from '@angular/core';
import { Subject } from 'rxjs';
import { filter, takeUntil } from 'rxjs/operators';
import { SKIP_ERROR_TOAST } from '../interceptors/error.interceptor';
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
 * thinner layout/wiring shell; it reads the published lists, listens
 * for completion side effects, and forwards user clicks (cancel /
 * dismiss) here.
 *
 * Bookkeeping responsibilities, all of which used to live inline on the
 * Dashboard component:
 *  - Demux the raw SSE stream into "currently active" + "errored but
 *    not yet dismissed" rows, so the dashboard table can keep failed
 *    rows visible with the error text + Dismiss button. The backend keeps a
 *    failed dataset task on the stream until it is dismissed (#4973), so the
 *    failed rows are read off every snapshot, not only while polling: a page
 *    reload, or another load starting, must not lose them.
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
  private progressEvents = inject(ProgressEventsService);
  private datasetsRegistryApi = inject(DatasetsRegistryApiService);
  private detectorsRegistryApi = inject(DetectorsRegistryApiService);
  private datasetState = inject(DatasetStateService);
  private achievements = inject(AchievementsService);

  // Signals (not BehaviorSubjects) so the dashboard's template reads of
  // `loadingTasks`/`orphanLoadingTasks`/`inlineTaskMap` repaint under zoneless
  // OnPush change detection: a `.set()` from the SSE-driven poller schedules CD
  // because the value is read (through getters) during template evaluation.
  // Active dataset rows, written by the polling loop. Failed rows live apart
  // (`_failedTasks`) because they outlast it: see the class doc.
  private readonly _activeTasks = signal<LoadingTask[]>([]);
  private readonly _failedTasks = signal<LoadingTask[]>([]);
  private readonly _detectorLoadingTasks = signal<LoadingTask[]>([]);

  // Failed tasks the user dismissed, hidden until the backend's snapshot drops
  // them too (the dismiss request is in flight until then).
  private readonly dismissedTaskIds = new Set<string>();

  // Task ids the user has clicked Cancel on but whose backend hasn't finished
  // unwinding yet. Drives the per-row "Cancelling…" acknowledgement badge so
  // the row doesn't freeze in its pre-cancel state and invite repeated clicks.
  // Pruned to the active set on every SSE snapshot (a task that has left the
  // active list is done cancelling), so the flag clears itself.
  private readonly _cancellingTaskIds = signal<ReadonlySet<string>>(new Set());

  private polling$ = new Subject<void>();
  private detectorPolling$ = new Subject<void>();

  private awaitedTaskIds = new Set<string>();
  private completedTaskIds = new Set<string>();
  private completedModelTaskIds = new Set<string>();
  private datasetPollingActive = false;
  private detectorPollingActive = false;

  /**
   * Completion callbacks registered by `startProgressPolling` /
   * `startDetectorProgressPolling` callers, fired (and cleared) when the
   * polling loop settles.  Kept as lists on the service (not closure
   * parameters of the loop) so a caller whose `startProgressPolling` call
   * early-returns because polling is already active still gets its
   * callback invoked - dropping it silently left `setActivePair` never
   * firing when a dataset was loaded while another import was running.
   * Each dataset callback carries the task id it belongs to (when known)
   * so an *unrelated* task's failure doesn't suppress it.
   *
   * Callbacks receive the settling SSE snapshot so a caller can read the
   * completed task's association fields (e.g. its ``dataset_id``) without a
   * second poll — used by the combine-datasets summary toast.
   */
  private datasetPollCallbacks: Array<{ taskId: string; cb: (completedTasks: LoadingTask[]) => void }> = [];
  private detectorPollCallbacks: Array<() => void> = [];

  constructor() {
    this.progressEvents.loadingTasks$.subscribe((tasks) => this.syncFailedTasks(tasks));
    // SSE pushes the initial snapshot on connect, so the first event on
    // each channel tells us whether there's anything in flight; auto-
    // resume polling without the dashboard having to coordinate it.
    this.progressEvents.loadingTasks$
      .pipe(filter((tasks) => tasks.some((t) => t.status !== 'idle')))
      .subscribe(() => this.startProgressPolling());
    this.progressEvents.detectorLoadingTasks$
      .pipe(filter((tasks) => tasks.some((t) => t.status !== 'idle')))
      .subscribe(() => this.startDetectorProgressPolling());
    // Backend restart: every `task_id` we're tracking refers to a task
    // that no longer exists. Tear down the existing polling loops (so
    // their stale `awaitedTaskIds` don't keep them latched on forever),
    // drop the per-task sets, and clear inline error rows that
    // reference vanished tasks. The next non-idle SSE snapshot will
    // re-engage polling cleanly via the constructor subscriptions above.
    this.progressEvents.serverReset$.subscribe(() => this.resetOnBackendRestart());
  }

  private resetOnBackendRestart(): void {
    this.polling$.next();
    this.detectorPolling$.next();
    this.awaitedTaskIds.clear();
    this.completedTaskIds.clear();
    this.completedModelTaskIds.clear();
    this.datasetPollCallbacks = [];
    this.detectorPollCallbacks = [];
    this.datasetPollingActive = false;
    this.detectorPollingActive = false;
    this._activeTasks.set([]);
    this._failedTasks.set([]);
    this.dismissedTaskIds.clear();
    this._detectorLoadingTasks.set([]);
    this._cancellingTaskIds.set(new Set());
    this.datasetState.setLoading(false);
  }

  /** True while the user has cancelled `taskId` but the backend is still
   *  unwinding it (the task is still in the active list). */
  isCancelling(taskId: string): boolean {
    return this._cancellingTaskIds().has(taskId);
  }

  private markCancelling(taskId: string): void {
    if (this._cancellingTaskIds().has(taskId)) return;
    const next = new Set(this._cancellingTaskIds());
    next.add(taskId);
    this._cancellingTaskIds.set(next);
  }

  private unmarkCancelling(taskId: string): void {
    const current = this._cancellingTaskIds();
    if (!current.has(taskId)) return;
    const next = new Set(current);
    next.delete(taskId);
    this._cancellingTaskIds.set(next);
  }

  /** Drop any cancelling flag whose task is no longer active — the backend has
   *  finished unwinding it, so the row is about to leave the table. */
  private pruneCancelling(activeTaskIds: Set<string>): void {
    const current = this._cancellingTaskIds();
    if (current.size === 0) return;
    const next = new Set<string>();
    for (const id of current) {
      if (activeTaskIds.has(id)) next.add(id);
    }
    if (next.size !== current.size) this._cancellingTaskIds.set(next);
  }

  /** Active dataset tasks, then failed ones not yet dismissed. */
  get loadingTasks(): LoadingTask[] {
    const failed = this._failedTasks();
    if (failed.length === 0) return this._activeTasks();
    // A task the loop last saw running may already be in the failed list.
    const failedIds = new Set(failed.map((t) => t.task_id));
    return [...this._activeTasks().filter((t) => !failedIds.has(t.task_id)), ...failed];
  }

  /** Keep the failed rows in step with a snapshot, minus the dismissed ones. */
  private syncFailedTasks(tasks: LoadingTask[]): void {
    const present = new Set(tasks.map((t) => t.task_id));
    for (const id of this.dismissedTaskIds) {
      if (!present.has(id)) this.dismissedTaskIds.delete(id);
    }
    this._failedTasks.set(
      tasks.filter(
        (t) => t.status === 'idle' && !!t.error && t.error !== 'Cancelled' && !this.dismissedTaskIds.has(t.task_id),
      ),
    );
  }

  get detectorLoadingTasks(): LoadingTask[] {
    return this._detectorLoadingTasks();
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

  startProgressPolling(awaitTaskId?: string, onComplete?: (completedTasks: LoadingTask[]) => void): void {
    // Register any task we've been told to expect.  See the field
    // comment on `awaitedTaskIds` for why this is needed.
    if (awaitTaskId) {
      this.awaitedTaskIds.add(awaitTaskId);
    }
    // Registered on the service (not captured by the loop) so it fires
    // even when the early-return below is taken - see the field comment.
    if (onComplete) {
      this.datasetPollCallbacks.push({ taskId: awaitTaskId ?? '', cb: onComplete });
    }
    // If polling is already active, don't restart; the existing loop
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
          // stream; drop it from the awaited set so the bail-out check
          // below can fire as soon as the stream goes quiet.
          for (const t of tasks) {
            this.awaitedTaskIds.delete(t.task_id);
          }

          // Separate active from finished. Failed rows are tracked off
          // every snapshot (`syncFailedTasks`); here they only decide which
          // completion callbacks to suppress.
          const active = tasks.filter((t) => t.status !== 'idle');
          const errored = tasks.filter((t) => t.status === 'idle' && !!t.error);
          const failed = errored.filter((t) => t.error !== 'Cancelled');

          this._activeTasks.set(active);
          this.pruneCancelling(new Set(active.map((t) => t.task_id)));

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
            // No more active tasks and no awaited task pending; stop
            // polling.
            this.polling$.next();
            this.datasetPollingActive = false;
            // Refresh unless we just did (justFinished already triggered it)
            if (justFinished.length === 0) {
              this.datasetState.refresh();
            }
            // Fire every registered completion callback whose own task
            // didn't fail.  Callbacks without a task id (legacy callers)
            // keep the old "any failure suppresses" gate.
            const callbacks = this.datasetPollCallbacks;
            this.datasetPollCallbacks = [];
            const failedIds = new Set(failed.map((t) => t.task_id));
            for (const entry of callbacks) {
              const suppressed = entry.taskId ? failedIds.has(entry.taskId) : failed.length > 0;
              if (!suppressed) {
                entry.cb(tasks);
              }
            }
          }
        },
      });
  }

  startDetectorProgressPolling(onComplete?: () => void): void {
    // Registered on the service so it fires even when polling is already
    // active - see the field comment on `detectorPollCallbacks`.
    if (onComplete) {
      this.detectorPollCallbacks.push(onComplete);
    }
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

          this._detectorLoadingTasks.set([...active, ...failed]);
          this.pruneCancelling(new Set(active.map((t) => t.task_id)));

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
            const callbacks = this.detectorPollCallbacks;
            this.detectorPollCallbacks = [];
            if (failed.length === 0) {
              for (const cb of callbacks) {
                cb();
              }
            }
          }
        },
      });
  }

  cancelLoadingTask(taskId: string): void {
    this.markCancelling(taskId);
    // A cancel that reached nothing answers 409 (the task had already
    // finished, or its progress was stale with no worker behind it — see
    // POST /api/dataset/cancel). The row is not cancelling in that case, so
    // drop the flag rather than leaving it stuck on "Cancelling…"; the backend
    // clears the stale progress itself, so the row leaves on the next SSE
    // snapshot.
    this.datasetsRegistryApi.cancelTask(taskId).subscribe({
      error: () => this.unmarkCancelling(taskId),
    });
  }

  /**
   * Dismiss a failed dataset task. The row goes at once; the backend, which
   * would otherwise keep re-sending it, is told to drop it. A refusal (the
   * task is already gone, say after a restart) is not worth a toast: the row
   * is hidden either way, which is all the user asked for.
   */
  dismissLoadingTask(taskId: string): void {
    this.dismissedTaskIds.add(taskId);
    this._failedTasks.set(this._failedTasks().filter((t) => t.task_id !== taskId));
    const context = new HttpContext().set(SKIP_ERROR_TOAST, true);
    this.datasetsRegistryApi.dismissTask(taskId, context).subscribe({ error: () => undefined });
  }

  cancelDetectorLoadingTask(taskId: string): void {
    this.markCancelling(taskId);
    this.detectorsRegistryApi.cancelDetectorLoadingTask(taskId).subscribe();
  }

  dismissDetectorLoadingTask(taskId: string): void {
    this._detectorLoadingTasks.set(
      this.detectorLoadingTasks.filter((t) => t.task_id !== taskId),
    );
  }
}
