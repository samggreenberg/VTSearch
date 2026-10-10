import { DestroyRef, Injectable, computed, effect, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

import { DashboardLoadingTasksService } from './dashboard-loading-tasks.service';
import { DatasetsRegistryApiService } from './datasets-registry-api.service';

/**
 * Load the Image dataset a face's photo lives in, then try again.
 *
 * Show in photo (#4750) and Collapse to photos (#4751) both read photos out of
 * the Image dataset the faces' import produced, and both get a 409
 * `source_not_loaded` naming it when it is registered but not loaded. This is
 * the "Load it" half they share: start the registry load, follow its loading
 * task, and call back once it is in memory.
 *
 * Component-scoped (list it in the component's `providers`): the load's
 * completion is registered on a root service and can land after the component
 * is gone, and destroying the component forgets the task so the callback
 * never fires into it.
 */
@Injectable()
export class SiblingLoadService {
  private readonly datasetsRegistryApi = inject(DatasetsRegistryApiService);
  private readonly loadingTasks = inject(DashboardLoadingTasksService);
  private readonly destroyRef = inject(DestroyRef);

  /** The task id of the load this component started, while it runs ('' before the task is known). */
  private readonly taskId = signal<string | null>(null);
  private onLoaded: (() => void) | null = null;

  /** A load is running. */
  readonly loading = computed(() => this.taskId() !== null);
  /** Why the last load failed, once it has. */
  readonly error = signal('');

  constructor() {
    this.destroyRef.onDestroy(() => this.taskId.set(null));
    // A load that settles with an error never calls back (the loading-task
    // service fires completions only on success), so watch its row for the
    // error instead; the failure is also toasted app-wide.
    effect(() => {
      const taskId = this.taskId();
      if (!taskId) return;
      const failed = this.loadingTasks.loadingTasks.find((t) => t.task_id === taskId && t.status === 'idle' && t.error);
      if (!failed) return;
      this.taskId.set(null);
      this.error.set(failed.error ?? 'Loading failed.');
    });
  }

  /** Load *datasetId*, then call *onLoaded*. A second call while one runs is ignored. */
  load(datasetId: string, onLoaded: () => void): void {
    if (this.loading()) return;
    this.error.set('');
    this.taskId.set('');
    this.onLoaded = onLoaded;
    this.datasetsRegistryApi
      .loadRegistered(datasetId)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (resp) => {
          const taskId = resp?.task_id ?? '';
          if (!taskId) {
            this.settle('');
            return;
          }
          this.taskId.set(taskId);
          this.loadingTasks.startProgressPolling(taskId, () => this.settle(taskId));
        },
        error: () => this.taskId.set(null),
      });
  }

  /** The dataset is loaded: call back, unless the load was forgotten since. */
  private settle(taskId: string): void {
    if (this.taskId() !== taskId) return;
    this.taskId.set(null);
    const onLoaded = this.onLoaded;
    this.onLoaded = null;
    onLoaded?.();
  }
}
