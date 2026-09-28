import { HttpContext } from '@angular/common/http';
import { Injectable, inject, signal } from '@angular/core';

import { SKIP_ERROR_TOAST } from '../interceptors/error.interceptor';
import { AutoDetectResultsData, AutoRunTaskInfo, LoadingTask } from '../models/api.models';
import { AuthService } from './auth.service';
import { DatasetsRegistryApiService } from './datasets-registry-api.service';
import { DetectorsFindApiService } from './detectors-find-api.service';
import { ProgressEventsService } from './progress-events.service';
import { ToastService } from './toast.service';

/**
 * Background AutoRun runs, from the browser's side (#4252).
 *
 * The server runs a user's AutoRun detectors on a dataset in two situations:
 * when a web import finishes (unless the Add Dataset dialog's **Run AutoRun**
 * box was unticked) and when the user picks **Run AutoRun** from a dataset's
 * ⋯ menu ({@link run}). Either way the run is a `loading-tasks` row keyed to
 * the dataset, so its progress and Cancel button already show inline on the
 * Dashboard; this service handles the ending. It watches for AutoRun rows
 * reaching `idle` and, for the user who started them:
 *
 *  - a run this tab started from the ⋯ menu opens the AutoRun Results dialog
 *    straight away (the user asked for it and is waiting on it);
 *  - any other run - an import's, or a ⋯ run from another tab - announces
 *    itself with a toast whose **View results** button opens the dialog, since
 *    it finishes on its own schedule, possibly while the user is elsewhere;
 *  - an import whose dataset none of the user's AutoRun detectors applies to
 *    finishes as a `skipped` row, shown as an info toast giving the reason.
 *
 * The dialog itself is mounted once in `AppComponent`, reading {@link results},
 * so a run's results can be opened from any view. Failed runs need nothing
 * here: `ToastService` already turns a failed task row into an error toast.
 */
@Injectable({ providedIn: 'root' })
export class AutoRunService {
  private registryApi = inject(DatasetsRegistryApiService);
  private findApi = inject(DetectorsFindApiService);
  private progressEvents = inject(ProgressEventsService);
  private toast = inject(ToastService);
  private auth = inject(AuthService);

  /** The run the AutoRun Results dialog shows, or `null` while it is closed. */
  readonly results = signal<AutoDetectResultsData | null>(null);

  private currentUser = '';
  /** Runs this tab started from the ⋯ menu, whose results open when they finish. */
  private readonly openWhenDone = new Set<string>();
  /** Finished runs already acted on: SSE re-sends every row on each heartbeat
   *  until the task ages out, and a run must be announced once. */
  private readonly handled = new Set<string>();

  constructor() {
    this.auth.status$.subscribe((status) => (this.currentUser = status?.user ?? ''));
    this.progressEvents.loadingTasks$.subscribe((tasks) => this.onTasks(tasks));
    // A restarted backend has forgotten every run; so do we.
    this.progressEvents.serverReset$.subscribe(() => {
      this.openWhenDone.clear();
      this.handled.clear();
    });
  }

  /** Run the current user's AutoRun detectors on a loaded dataset; the results
   *  dialog opens when the run finishes. A refusal (no AutoRun detector
   *  applies, the dataset is not loaded) is toasted by the global error
   *  interceptor with the server's reason. */
  run(datasetId: string): void {
    this.registryApi.runAutorun(datasetId).subscribe({
      next: (res) => {
        if (res.task_id) this.openWhenDone.add(res.task_id);
      },
    });
  }

  /** Fetch a finished run's results and open them in the dialog. */
  openResults(runId: string): void {
    // A 404 here is not a failed request but an expired run (the server keeps
    // only a few, and none across a restart): say that instead of the raw error.
    const context = new HttpContext().set(SKIP_ERROR_TOAST, true);
    this.findApi.getAutorunRun(runId, context).subscribe({
      next: (run) => this.results.set(run),
      error: () =>
        this.toast.warning({
          message: 'These AutoRun results are no longer available',
          detail: 'The server keeps only its most recent runs, and none from before a restart.',
        }),
    });
  }

  closeResults(): void {
    this.results.set(null);
  }

  private onTasks(tasks: LoadingTask[]): void {
    for (const task of tasks) {
      const info = task.autorun;
      if (!info || task.status !== 'idle' || this.handled.has(task.task_id)) continue;
      this.handled.add(task.task_id);
      const openNow = this.openWhenDone.delete(task.task_id);
      // Another user's run, or one that failed or was cancelled: nothing to show.
      if (task.error || info.owner !== this.currentUser) continue;
      if (info.skipped) {
        // The user asked for AutoRun on this import and none of their
        // detectors applies; say why rather than leave them waiting.
        this.toast.info({
          message: `AutoRun didn't run on "${info.dataset_name}"`,
          detail: info.skipped,
          dedupKey: `autorun:${info.run_id}`,
        });
        continue;
      }
      if (openNow) {
        this.openResults(info.run_id);
      } else {
        this.announce(info);
      }
    }
  }

  private announce(info: AutoRunTaskInfo): void {
    const hits = info.total_hits ?? 0;
    const detectors = info.detectors_run ?? 0;
    const detectorWord = `${detectors} detector${detectors === 1 ? '' : 's'}`;
    const message =
      hits > 0
        ? `AutoRun found ${hits.toLocaleString()} hit${hits === 1 ? '' : 's'} in "${info.dataset_name}"`
        : `AutoRun found no hits in "${info.dataset_name}"`;
    const exportFailed = info.auto_export?.success === false;
    const detail = exportFailed
      ? `${detectorWord}. Sending the results via ${info.auto_export?.exporter} failed: ${info.auto_export?.error ?? 'unknown error'}`
      : info.auto_export?.success
        ? `${detectorWord}. ${info.auto_export.message ?? `Sent via ${info.auto_export.exporter}.`}`
        : `${detectorWord}.`;
    const show = exportFailed ? this.toast.warning.bind(this.toast) : this.toast.success.bind(this.toast);
    show({
      message,
      detail,
      action: {
        label: 'View results',
        title: `Open the AutoRun results for ${info.dataset_name}`,
        onClick: () => this.openResults(info.run_id),
      },
      // The button is the only way back to these results in the app, and the
      // run may have finished while the user was looking elsewhere.
      autoDismissMs: 0,
      dedupKey: `autorun:${info.run_id}`,
    });
  }
}
