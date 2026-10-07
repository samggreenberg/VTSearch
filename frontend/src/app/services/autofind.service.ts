import { HttpContext } from '@angular/common/http';
import { Injectable, inject, signal } from '@angular/core';

import { SKIP_ERROR_TOAST } from '../interceptors/error.interceptor';
import { AutoDetectResultsData, AutoFindTaskInfo, LoadingTask } from '../models/api.models';
import { AuthService } from './auth.service';
import { DatasetsRegistryApiService } from './datasets-registry-api.service';
import { DetectorsFindApiService } from './detectors-find-api.service';
import { ProgressEventsService } from './progress-events.service';
import { ToastService } from './toast.service';

/**
 * Background AutoFind runs, from the browser's side (#4252).
 *
 * The server runs a user's AutoFind detectors on a dataset in two situations:
 * when a web import finishes (unless the Add Dataset dialog's **Run AutoFind**
 * box was unticked) and when the user picks **Run AutoFind** from a dataset's
 * ⋯ menu ({@link run}). The Dashboard's big **Find** button starts one run
 * per ticked dataset with the ticked detectors in place of the AutoFind list
 * ({@link find}, #4529). Either way the run is a `loading-tasks` row keyed to
 * the dataset, so its progress and Cancel button already show inline on the
 * Dashboard; this service handles the ending. It watches for AutoFind rows
 * reaching `idle` and, for the user who started them:
 *
 *  - a Find this tab started opens the Find Results dialog straight away (the
 *    user asked to look at the results and is waiting on them), unless the
 *    dialog is already showing another run - the button's runs land one per
 *    dataset - in which case it is announced as below rather than replacing
 *    what the user is reading;
 *  - any other run - an import's, a ⋯ Run AutoFind, or a Find from another
 *    tab - announces itself with a toast whose **View results** button opens
 *    the dialog. AutoFind is the unattended path, the GUI twin of the CLI's
 *    `--autodetect`: it says it is done and leaves the results a click away,
 *    and a user who wants to work with them runs Find instead (#4615);
 *  - an import whose dataset none of the user's AutoFind detectors applies to
 *    finishes as a `skipped` row, shown as an info toast giving the reason.
 *
 * The dialog itself is mounted once in `AppComponent`, reading {@link results},
 * so a run's results can be opened from any view. Failed runs need nothing
 * here: `ToastService` already turns a failed task row into an error toast.
 */
@Injectable({ providedIn: 'root' })
export class AutoFindService {
  private registryApi = inject(DatasetsRegistryApiService);
  private findApi = inject(DetectorsFindApiService);
  private progressEvents = inject(ProgressEventsService);
  private toast = inject(ToastService);
  private auth = inject(AuthService);

  /** The run the Find Results dialog shows, or `null` while it is closed. */
  readonly results = signal<AutoDetectResultsData | null>(null);

  private currentUser = '';
  /** Finds this tab started from the big button, whose results open when they finish. */
  private readonly openWhenDone = new Set<string>();
  /** Finished runs already acted on: SSE re-sends every row on each heartbeat
   *  until the task ages out, and a run must be announced once. */
  private readonly handled = new Set<string>();
  /** A results fetch is in flight, so the dialog is spoken for even before
   *  {@link results} is set: two runs landing in one SSE batch must not both
   *  open it, the second replacing the first. */
  private opening = false;

  constructor() {
    this.auth.status$.subscribe((status) => (this.currentUser = status?.user ?? ''));
    this.progressEvents.loadingTasks$.subscribe((tasks) => this.onTasks(tasks));
    // A restarted backend has forgotten every run; so do we.
    this.progressEvents.serverReset$.subscribe(() => {
      this.openWhenDone.clear();
      this.handled.clear();
    });
  }

  /** Run AutoFind on a loaded dataset: the current user's AutoFind detectors.
   *  A toast says when it is done, with the results a click away. A refusal
   *  (no detector applies, the dataset is not loaded) is toasted by the global
   *  error interceptor with the server's reason. */
  run(datasetId: string): void {
    this.registryApi.runAutofind(datasetId).subscribe();
  }

  /** Find: run exactly *detectorIds* on a loaded dataset, the same background
   *  run as {@link run}, and open the Find Results dialog when it finishes. */
  find(datasetId: string, detectorIds: string[]): void {
    this.registryApi.runAutofind(datasetId, detectorIds).subscribe({
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
    this.opening = true;
    this.findApi.getAutofindRun(runId, context).subscribe({
      next: (run) => {
        this.opening = false;
        this.results.set(run);
      },
      error: () => {
        this.opening = false;
        this.toast.warning({
          message: 'These AutoFind results are no longer available',
          detail: 'The server keeps only its most recent runs, and none from before a restart.',
        });
      },
    });
  }

  closeResults(): void {
    this.results.set(null);
  }

  private onTasks(tasks: LoadingTask[]): void {
    for (const task of tasks) {
      const info = task.autofind;
      if (!info || task.status !== 'idle' || this.handled.has(task.task_id)) continue;
      this.handled.add(task.task_id);
      const openNow = this.openWhenDone.delete(task.task_id);
      // Another user's run, or one that failed or was cancelled: nothing to show.
      if (task.error || info.owner !== this.currentUser) continue;
      if (info.skipped) {
        // The user asked for AutoFind on this import and none of their
        // detectors applies; say why rather than leave them waiting.
        this.toast.info({
          message: `AutoFind didn't run on "${info.dataset_name}"`,
          detail: info.skipped,
          dedupKey: `autofind:${info.run_id}`,
        });
        continue;
      }
      // A run the user asked for opens, unless the dialog already shows (or
      // is fetching) another: then the toast keeps this one a click away.
      if (openNow && !this.opening && this.results() === null) {
        this.openResults(info.run_id);
      } else {
        this.announce(info);
      }
    }
  }

  private announce(info: AutoFindTaskInfo): void {
    const hits = info.total_hits ?? 0;
    const detectors = info.detectors_run ?? 0;
    const detectorWord = `${detectors} detector${detectors === 1 ? '' : 's'}`;
    const what = info.trigger === 'find' ? 'Find' : 'AutoFind';
    const found = hits > 0 ? `${hits.toLocaleString()} hit${hits === 1 ? '' : 's'}` : 'no hits';
    const message = `${what} finished on "${info.dataset_name}": ${found}`;
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
        title: `Open the ${what} results for ${info.dataset_name}`,
        onClick: () => this.openResults(info.run_id),
      },
      // The button is the only way back to these results in the app, and the
      // run may have finished while the user was looking elsewhere.
      autoDismissMs: 0,
      dedupKey: `autofind:${info.run_id}`,
    });
  }
}
