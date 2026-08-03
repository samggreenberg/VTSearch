import { ChangeDetectionStrategy, Component, ElementRef, inject, input, OnDestroy, OnInit, output, viewChild } from '@angular/core';

import { Subject, takeUntil, timer, switchMap, filter, take, tap } from 'rxjs';
import { ModalComponent } from '../../modal/modal.component';
import { JobProgressComponent } from '../../job-progress/job-progress.component';
import { SortingApiService } from '../../../services/sorting-api.service';
import { ChartsService } from '../../../services/charts.service';
import { SettingsStateService } from '../../../services/settings-state.service';
import { ProgressEventsService } from '../../../services/progress-events.service';
import {
  ErrorCostPoint,
  StabilityPoint,
  DiversityPoint,
} from '../../../models/api.models';
import type { EvalTrainAndScoreResponse } from '../../../generated/api-client/models/eval-train-and-score-response';

export type ProgressMetric = 'smart' | 'stable' | 'diverse';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-progress-modal',
  standalone: true,
  imports: [ModalComponent, JobProgressComponent],
  templateUrl: './progress-modal.component.html',
  styleUrl: './progress-modal.component.scss',
})
export class ProgressModalComponent implements OnInit, OnDestroy {
  private sortingApi = inject(SortingApiService);
  private chartsService = inject(ChartsService);
  private settingsState = inject(SettingsStateService);
  private progressEvents = inject(ProgressEventsService);

  readonly metric = input<ProgressMetric>('smart');
  readonly closed = output<void>();

  // Optional query: the canvas only renders in the results `@else` branch.
  readonly chartCanvas = viewChild<ElementRef<HTMLCanvasElement>>('chartCanvas');

  analyzing = true;
  analysisProgress = 0;
  chartData: ErrorCostPoint[] | StabilityPoint[] | DiversityPoint[] = [];
  /** Series over the steps the running job has computed so far. Drawn into the
   *  same canvas while `analyzing` is still true, so a long walk shows a curve
   *  filling in instead of an empty modal. */
  partialData: ErrorCostPoint[] | StabilityPoint[] | DiversityPoint[] = [];
  emptyHistory = false;
  /** True once we've fallen back to the async train-and-score job, which
   *  swaps the brief "loading" line for a real progress bar + Cancel. */
  runningJob = false;
  /** Job id of the in-flight eval train-and-score run. Set once the
   *  backend hands back a job envelope; consumed by ``onCancel``. */
  private currentJobId: string | null = null;

  private destroy$ = new Subject<void>();

  get title(): string {
    switch (this.metric()) {
      case 'smart':
        return 'Smart: Detector Accuracy Over Time';
      case 'stable':
        return 'Stable: How Often The Detector Changes Its Mind';
      case 'diverse':
        return 'Diverse: How Much Of Your Collection Your Votes Cover';
    }
  }

  ngOnInit(): void {
    // Always try the cached read first: it never advances the per-step cache,
    // so it returns immediately whether or not the cache is warm. When the
    // background `/api/labeling-status` worker has kept up (the common case)
    // the plot paints instantly; otherwise we fall back to the async job,
    // which does the retraining off the request thread with a progress bar.
    this.loadCachedHistory();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  private loadCachedHistory(): void {
    this.analyzing = true;
    this.sortingApi
      .getIndicatorScoreHistory(this.metric())
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (res) => {
          // `complete: false` means the per-step cache is behind the label
          // history. The endpoint deliberately does not advance it (that build
          // is what used to hang this modal for tens of seconds), so hand off
          // to the background job — but paint whatever it *did* have first.
          // Those cached steps are the same ones the Smart/Stable light was
          // derived from, so there is no reason to make the user wait for a
          // full recompute before seeing them.
          if (!res.complete) {
            this.applyPartial(res.history);
            this.runAnalysis();
            return;
          }
          this.analyzing = false;
          this.chartData = (res.history || []) as ErrorCostPoint[] | StabilityPoint[] | DiversityPoint[];
          this.emptyHistory = this.chartData.length === 0;
          if (!this.emptyHistory) {
            setTimeout(() => this.renderChart(this.chartData), 50);
          }
        },
        // A failed cached read is not fatal: the job path recomputes the
        // series from scratch, so fall back rather than showing "no history".
        error: () => this.runAnalysis(),
      });
  }

  private runAnalysis(): void {
    this.analyzing = true;
    this.runningJob = true;
    this.analysisProgress = 0;

    // Progress comes from the `eval` SSE channel on /api/events. Use a
    // dedicated notifier to stop watching once the bar reaches 100%: the
    // backend emits the `idle/Done` eval frame *inside* `_run`, before the
    // job flips to `done`, so this fires while the result poller is still
    // polling. It must NOT tear down the poller — hence a separate subject
    // rather than `this.destroy$.next()`, which would kill the poller too
    // and leave `analyzing` hung forever.
    const stopWatchingProgress$ = new Subject<void>();
    this.progressEvents.votingIterations$
      .pipe(takeUntil(this.destroy$), takeUntil(stopWatchingProgress$))
      .subscribe({
        next: (res) => {
          if (res.total > 0) {
            this.analysisProgress = Math.round((res.progress / res.total) * 100);
          }
          if (res.done) {
            stopWatchingProgress$.next();
            stopWatchingProgress$.complete();
          }
        },
      });

    // Request train-and-score; the new endpoint returns a job envelope.
    // `takeUntil(destroy$)` guards the case where the modal is dismissed
    // while the POST is in flight: without it, a late `next` would arm
    // `pollEvalJob()` against an already-completed `destroy$` (RxJS
    // `takeUntil` never fires on a pre-completed notifier), leaking a poller.
    this.sortingApi
      .trainAndScore(this.metric())
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (res) => {
          if (res.status === 'done') {
            this.applyEvalResult(res);
          } else if (res.status === 'running') {
            this.currentJobId = res.job_id;
            this.pollEvalJob(res.job_id);
          } else {
            this.analyzing = false;
          }
        },
        error: () => {
          this.analyzing = false;
        },
      });
  }

  private pollEvalJob(jobId: string): void {
    timer(200, 500)
      .pipe(
        takeUntil(this.destroy$),
        switchMap(() => this.sortingApi.getEvalTrainAndScoreResult(jobId)),
        // `running` polls carry the series over the steps computed so far;
        // paint it so the modal shows a curve growing rather than a bare
        // progress bar, then keep polling.
        tap((res) => {
          if (res.status === 'running') this.applyPartial(res.partial);
        }),
        filter((res) => res.status !== 'running'),
        take(1),
      )
      .subscribe({
        next: (res) => {
          this.currentJobId = null;
          if (res.status === 'done') {
            this.applyEvalResult(res);
          } else {
            this.analyzing = false;
          }
        },
        error: () => {
          this.currentJobId = null;
          this.analyzing = false;
        },
      });
  }

  /** Render the in-progress series from a `running` poll.
   *
   *  Kept separate from `applyEvalResult`: `analyzing` deliberately stays true
   *  so the progress bar and Cancel button remain, and an empty partial is a
   *  no-op rather than the "no history" empty state — the job simply has not
   *  reached its first publish yet. */
  private applyPartial(partial: Array<Record<string, unknown>> | undefined): void {
    if (!partial?.length) return;
    this.partialData = partial as unknown as ErrorCostPoint[] | StabilityPoint[] | DiversityPoint[];
    setTimeout(() => this.renderChart(this.partialData), 0);
  }

  /** Cancel the in-flight eval job (if any) and close the modal.
   *
   *  This is the single dismissal path for the modal: the in-body Cancel
   *  button, and Escape / the X / a backdrop click (routed here from the
   *  inner `vt-modal`'s `(closed)`) all land here, so every way of leaving
   *  the modal stops the running eval job rather than orphaning it. Safe on
   *  the cached-history path, where `currentJobId` is null and no cancel
   *  request is sent. */
  onCancel(): void {
    const jobId = this.currentJobId;
    this.currentJobId = null;
    if (jobId) {
      this.sortingApi.cancelEvalTrainAndScore(jobId).pipe(takeUntil(this.destroy$)).subscribe();
    }
    this.analyzing = false;
    this.close();
  }

  private applyEvalResult(res: EvalTrainAndScoreResponse): void {
    this.analyzing = false;
    this.runningJob = false;
    this.partialData = [];
    if (this.metric() === 'smart') {
      this.chartData = (res.error_cost || []) as ErrorCostPoint[];
    } else if (this.metric() === 'stable') {
      this.chartData = (res.stability || []) as StabilityPoint[];
    } else {
      this.chartData = (res.diversity || []) as DiversityPoint[];
    }
    // Same empty-state handling as the cached path: a job that legitimately
    // produces no points (too little history) shows the explanatory message
    // rather than an empty set of axes.
    this.emptyHistory = this.chartData.length === 0;
    if (!this.emptyHistory) {
      setTimeout(() => this.renderChart(this.chartData), 50);
    }
  }

  private renderChart(data: ErrorCostPoint[] | StabilityPoint[] | DiversityPoint[]): void {
    const chartCanvas = this.chartCanvas();
    if (!chartCanvas) return;
    const canvas = chartCanvas.nativeElement;
    switch (this.metric()) {
      case 'smart':
        this.chartsService.renderErrorCostChart(canvas, data as ErrorCostPoint[]);
        break;
      case 'stable':
        this.chartsService.renderStabilityChart(canvas, data as StabilityPoint[]);
        break;
      case 'diverse':
        this.chartsService.renderDiversityChart(
          canvas,
          data as DiversityPoint[],
          this.settingsState.settingsSignal()?.autopilot_goal_diversity ?? 40,
        );
        break;
    }
  }

  close(): void {
    this.closed.emit();
  }
}
