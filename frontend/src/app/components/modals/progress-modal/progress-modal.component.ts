import { ChangeDetectionStrategy, Component, ElementRef, inject, Input, input, OnDestroy, OnInit, output, ViewChild } from '@angular/core';

import { Subject, takeUntil, timer, switchMap, filter, take } from 'rxjs';
import { ModalComponent } from '../../modal/modal.component';
import { JobProgressComponent } from '../../job-progress/job-progress.component';
import { SortingApiService } from '../../../services/sorting-api.service';
import { ChartsService } from '../../../services/charts.service';
import { SettingsStateService } from '../../../services/settings-state.service';
import { ProgressEventsService } from '../../../services/progress-events.service';
import {
  ErrorCostDataPoint,
  StabilityDataPoint,
  DiversityDataPoint,
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

  @Input() metric: ProgressMetric = 'smart';
  readonly useCachedHistory = input(false);
  readonly closed = output<void>();

  @ViewChild('chartCanvas') chartCanvas!: ElementRef<HTMLCanvasElement>;

  analyzing = true;
  analysisProgress = 0;
  chartData: ErrorCostDataPoint[] | StabilityDataPoint[] | DiversityDataPoint[] = [];
  emptyHistory = false;
  /** Job id of the in-flight eval train-and-score run. Set once the
   *  backend hands back a job envelope; consumed by ``onCancel``. */
  private currentJobId: string | null = null;

  private destroy$ = new Subject<void>();

  get title(): string {
    switch (this.metric) {
      case 'smart':
        return 'Smart: Detector Accuracy Over Time';
      case 'stable':
        return 'Stable: How Often The Detector Changes Its Mind';
      case 'diverse':
        return 'Diverse: How Much Of Your Collection Your Votes Cover';
    }
  }

  ngOnInit(): void {
    if (this.useCachedHistory()) {
      this.loadCachedHistory();
    } else {
      this.runAnalysis();
    }
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  private loadCachedHistory(): void {
    this.analyzing = true;
    this.sortingApi.getIndicatorScoreHistory(this.metric).subscribe({
      next: (res) => {
        this.analyzing = false;
        this.chartData = (res.history || []) as ErrorCostDataPoint[] | StabilityDataPoint[] | DiversityDataPoint[];
        this.emptyHistory = this.chartData.length === 0;
        if (!this.emptyHistory) {
          setTimeout(() => this.renderChart(), 50);
        }
      },
      error: () => {
        this.analyzing = false;
        this.emptyHistory = true;
      },
    });
  }

  private runAnalysis(): void {
    this.analyzing = true;
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
      .trainAndScore(this.metric)
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
    if (this.metric === 'smart') {
      this.chartData = (res.error_cost || []) as ErrorCostDataPoint[];
    } else if (this.metric === 'stable') {
      this.chartData = (res.stability || []) as StabilityDataPoint[];
    } else {
      this.chartData = (res.diversity || []) as DiversityDataPoint[];
    }
    setTimeout(() => this.renderChart(), 50);
  }

  private renderChart(): void {
    if (!this.chartCanvas) return;
    const canvas = this.chartCanvas.nativeElement;
    switch (this.metric) {
      case 'smart':
        this.chartsService.renderErrorCostChart(canvas, this.chartData as ErrorCostDataPoint[]);
        break;
      case 'stable':
        this.chartsService.renderStabilityChart(canvas, this.chartData as StabilityDataPoint[]);
        break;
      case 'diverse':
        this.chartsService.renderDiversityChart(
          canvas,
          this.chartData as DiversityDataPoint[],
          this.settingsState.settingsSignal()?.autopilot_goal_diversity ?? 40,
        );
        break;
    }
  }

  close(): void {
    this.closed.emit();
  }
}
