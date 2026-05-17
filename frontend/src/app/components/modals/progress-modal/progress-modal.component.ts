import { Component, ElementRef, EventEmitter, Input, OnDestroy, OnInit, Output, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subject, takeUntil, timer, switchMap, filter, take } from 'rxjs';
import { ModalComponent } from '../../modal/modal.component';
import { ProgressBarComponent } from '../../progress-bar/progress-bar.component';
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
  selector: 'vt-progress-modal',
  standalone: true,
  imports: [CommonModule, ModalComponent, ProgressBarComponent],
  templateUrl: './progress-modal.component.html',
  styleUrl: './progress-modal.component.scss',
})
export class ProgressModalComponent implements OnInit, OnDestroy {
  @Input() metric: ProgressMetric = 'smart';
  @Input() useCachedHistory = false;
  @Output() closed = new EventEmitter<void>();

  @ViewChild('chartCanvas') chartCanvas!: ElementRef<HTMLCanvasElement>;

  analyzing = true;
  analysisProgress = 0;
  chartData: ErrorCostDataPoint[] | StabilityDataPoint[] | DiversityDataPoint[] = [];
  emptyHistory = false;

  private destroy$ = new Subject<void>();

  constructor(
    private sortingApi: SortingApiService,
    private chartsService: ChartsService,
    private settingsState: SettingsStateService,
    private progressEvents: ProgressEventsService,
  ) {}

  get title(): string {
    switch (this.metric) {
      case 'smart':
        return 'Smart: Error Cost Over Time';
      case 'stable':
        return 'Stable: Prediction Flips Over Time';
      case 'diverse':
        return 'Diverse: Diversity Coverage Over Time';
    }
  }

  ngOnInit(): void {
    if (this.useCachedHistory) {
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

    // Progress comes from the `eval` SSE channel on /api/events.
    this.progressEvents.votingIterations$
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (res) => {
          if (res.total > 0) {
            this.analysisProgress = Math.round((res.progress / res.total) * 100);
          }
          if (res.done) {
            this.destroy$.next(); // stop watching
          }
        },
      });

    // Request train-and-score; the new endpoint returns a job envelope.
    this.sortingApi.trainAndScore(this.metric).subscribe({
      next: (res) => {
        if (res.status === 'done') {
          this.applyEvalResult(res);
        } else if (res.status === 'running') {
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
          if (res.status === 'done') {
            this.applyEvalResult(res);
          } else {
            this.analyzing = false;
          }
        },
        error: () => {
          this.analyzing = false;
        },
      });
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
          this.settingsState.settings?.autopilot_goal_diversity ?? 40,
        );
        break;
    }
  }

  close(): void {
    this.closed.emit();
  }
}
