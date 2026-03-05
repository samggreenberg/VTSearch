import { Component, ElementRef, EventEmitter, Input, OnDestroy, OnInit, Output, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subject, takeUntil, timer, switchMap } from 'rxjs';
import { ModalComponent } from '../../modal/modal.component';
import { ProgressBarComponent } from '../../progress-bar/progress-bar.component';
import { SortingApiService } from '../../../services/sorting-api.service';
import { ChartsService } from '../../../services/charts.service';
import {
  ErrorCostDataPoint,
  StabilityDataPoint,
  DiversityDataPoint,
} from '../../../models/api.models';

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
  @Output() closed = new EventEmitter<void>();

  @ViewChild('chartCanvas') chartCanvas!: ElementRef<HTMLCanvasElement>;

  analyzing = true;
  analysisProgress = 0;
  chartData: ErrorCostDataPoint[] | StabilityDataPoint[] | DiversityDataPoint[] = [];

  private destroy$ = new Subject<void>();

  constructor(
    private sortingApi: SortingApiService,
    private chartsService: ChartsService,
  ) {}

  get title(): string {
    switch (this.metric) {
      case 'smart':
        return 'Smart: Error Cost Analysis';
      case 'stable':
        return 'Stable: Prediction Flip Analysis';
      case 'diverse':
        return 'Diverse: Diversity Coverage';
    }
  }

  ngOnInit(): void {
    this.runAnalysis();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  private runAnalysis(): void {
    this.analyzing = true;
    this.analysisProgress = 0;

    // Poll progress
    timer(0, 500)
      .pipe(
        takeUntil(this.destroy$),
        switchMap(() => this.sortingApi.getVotingIterations()),
      )
      .subscribe({
        next: (res) => {
          if (res.total > 0) {
            this.analysisProgress = Math.round((res.progress / res.total) * 100);
          }
          if (res.done) {
            this.destroy$.next(); // stop polling
          }
        },
      });

    // Request train-and-score
    this.sortingApi.trainAndScore(this.metric).subscribe({
      next: (res) => {
        this.analyzing = false;
        if (this.metric === 'smart') {
          this.chartData = res.error_cost || [];
        } else if (this.metric === 'stable') {
          this.chartData = res.stability || [];
        } else {
          this.chartData = res.diversity || [];
        }
        setTimeout(() => this.renderChart(), 50);
      },
      error: () => {
        this.analyzing = false;
      },
    });
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
        this.chartsService.renderDiversityChart(canvas, this.chartData as DiversityDataPoint[]);
        break;
    }
  }

  close(): void {
    this.closed.emit();
  }
}
