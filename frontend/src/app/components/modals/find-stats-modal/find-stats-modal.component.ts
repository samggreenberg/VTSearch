import { Component, OnInit, inject, output, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ModalComponent } from '../../modal/modal.component';
import { DetectorsFindApiService } from '../../../services/detectors-find-api.service';
import type { FindStatsResponse } from '../../../generated/api-client/models/find-stats-response';

/** A scaled point on the FP/FN-vs-inclusion chart. */
interface ChartPoint {
  inclusion: number;
  x: number;
  yFp: number;
  yFn: number;
}

/**
 * Detector-evaluation Stats for a Find run: the 2×2 confusion of the adopted
 * label set against the detector's original call, the derived agreement /
 * precision rates, and the headline FP/FN-vs-inclusion sweep rendered as a
 * dependency-free inline SVG line chart (current inclusion marked).
 * See docs/plans/find-verification-workflow.md.
 */
@Component({
  selector: 'vt-find-stats-modal',
  standalone: true,
  imports: [CommonModule, ModalComponent],
  templateUrl: './find-stats-modal.component.html',
  styleUrl: './find-stats-modal.component.scss',
})
export class FindStatsModalComponent implements OnInit {
  private findApi = inject(DetectorsFindApiService);

  readonly closed = output<void>();

  // Signalized so the `ngOnInit` subscribe (an unpatched callback under zoneless)
  // schedules CD when the stats land. See docs/plans/zoneless-migration.md.
  readonly loading = signal(true);
  readonly error = signal('');
  readonly stats = signal<FindStatsResponse | null>(null);

  // Chart geometry (SVG user units; the viewBox scales to the container width).
  readonly chartWidth = 320;
  readonly chartHeight = 150;
  private readonly padLeft = 34;
  private readonly padRight = 10;
  private readonly padTop = 10;
  private readonly padBottom = 22;

  ngOnInit(): void {
    this.findApi.getFindStats().subscribe({
      next: (data) => {
        this.stats.set(data);
        this.loading.set(false);
      },
      error: (err) => {
        this.error.set(err?.error?.error || err?.error?.message || 'Failed to load stats');
        this.loading.set(false);
      },
    });
  }

  close(): void {
    this.closed.emit();
  }

  get agreementPct(): string {
    const s = this.stats();
    return s ? `${Math.round(s.agreement_rate * 100)}%` : '-';
  }

  get precisionPct(): string {
    const s = this.stats();
    return s ? `${Math.round(s.precision * 100)}%` : '-';
  }

  // --- FP/FN-vs-inclusion chart -------------------------------------------

  private get plotW(): number {
    return this.chartWidth - this.padLeft - this.padRight;
  }

  private get plotH(): number {
    return this.chartHeight - this.padTop - this.padBottom;
  }

  /** Largest FP/FN count in the sweep; the chart's y-axis top (min 1). */
  get maxCount(): number {
    const s = this.stats();
    if (!s) return 1;
    let m = 1;
    for (const p of s.sweep) {
      m = Math.max(m, p.false_pos, p.false_neg);
    }
    return m;
  }

  private xFor(inclusion: number): number {
    return this.padLeft + ((inclusion + 10) / 20) * this.plotW;
  }

  private yFor(count: number): number {
    return this.padTop + (1 - count / this.maxCount) * this.plotH;
  }

  get points(): ChartPoint[] {
    const s = this.stats();
    if (!s) return [];
    return s.sweep.map((p) => ({
      inclusion: p.inclusion,
      x: this.xFor(p.inclusion),
      yFp: this.yFor(p.false_pos),
      yFn: this.yFor(p.false_neg),
    }));
  }

  get fpPolyline(): string {
    return this.points.map((p) => `${p.x.toFixed(1)},${p.yFp.toFixed(1)}`).join(' ');
  }

  get fnPolyline(): string {
    return this.points.map((p) => `${p.x.toFixed(1)},${p.yFn.toFixed(1)}`).join(' ');
  }

  /** X position of the current-inclusion marker line. */
  get currentX(): number {
    const s = this.stats();
    return s ? this.xFor(s.inclusion) : 0;
  }

  get axisTop(): number {
    return this.padTop;
  }

  get axisBottom(): number {
    return this.chartHeight - this.padBottom;
  }

  get axisLeft(): number {
    return this.padLeft;
  }

  get axisRight(): number {
    return this.chartWidth - this.padRight;
  }
}
