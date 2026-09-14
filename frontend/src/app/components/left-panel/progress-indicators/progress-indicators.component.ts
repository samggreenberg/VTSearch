import { ChangeDetectionStrategy, Component, input, output } from '@angular/core';

import { JobProgressComponent } from '../../job-progress/job-progress.component';
import {
  ProgressBarState,
  formatEta,
  progressBarState,
} from '../../../utils/format-progress';
import type { LabelingStatusResponse } from '../../../generated/api-client/models/labeling-status-response';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-progress-indicators',
  standalone: true,
  imports: [JobProgressComponent],
  templateUrl: './progress-indicators.component.html',
  styleUrl: './progress-indicators.component.scss',
})
export class ProgressIndicatorsComponent {
  readonly labelingStatus = input<LabelingStatusResponse | null>(null);
  readonly sortBusy = input(false);
  readonly sortStatus = input('');
  readonly sortProgress = input(0);
  readonly sortProgressTotal = input(0);
  /** Whole-job fraction (0..1) for multi-step scoring; ``null`` falls back to
   *  current/total. See ProgressEvent.overall. */
  readonly sortOverall = input<number | null>(null);
  /** Whole-job fraction where the current step's slice ends; bounds the
   *  pulsing zone during a count-less step. See ProgressEvent.overall_step_end. */
  readonly sortStepEnd = input<number | null>(null);
  /** Overall remaining-seconds estimate; ``null`` hides the ETA chip. */
  readonly sortEtaSeconds = input<number | null>(null);

  readonly indicatorClick = output<string>();
  readonly cancel = output<void>();

  onCancel(): void {
    this.cancel.emit();
  }

  /** Unified bar state: prefers the whole-job ``overall`` fraction so the bar
   *  fills once across all phases instead of resetting per phase. */
  get sortBar(): ProgressBarState {
    return progressBarState({
      current: this.sortProgress(),
      total: this.sortProgressTotal(),
      overall: this.sortOverall(),
      overall_step_end: this.sortStepEnd(),
    });
  }

  /** Overall ETA chip (empty when no estimate is available). */
  get sortEta(): string {
    return formatEta(this.sortEtaSeconds());
  }

  get isIndeterminate(): boolean {
    return this.sortBar.indeterminate;
  }

  get smartStatus(): string {
    return this.labelingStatus()?.smart.status || '';
  }

  get stableStatus(): string {
    return this.labelingStatus()?.stable.status || '';
  }

  get spanStatus(): string {
    return this.labelingStatus()?.span.status || '';
  }

  get smartSubtext(): string {
    const status = this.labelingStatus();
    if (!status?.smart) return '';
    const s = status.smart;
    if (s['cost'] != null) return `Cost: ${(s['cost'] as number).toFixed(3)}`;
    return '';
  }

  get stableSubtext(): string {
    const status = this.labelingStatus();
    if (!status?.stable) return '';
    const s = status.stable;
    if (s['avg_flip_rate'] != null) return `Flips: ${((s['avg_flip_rate'] as number) * 100).toFixed(1)}% of items per retrain`;
    return '';
  }

  /** Set when Stable is green only because the remaining flips are irreducible boundary wobble (#3831). */
  get stablePlateau(): boolean {
    return this.labelingStatus()?.stable['plateau'] === true;
  }

  get spanSubtext(): string {
    const status = this.labelingStatus();
    if (!status?.span) return '';
    const s = status.span;
    if (s['diversity_level'] != null && s['max_level'] != null) {
      return `${Math.round(s['diversity_level'] as number)}/${s['max_level']}`;
    }
    return '';
  }

  /** Set when Smart is green only because the cost's downward drift is smaller
   *  than its step-to-step scatter (#3832). */
  get smartDriftWithinNoise(): boolean {
    return this.labelingStatus()?.smart['drift_within_noise'] === true;
  }

  get smartTooltip(): string {
    const meaning = 'Smart: the model fits your votes consistently.';
    const drift = this.smartDriftWithinNoise
      ? ' The cost still drifts down, but by less than it bounces around between retrains, so the drift is noise rather than progress.'
      : '';
    return this.smartSubtext ? `${meaning}${drift} ${this.smartSubtext}.` : `${meaning}${drift}`;
  }

  get stableTooltip(): string {
    const meaning = 'Stable: predictions stopped shifting between retrains.';
    const plateau = this.stablePlateau
      ? ' Some items near the cutoff still change calls, but no longer fewer each retrain: the remaining ambiguity looks irreducible in this embedding.'
      : '';
    return this.stableSubtext ? `${meaning}${plateau} ${this.stableSubtext}.` : `${meaning}${plateau}`;
  }

  get spanTooltip(): string {
    const meaning = 'Diverse: your votes cover the dataset broadly.';
    return this.spanSubtext ? `${meaning} Level ${this.spanSubtext}.` : meaning;
  }

  onClick(name: string): void {
    this.indicatorClick.emit(name);
  }
}
