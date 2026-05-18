import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ProgressBarComponent } from '../../progress-bar/progress-bar.component';
import type { LabelingStatusResponse } from '../../../generated/api-client/models/labeling-status-response';

@Component({
  selector: 'vt-progress-indicators',
  standalone: true,
  imports: [CommonModule, ProgressBarComponent],
  templateUrl: './progress-indicators.component.html',
  styleUrl: './progress-indicators.component.scss',
})
export class ProgressIndicatorsComponent {
  @Input() labelingStatus: LabelingStatusResponse | null = null;
  @Input() sortBusy = false;
  @Input() sortStatus = '';
  @Input() sortProgress = 0;
  @Input() sortProgressTotal = 0;

  @Output() indicatorClick = new EventEmitter<string>();

  get isIndeterminate(): boolean {
    return this.sortProgressTotal <= 0;
  }

  get smartStatus(): string {
    return this.labelingStatus?.smart.status || '';
  }

  get stableStatus(): string {
    return this.labelingStatus?.stable.status || '';
  }

  get spanStatus(): string {
    return this.labelingStatus?.span.status || '';
  }

  get smartSubtext(): string {
    if (!this.labelingStatus?.smart) return '';
    const s = this.labelingStatus.smart;
    if (s['cost'] != null) return `Cost: ${(s['cost'] as number).toFixed(3)}`;
    return '';
  }

  get stableSubtext(): string {
    if (!this.labelingStatus?.stable) return '';
    const s = this.labelingStatus.stable;
    if (s['flips'] != null) return `Flips: ${s['flips']}`;
    return '';
  }

  get spanSubtext(): string {
    if (!this.labelingStatus?.span) return '';
    const s = this.labelingStatus.span;
    if (s['diversity_level'] != null && s['max_level'] != null) {
      return `${Math.round(s['diversity_level'] as number)}/${s['max_level']}`;
    }
    return '';
  }

  onClick(name: string): void {
    this.indicatorClick.emit(name);
  }
}
