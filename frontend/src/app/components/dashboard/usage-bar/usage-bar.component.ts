import { ChangeDetectionStrategy, Component, input } from '@angular/core';

import { ProgressBarComponent } from '../../progress-bar/progress-bar.component';

export interface UsageBytes {
  total: number;
  used: number;
  free: number;
}

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-usage-bar',
  standalone: true,
  imports: [ProgressBarComponent],
  templateUrl: './usage-bar.component.html',
  styleUrl: './usage-bar.component.scss',
  host: {
    '[class.side-left]': "side() === 'left'",
    '[class.side-right]': "side() === 'right'",
  },
})
export class UsageBarComponent {
  readonly usage = input<UsageBytes | null>(null);
  readonly label = input('');
  readonly side = input<'left' | 'right'>('right');
  readonly titlePrefix = input('');

  get usedPct(): number {
    const usage = this.usage();
    if (!usage || usage.total <= 0) return 0;
    return (usage.used / usage.total) * 100;
  }

  get freeText(): string {
    const usage = this.usage();
    if (!usage) return '';
    return `${this.formatBytes(usage.free)} free of ${this.formatBytes(usage.total)}`;
  }

  get title(): string {
    const prefix = this.titlePrefix() || this.label();
    return prefix ? `${prefix}: ${this.freeText}` : this.freeText;
  }

  private formatBytes(n: number): string {
    if (n < 1024) return `${n} B`;
    const units = ['KB', 'MB', 'GB', 'TB', 'PB'];
    let v = n / 1024;
    let i = 0;
    while (v >= 1024 && i < units.length - 1) {
      v /= 1024;
      i++;
    }
    return `${v >= 100 ? v.toFixed(0) : v.toFixed(1)} ${units[i]}`;
  }
}
