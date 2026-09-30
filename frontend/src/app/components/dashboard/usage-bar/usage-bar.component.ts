import { ChangeDetectionStrategy, Component, input } from '@angular/core';

import { ProgressBarComponent } from '../../progress-bar/progress-bar.component';

export interface UsageBytes {
  total: number;
  used: number;
  free: number;
  /** Footprint of one dataset, the unit the server measures headroom in. */
  datasetBytes?: number;
  /** `largest` when measured from the registry, `default` for the stand-in. */
  datasetBytesSource?: string;
  /** Whether `free` holds fewer than a few more datasets of `datasetBytes`. */
  low?: boolean;
}

/** A `/api/dashboard/*-usage` probe's response, as the bar reads it. */
export function toUsageBytes(r: {
  total: number;
  used: number;
  free: number;
  dataset_bytes?: number;
  dataset_bytes_source?: string;
  low?: boolean;
}): UsageBytes {
  return {
    total: r.total,
    used: r.used,
    free: r.free,
    datasetBytes: r.dataset_bytes,
    datasetBytesSource: r.dataset_bytes_source,
    low: r.low,
  };
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

  /** The free space counted in datasets, which is what decides whether the
   *  bar shows on its Default setting: "room for 4 more datasets the size of
   *  the largest (2.1 GB)". Empty when the server sent no dataset size. */
  get headroomText(): string {
    const usage = this.usage();
    const unit = usage?.datasetBytes;
    if (!usage || !unit || unit <= 0) return '';
    const n = Math.floor(Math.max(0, usage.free) / unit);
    const size = this.formatBytes(unit);
    if (usage.datasetBytesSource === 'default') {
      return n === 0
        ? `no room for a typical ${size} dataset`
        : `room for ${n} typical ${size} dataset${n === 1 ? '' : 's'}`;
    }
    return n === 0
      ? `no room for another dataset the size of the largest (${size})`
      : `room for ${n} more dataset${n === 1 ? '' : 's'} the size of the largest (${size})`;
  }

  get title(): string {
    const prefix = this.titlePrefix() || this.label();
    const headroom = this.headroomText;
    const text = headroom ? `${this.freeText}, ${headroom}` : this.freeText;
    return prefix ? `${prefix}: ${text}` : text;
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
