import { Component, Input } from '@angular/core';

export interface UsageBytes {
  total: number;
  used: number;
  free: number;
}

@Component({
  selector: 'vt-usage-bar',
  standalone: true,
  imports: [],
  templateUrl: './usage-bar.component.html',
  styleUrl: './usage-bar.component.scss',
  host: {
    '[class.side-left]': "side === 'left'",
    '[class.side-right]': "side === 'right'",
  },
})
export class UsageBarComponent {
  @Input() usage: UsageBytes | null = null;
  @Input() label = '';
  @Input() side: 'left' | 'right' = 'right';
  @Input() titlePrefix = '';

  get usedPct(): number {
    if (!this.usage || this.usage.total <= 0) return 0;
    return (this.usage.used / this.usage.total) * 100;
  }

  get freeText(): string {
    if (!this.usage) return '';
    return `${this.formatBytes(this.usage.free)} free of ${this.formatBytes(this.usage.total)}`;
  }

  get title(): string {
    const prefix = this.titlePrefix || this.label;
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
