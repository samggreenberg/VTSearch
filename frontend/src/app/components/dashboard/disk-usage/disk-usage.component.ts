import { Component, Input } from '@angular/core';

export interface DiskUsageBytes {
  total: number;
  used: number;
  free: number;
}

@Component({
  selector: 'vt-disk-usage',
  standalone: true,
  imports: [],
  templateUrl: './disk-usage.component.html',
  styleUrl: './disk-usage.component.scss',
})
export class DiskUsageComponent {
  @Input() usage: DiskUsageBytes | null = null;

  get usedPct(): number {
    if (!this.usage || this.usage.total <= 0) return 0;
    return (this.usage.used / this.usage.total) * 100;
  }

  get freeText(): string {
    if (!this.usage) return '';
    return `${this.formatBytes(this.usage.free)} free of ${this.formatBytes(this.usage.total)}`;
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
