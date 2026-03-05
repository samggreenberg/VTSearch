import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SortedItem } from '../left-panel.component';

@Component({
  selector: 'vt-stripe-overview',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './stripe-overview.component.html',
  styleUrl: './stripe-overview.component.scss',
})
export class StripeOverviewComponent {
  @Input() sortOrder: SortedItem[] | null = null;
  @Input() threshold: number | null = null;
  @Input() selectedId: number | null = null;
  @Input() goodVotes: Set<number> = new Set();
  @Input() badVotes: Set<number> = new Set();
  @Input() totalCount = 0;

  get visible(): boolean {
    return this.sortOrder !== null && this.sortOrder.length > 0;
  }

  get dots(): { top: number; type: 'good' | 'bad' | 'selected' }[] {
    if (!this.sortOrder || this.sortOrder.length === 0) return [];

    const result: { top: number; type: 'good' | 'bad' | 'selected' }[] = [];
    const total = this.sortOrder.length;

    for (let i = 0; i < total; i++) {
      const item = this.sortOrder[i];
      const topPct = (i / total) * 100;

      if (this.goodVotes.has(item.id)) {
        result.push({ top: topPct, type: 'good' });
      } else if (this.badVotes.has(item.id)) {
        result.push({ top: topPct, type: 'bad' });
      }

      if (item.id === this.selectedId) {
        result.push({ top: topPct, type: 'selected' });
      }
    }

    return result;
  }

  get thresholdPosition(): number | null {
    if (!this.sortOrder || this.threshold === null) return null;
    const total = this.sortOrder.length;
    for (let i = 0; i < total; i++) {
      if (this.sortOrder[i].score < this.threshold) {
        return (i / total) * 100;
      }
    }
    return null;
  }
}
