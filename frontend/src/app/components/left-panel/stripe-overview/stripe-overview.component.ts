import { Component, Input, Output, EventEmitter } from '@angular/core';
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
  @Input() scrollInfo: { scrollTop: number; scrollHeight: number; clientHeight: number } | null = null;

  @Output() stripeClick = new EventEmitter<number>();

  onStripeClick(event: MouseEvent): void {
    if (!this.sortOrder || this.sortOrder.length === 0) return;
    const el = event.currentTarget as HTMLElement;
    const rect = el.getBoundingClientRect();
    const y = event.clientY - rect.top;
    const percentage = y / rect.height;
    const index = Math.max(0, Math.min(Math.floor(percentage * this.sortOrder.length), this.sortOrder.length - 1));
    this.stripeClick.emit(index);
  }

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

  get viewportTop(): number {
    if (!this.scrollInfo || this.scrollInfo.scrollHeight === 0) return 0;
    return (this.scrollInfo.scrollTop / this.scrollInfo.scrollHeight) * 100;
  }

  get viewportHeight(): number {
    if (!this.scrollInfo || this.scrollInfo.scrollHeight === 0) return 100;
    return (this.scrollInfo.clientHeight / this.scrollInfo.scrollHeight) * 100;
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
