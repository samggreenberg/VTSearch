import { ChangeDetectionStrategy, Component, computed, input, output } from '@angular/core';

import { SortedItem } from '../left-panel.component';

/**
 * Sort size above which the minimap is disabled rather than drawn.  The stripe
 * is a minimap of the *entire* sort order — every dot is positioned by an item's
 * rank across the whole ranking — so beyond a large-sort size it is both
 * O(N)-expensive to build and, once the frontend moves to a windowed sort model
 * (scalability.md S3/S17/S19), impossible to draw honestly from the loaded
 * window.  Above this count we render a disabled strip with a clear tooltip
 * instead of a misleading picture.  This is the natural home for the future
 * window's own cutoff.
 */
export const STRIPE_MAX_ITEMS = 20000;

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-stripe-overview',
  standalone: true,
  imports: [],
  templateUrl: './stripe-overview.component.html',
  styleUrl: './stripe-overview.component.scss',
})
export class StripeOverviewComponent {
  readonly sortOrder = input<SortedItem[] | null>(null);
  readonly threshold = input<number | null>(null);
  readonly selectedId = input<number | null>(null);
  readonly goodVotes = input<Set<number>>(new Set());
  readonly badVotes = input<Set<number>>(new Set());
  readonly totalCount = input(0);
  readonly stripeClick = output<number>();

  /** Dots, recomputed automatically when the inputs they depend on change. */
  readonly cachedDots = computed(() => this.buildDots());
  /** Threshold position, recomputed automatically when inputs change. */
  readonly cachedThresholdPosition = computed(() => this.buildThresholdPosition());

  /**
   * True when the sort is too large for an honest minimap.  In this state the
   * strip stays in the layout (so panels don't shift) but shows a disabled
   * placeholder with an explanatory tooltip instead of dots.
   */
  readonly oversized = computed(() => (this.sortOrder()?.length ?? 0) > STRIPE_MAX_ITEMS);

  onStripeKeyboard(): void {
    if (this.oversized()) return;
    const sortOrder = this.sortOrder();
    if (!sortOrder || sortOrder.length === 0) return;
    const midIndex = Math.floor(sortOrder.length / 2);
    this.stripeClick.emit(midIndex);
  }

  onStripeClick(event: MouseEvent): void {
    if (this.oversized()) return;
    const sortOrder = this.sortOrder();
    if (!sortOrder || sortOrder.length === 0) return;
    const el = event.currentTarget as HTMLElement;
    const rect = el.getBoundingClientRect();
    const y = event.clientY - rect.top;
    const percentage = y / rect.height;
    const index = Math.max(0, Math.min(Math.floor(percentage * sortOrder.length), sortOrder.length - 1));
    this.stripeClick.emit(index);
  }

  get visible(): boolean {
    const sortOrder = this.sortOrder();
    return sortOrder !== null && sortOrder.length > 0;
  }

  private buildDots(): { top: number; type: 'good' | 'bad' | 'selected' }[] {
    const sortOrder = this.sortOrder();
    if (!sortOrder || sortOrder.length === 0 || sortOrder.length > STRIPE_MAX_ITEMS) return [];

    const goodVotes = this.goodVotes();
    const badVotes = this.badVotes();
    const selectedId = this.selectedId();
    const result: { top: number; type: 'good' | 'bad' | 'selected' }[] = [];
    const total = sortOrder.length;

    for (let i = 0; i < total; i++) {
      const item = sortOrder[i];
      const topPct = (i / total) * 100;

      if (goodVotes.has(item.id)) {
        result.push({ top: topPct, type: 'good' });
      } else if (badVotes.has(item.id)) {
        result.push({ top: topPct, type: 'bad' });
      }

      if (item.id === selectedId) {
        result.push({ top: topPct, type: 'selected' });
      }
    }

    return result;
  }

  private buildThresholdPosition(): number | null {
    const sortOrder = this.sortOrder();
    const threshold = this.threshold();
    if (!sortOrder || threshold === null || sortOrder.length > STRIPE_MAX_ITEMS) return null;
    const total = sortOrder.length;
    for (let i = 0; i < total; i++) {
      if (sortOrder[i].score < threshold) {
        return (i / total) * 100;
      }
    }
    return null;
  }
}
