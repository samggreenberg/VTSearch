import { Component, OnDestroy, input, output, signal } from '@angular/core';
import { IconComponent } from '../../icon/icon.component';

@Component({
  selector: 'vt-voting-overlay',
  standalone: true,
  imports: [IconComponent],
  templateUrl: './voting-overlay.component.html',
  styleUrl: './voting-overlay.component.scss',
})
export class VotingOverlayComponent implements OnDestroy {
  readonly isGood = input(false);
  readonly isBad = input(false);
  readonly disabled = input(false);
  readonly spinningVote = input<'good' | 'bad' | null>(null);
  /** When true, renders the faint first-vote hint above the buttons. The
   *  parent decides when to show this (zero votes + not previously dismissed)
   *  and dismisses it on first vote. */
  readonly showHint = input(false);
  readonly voted = output<'good' | 'bad'>();

  /** Transient flash classes; signals so the `setTimeout` reset repaints under
   *  zoneless change detection. */
  readonly goodFlash = signal(false);
  readonly badFlash = signal(false);

  private goodTimer: ReturnType<typeof setTimeout> | null = null;
  private badTimer: ReturnType<typeof setTimeout> | null = null;

  onVoteGood(event?: Event): void {
    if (this.disabled()) return;
    this.dropFocus(event);
    this.goodFlash.set(true);
    this.voted.emit('good');
    if (this.goodTimer) clearTimeout(this.goodTimer);
    this.goodTimer = setTimeout(() => this.goodFlash.set(false), 300);
  }

  onVoteBad(event?: Event): void {
    if (this.disabled()) return;
    this.dropFocus(event);
    this.badFlash.set(true);
    this.voted.emit('bad');
    if (this.badTimer) clearTimeout(this.badTimer);
    this.badTimer = setTimeout(() => this.badFlash.set(false), 300);
  }

  /**
   * Blur the clicked button so it doesn't keep DOM focus. Without this, the
   * button stays focused after a mouse click; the focus ring is suppressed by
   * `:focus:not(:focus-visible)` only until the user presses a modifier (e.g.
   * Shift to draw a region), at which point the browser promotes the focused
   * element to `:focus-visible` and paints a ring around the last-voted button.
   * Dropping focus on click keeps Shift purely a cursor modifier.
   */
  private dropFocus(event?: Event): void {
    const target = event?.currentTarget;
    if (target instanceof HTMLElement) target.blur();
  }

  ngOnDestroy(): void {
    if (this.goodTimer) clearTimeout(this.goodTimer);
    if (this.badTimer) clearTimeout(this.badTimer);
  }
}
