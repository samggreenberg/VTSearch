import { ChangeDetectionStrategy, Component, computed, input, OnDestroy, output, signal } from '@angular/core';
import { IconComponent } from '../../icon/icon.component';
import { ToastyHintComponent } from '../../toasty-hint/toasty-hint.component';
import { HintId } from '../../../services/hints.service';
import { countPhrase } from '../../../utils/count-phrase';

/**
 * The hints that stand over the Good / Bad buttons (#4680, #4807). The host
 * picks one: Train's label view walks `start-voting` and the Autopilot phase
 * hints, and shows `revote` on an item the user picked to change (#4824); the
 * Tune view's stage shows `test-vote`.
 */
export type VoteHintId = Extract<
  HintId,
  'start-voting' | 'revote' | 'find-goods' | 'find-bads' | 'find-more' | 'keep-voting' | 'spot-check' | 'test-vote'
>;

/** What the Train phase hints count: the Goods and Bads so far, against Autopilot's targets. */
export interface VoteHintCounts {
  goods: number;
  goodsNeeded: number;
  bads: number;
  badsNeeded: number;
}

export const NO_VOTE_HINT_COUNTS: VoteHintCounts = { goods: 0, goodsNeeded: 0, bads: 0, badsNeeded: 0 };

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-voting-overlay',
  standalone: true,
  imports: [IconComponent, ToastyHintComponent],
  templateUrl: './voting-overlay.component.html',
  styleUrl: './voting-overlay.component.scss',
})
export class VotingOverlayComponent implements OnDestroy {
  readonly isGood = input(false);
  readonly isBad = input(false);
  /** Voting is momentarily blocked (a vote is in flight, Find is waiting on a
   *  score). Silently swallows clicks and keeps the buttons looking normal:
   *  these states last a few hundred milliseconds, and flickering the pair grey
   *  on every vote would be worse than doing nothing. */
  readonly disabled = input(false);
  /** There is nothing to vote on at all — the host's queue is empty. Unlike
   *  {@link disabled} this is a standing state the user has to be *told* about,
   *  so it sets the buttons' native `disabled` attribute: they grey out, stop
   *  taking hover, and drop out of the tab order. */
  readonly unavailable = input(false);
  readonly spinningVote = input<'good' | 'bad' | null>(null);
  /** Which of Toasty's hints stands over the buttons, if any (#4680, #4807).
   *  The host decides when, and swaps or drops it as the user votes. */
  readonly hint = input<VoteHintId | null>(null);
  /** The counts the Train phase hints fill in. */
  readonly hintCounts = input<VoteHintCounts>(NO_VOTE_HINT_COUNTS);
  readonly voted = output<'good' | 'bad'>();

  readonly goodsNeededPhrase = computed(() => countPhrase(this.hintCounts().goodsNeeded, 'Good', 'Goods'));
  readonly badsNeededPhrase = computed(() => countPhrase(this.hintCounts().badsNeeded, 'Bad', 'Bads'));
  /** What `revote` says the item is labeled, and the button and key that change it. The host
   *  drops the hint the moment the label moves, so these never name the label a vote just set. */
  readonly revoteLabel = computed(() => (this.isGood() ? 'Good' : 'Bad'));
  readonly revoteOther = computed(() => (this.isGood() ? 'Bad' : 'Good'));
  readonly revoteKey = computed(() => (this.isGood() ? 'left' : 'right'));

  /** Transient flash classes; signals so the `setTimeout` reset repaints under
   *  zoneless change detection. */
  readonly goodFlash = signal(false);
  readonly badFlash = signal(false);

  private goodTimer: ReturnType<typeof setTimeout> | null = null;
  private badTimer: ReturnType<typeof setTimeout> | null = null;

  onVoteGood(event?: Event): void {
    if (this.disabled() || this.unavailable()) return;
    this.dropFocus(event);
    this.goodFlash.set(true);
    this.voted.emit('good');
    if (this.goodTimer) clearTimeout(this.goodTimer);
    this.goodTimer = setTimeout(() => this.goodFlash.set(false), 300);
  }

  onVoteBad(event?: Event): void {
    if (this.disabled() || this.unavailable()) return;
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
