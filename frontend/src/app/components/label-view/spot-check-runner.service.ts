import { DestroyRef, Injectable, computed, inject, signal } from '@angular/core';
import { type Subscription, finalize } from 'rxjs';
import { MediaMetadataCacheService } from '../../services/media-metadata-cache.service';
import { MediaStateService } from '../../services/media-state.service';
import { PairScopeService } from '../../services/pair-scope.service';
import { SortingApiService } from '../../services/sorting-api.service';
import { VoteStateService } from '../../services/vote-state.service';
import type { PrecisionCheckResponse } from '../../generated/api-client/models/precision-check-response';
import type { PrecisionCheckVote } from '../../generated/api-client/models/precision-check-vote';

/** Where the check is: none, drawing its first picks, on a round, or sending one. */
type Status = 'idle' | 'starting' | 'running' | 'sending';

/**
 * Autopilot's Spot Check step, run inside the Train view's ordinary vote flow
 * (#5136). The step is #4273's, walking the ranking in bands since #4388, and
 * only Autopilot starts it (`AutopilotStateService` phase `spot`): before
 * Done, and earlier while the labels separate weakly (#4496).
 *
 * The check is just another way to pick the next item. Nothing on screen
 * changes for it: no heading, no button to stop it, no numbers. The centre
 * panel shows its picks one at a time and the user votes on them exactly as
 * on Boundary's or Diversity's, keys, undo and the ↓ walk included.
 *
 * The server cuts the unvoted ranking into bands (the top 8, the next 8, then
 * 16, 32, ...) and draws each band's 5 picks uniformly at random. A band at a
 * time, the check walks deeper while its estimate of the balance (F-beta)
 * does not fall, and back to a shorter list once it does. Its result never
 * moves the line (#4452); its picks are ordinary votes the next retrain learns
 * from.
 *
 * - **It starts where the user is.** The host starts it as the step begins,
 *   but the item on screen stays: the next advance is the check's first pick,
 *   as a phase change waits for the next advance.
 * - **A round is sent whole.** Each vote on a pick is an ordinary vote the
 *   moment it is cast. The round goes to the check once every pick carries a
 *   label, with the labels they carry then, so a slip put right with ↓ counts
 *   as corrected and an undone pick is shown again. The server records them as
 *   check votes (provenance `check`). The advance into the next round waits on
 *   that round trip, which {@link advancePending} reports.
 * - **Ending leaves the state as it was.** Stopping Autopilot, or the view
 *   going away, under a running check cancels it server-side, after handing it
 *   the picks already voted this round so they carry its provenance; the
 *   rounds already sent stay votes. A request that fails ends the step the same
 *   way, quietly, and Autopilot carries on.
 */
@Injectable()
export class SpotCheckRunnerService {
  private readonly sortingApi = inject(SortingApiService);
  private readonly mediaState = inject(MediaStateService);
  private readonly metadataCache = inject(MediaMetadataCacheService);
  private readonly voteState = inject(VoteStateService);
  private readonly pairScope = inject(PairScopeService);

  private readonly status = signal<Status>('idle');
  /** This round's picks, in draw order, until the round goes to the server. */
  private readonly picks = signal<readonly number[]>([]);
  /** An advance came while the server had the next picks to deal: its response makes it. */
  private readonly owed = signal(false);
  /** The start or round in the air, dropped with the check so a late answer cannot revive it. */
  private inFlight: Subscription | null = null;
  /** A round has gone to the server. */
  private readonly sent = signal(false);

  /**
   * Set by the host: the check ended on its own. *finished* is `true` when it
   * ended with a result, `false` when the server refused it or a request
   * failed; *owed* when an advance was waiting on it, which the host then
   * makes. Not called for {@link cancel}, which the host asks for.
   */
  ended: (finished: boolean, owed: boolean) => void = () => undefined;

  /** The next item is a round trip away: the centre keeps the voted item swiped off until it lands (#4307). */
  readonly advancePending = computed(() => this.owed());

  /** No pick of the check has a vote yet: Toasty's `spot-check` hint is for its first (#4807). */
  readonly fresh = computed(() => this.status() === 'running' && !this.sent() && this.labeledVotes().length === 0);

  constructor() {
    // Gone from under a running check (the view left): cancel it, as stopping Autopilot does.
    inject(DestroyRef).onDestroy(() => this.cancel());
  }

  /** Whether *id* is one of the round's picks: what the centre shows while the check runs. */
  isPick(id: number): boolean {
    return this.picks().includes(id);
  }

  /** Draw the first round. A no-op while a check is under way. */
  start(): void {
    if (this.status() !== 'idle') return;
    this.status.set('starting');
    this.inFlight = this.sortingApi
      .startPrecisionCheck()
      .pipe(this.pairScope.scoped())
      .subscribe({
        next: (resp) => this.apply(resp),
        error: () => this.end(false),
      });
  }

  /**
   * Show the round's next unlabeled pick: the advance a vote makes while the
   * step runs. A whole round goes to the server, and the advance waits on it.
   * *excludeId* is the item just voted on, passed over unless it is the only
   * pick still owed a label (a vote that toggled its label off).
   */
  advance(excludeId?: number): void {
    const status = this.status();
    if (status !== 'running') {
      this.owed.set(true);
      // No check under way, the step still on: a pair switch dropped it. Draw a fresh one.
      if (status === 'idle') this.start();
      return;
    }
    const unlabeled = this.unlabeledPicks();
    const next = unlabeled.find((id) => id !== excludeId) ?? unlabeled[0];
    if (next === undefined) {
      this.send();
      return;
    }
    this.owed.set(false);
    this.mediaState.selectMedia(next);
  }

  /** The picks the next votes would land on after *currentId*, for the image prefetch (#3896). */
  upcoming(currentId: number | null, depth: number): number[] {
    return this.unlabeledPicks()
      .filter((id) => id !== currentId)
      .slice(0, depth);
  }

  /**
   * End a check the host is done with (Autopilot stopped, the view left).
   * The picks already voted this round go to the check first, so they carry
   * its provenance; the cancel outlives the view on purpose.
   */
  cancel(): void {
    const status = this.status();
    if (status === 'idle') return;
    const voted = status === 'running' ? this.labeledVotes() : [];
    this.reset();
    const cancel = () => this.sortingApi.cancelPrecisionCheck().subscribe({ error: () => undefined });
    if (voted.length === 0) {
      cancel();
      return;
    }
    this.sortingApi
      .votePrecisionCheck(voted)
      .pipe(finalize(cancel))
      .subscribe({ error: () => undefined });
  }

  /** Forget the check without a word to the server: the pair switched, and its requests were torn down. */
  reset(): void {
    this.inFlight?.unsubscribe();
    this.inFlight = null;
    this.status.set('idle');
    this.picks.set([]);
    this.owed.set(false);
    this.sent.set(false);
  }

  private send(): void {
    this.status.set('sending');
    this.owed.set(true);
    this.sent.set(true);
    this.inFlight = this.sortingApi
      .votePrecisionCheck(this.labeledVotes())
      .pipe(this.pairScope.scoped())
      .subscribe({
        next: (resp) => this.apply(resp),
        error: () => {
          this.sortingApi.cancelPrecisionCheck().subscribe({ error: () => undefined });
          this.end(false);
        },
      });
  }

  /** Take a response: a round to vote on, or the end of the check. */
  private apply(resp: PrecisionCheckResponse): void {
    const c = resp.check;
    if (c?.status === 'running' && c.picks.length > 0) {
      this.picks.set([...c.picks]);
      this.metadataCache.ensureLoaded(c.picks);
      this.status.set('running');
      if (this.owed()) this.advance();
      return;
    }
    this.end(c?.status === 'checked');
  }

  private end(finished: boolean): void {
    const owed = this.owed();
    this.reset();
    this.ended(finished, owed);
  }

  private labelOf(id: number): 'good' | 'bad' | null {
    if (this.voteState.goodVotes.has(id)) return 'good';
    if (this.voteState.badVotes.has(id)) return 'bad';
    return null;
  }

  private unlabeledPicks(): number[] {
    return this.picks().filter((id) => this.labelOf(id) === null);
  }

  private labeledVotes(): PrecisionCheckVote[] {
    const votes: PrecisionCheckVote[] = [];
    for (const id of this.picks()) {
      const label = this.labelOf(id);
      if (label !== null) votes.push({ id, label });
    }
    return votes;
  }
}
