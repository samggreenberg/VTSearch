import { Injectable } from '@angular/core';

/** How many recently-voted ids the trail keeps.  Walking back further than
 *  this is what the Good/Bad piles in the right panel are for. */
const TRAIL_MAX = 50;

/**
 * The trail of items the user has voted on, in the order they voted on them.
 *
 * This is what the ``↓`` shortcut walks (#4032): "wait, go back to the one I
 * was just at", either to change the vote or just to look again.  ``↑`` is the
 * way forward, but it is *not* the inverse walk — it re-enters the queue at
 * whatever the host's advance rule picks next, which is what the user means by
 * "never mind, back to work".  So only the backward direction needs history.
 *
 * Deliberately separate from {@link VoteStateService}'s undo stack even though
 * both are fed by the same clicks.  The undo stack is a stack of *reversible
 * actions*: an undo pops an entry off it, because that vote no longer stands.
 * The trail is a record of *where the user has been*, and an undone item is
 * still somewhere they were — indeed it is exactly where they are standing
 * when they undo.  Sharing one structure would make each operation corrupt the
 * other's meaning.
 *
 * Root-provided so both Train and Find walk the same trail, and reset by
 * ``VoteStateService.clear()`` on a dataset/detector switch — ids from the
 * previous pair name nothing selectable in the new one.
 */
@Injectable({ providedIn: 'root' })
export class VoteHistoryService {
  /** Voted ids, oldest first, each appearing once (a re-vote moves the id to
   *  the newest end rather than duplicating it). */
  private trail: number[] = [];

  /**
   * Index in {@link trail} that the last {@link stepBack} landed on, or
   * ``null`` when no walk is in progress.  A walk is only continued while the
   * user is still standing where it left them; see {@link stepBack}.
   */
  private cursor: number | null = null;

  /** Note that the user just voted on (or un-voted) *mediaId*. */
  record(mediaId: number): void {
    const at = this.trail.indexOf(mediaId);
    if (at !== -1) this.trail.splice(at, 1);
    this.trail.push(mediaId);
    if (this.trail.length > TRAIL_MAX) this.trail.shift();
    // A fresh vote is a new "here": the next ↓ starts from the newest end
    // again rather than resuming the walk that led to this item.
    this.cursor = null;
  }

  /** Drop the trail (dataset/detector switch). */
  clear(): void {
    this.trail = [];
    this.cursor = null;
  }

  /** The trail, oldest first.  Read-only; for tests and debugging. */
  get recentlyVoted(): readonly number[] {
    return this.trail;
  }

  /**
   * One step back along the trail, or ``null`` when there is nowhere to go.
   *
   * *selectedId* is where the user is standing now.  Unless that is exactly
   * where the previous ``↓`` put them, the walk restarts from the newest
   * entry: any other move — advancing off a vote, clicking an item in a pile,
   * a re-sort selecting for them — means the walk they were on is over, and
   * resuming it from wherever it had got to would jump somewhere they have no
   * reason to expect.
   */
  stepBack(selectedId: number | null): number | null {
    if (this.cursor === null || this.trail[this.cursor] !== selectedId) {
      this.cursor = this.trail.length;
    }
    const next = this.cursor - 1;
    if (next < 0) return null;
    this.cursor = next;
    return this.trail[next];
  }
}
