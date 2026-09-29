import { Injectable } from '@angular/core';

/** How many recently-voted ids the trail keeps.  Walking back further than
 *  this is what the Good/Bad piles in the right panel are for. */
const TRAIL_MAX = 50;

/**
 * The trail of items the user has voted on, in the order they voted on them.
 *
 * This is what the ``↓`` shortcut walks (#4032): "wait, go back to the one I
 * was just at", either to change the vote or just to look again.  ``↑`` is the
 * way forward, but it is *not* the inverse walk — one press ends the walk,
 * which is what the user means by "never mind, back to work".
 *
 * Back to work means back to the item they left (#4306), not to whatever the
 * host's advance rule would pick now.  The two often differ.  In Train a vote
 * advances off the ranking it was cast against, and the learned re-sort that
 * vote schedules lands a moment later without moving the selection, so the
 * item on screen need not be the new ranking's pick; in Find every advance
 * flips which side of the cutoff it serves.  Re-running the advance on ``↑``
 * would swap the item for one the user has never seen, so the walk remembers
 * where it started ({@link origin}) and ``↑`` returns there.
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

  /**
   * Where the user was standing when the walk in progress began, or ``null``
   * when no walk is in progress or it began with nothing on screen.  This is
   * what {@link stepForward} returns them to.
   */
  private origin: number | null = null;

  /** Note that the user just voted on (or un-voted) *mediaId*. */
  record(mediaId: number): void {
    const at = this.trail.indexOf(mediaId);
    if (at !== -1) this.trail.splice(at, 1);
    this.trail.push(mediaId);
    if (this.trail.length > TRAIL_MAX) this.trail.shift();
    // A fresh vote is a new "here": the next ↓ starts from the newest end
    // again rather than resuming the walk that led to this item, and ↑ takes
    // the host's advance, since the vote changed what it should be.
    this.endWalk();
  }

  /** Drop the trail (dataset/detector switch). */
  clear(): void {
    this.trail = [];
    this.endWalk();
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
   *
   * A walk that (re)starts remembers *selectedId* as its origin, for
   * {@link stepForward}.  Pass ``null`` when nothing is on screen (a "nothing
   * left" pane standing in for the viewer): that is where ``↑`` should return
   * the user, even though the selection still names the last item shown.
   */
  stepBack(selectedId: number | null): number | null {
    let cursor = this.walking(selectedId) ? this.cursor : null;
    if (cursor === null) {
      if (this.trail.length === 0) return null;
      cursor = this.trail.length;
      this.origin = selectedId;
    }
    // Nothing older: stay put rather than wrap, keeping the walk (and its
    // origin) alive for the ↑ that follows.
    if (cursor === 0) return null;
    this.cursor = cursor - 1;
    return this.trail[this.cursor];
  }

  /**
   * End the walk, returning the item it started from — or ``null`` when the
   * host should take its own forward step instead.
   *
   * That is the case whenever the user is no longer standing where the last
   * {@link stepBack} left them (the same test ``stepBack`` makes, so the two
   * agree on whether a walk is in progress), when no walk was ever begun, and
   * when the walk began with nothing on screen.
   */
  stepForward(selectedId: number | null): number | null {
    const back = this.walking(selectedId) ? this.origin : null;
    this.endWalk();
    return back;
  }

  /** Whether *selectedId* is exactly where the previous {@link stepBack} put
   *  the user, i.e. a walk is in progress and nothing has moved them off it. */
  private walking(selectedId: number | null): boolean {
    return this.cursor !== null && this.trail[this.cursor] === selectedId;
  }

  private endWalk(): void {
    this.cursor = null;
    this.origin = null;
  }
}
