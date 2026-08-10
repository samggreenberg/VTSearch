import { Injectable, OnDestroy, inject, signal } from '@angular/core';
import { Observable, Subject } from 'rxjs';
import { map, takeUntil, tap } from 'rxjs/operators';
import type { MediaVoteResponse } from '../generated/api-client/models/media-vote-response';
import { VotesResponse } from '../generated/api-client/models/votes-response';
import { adaptivePoll } from './adaptive-poll';
import { MediasApiService } from './medias-api.service';
import { SortingApiService } from './sorting-api.service';

/**
 * One vote captured for Cmd/Ctrl-Z undo.  `previousPolarity` is the polarity
 * the media had *before* the click that produced this entry, so undo can
 * restore it with a single inverse POST to /api/medias/<id>/vote.  It is the
 * *human* polarity (see {@link VoteStateService.currentState}): in Find mode
 * an unverified item's flood-filled good/bad membership is the detector's
 * presumption, not a decision, so it snapshots as `null`.
 */
export interface UndoEntry {
  mediaId: number;
  clickedDirection: 'good' | 'bad';
  previousPolarity: 'good' | 'bad' | null;
  /**
   * Region box the media's good vote carried *before* the click, or ``null``
   * when it had none (was un-voted / bad / a box-less good).  Replayed on
   * undo so restoring back to a good vote also restores its crop instead of
   * silently dropping it (a box-less good POST).
   */
  previousRegionBox: number[] | null;
  /**
   * Region box the click itself applied (the box drawn for the good vote that
   * produced this entry), or ``null`` when the click set bad / un-voted / a
   * box-less good.  Replayed on redo so re-applying a good vote restores its
   * crop.
   */
  clickedRegionBox: number[] | null;
  mediaName: string;
}

export interface UndoToast {
  action: 'undo' | 'redo';
  mediaName: string;
}

type VoteState = 'good' | 'bad' | 'none';

const UNDO_STACK_MAX = 20;

/**
 * Shared vote state. The per-pile sets/maps are backed by signals so a write
 * from any context — a vote POST continuation, the 2s poll timer, an undo/redo
 * — notifies Angular's scheduler and repaints the views that bind these getters
 * with no zone.js.
 * Each value is exposed via a value-returning getter over a private signal, so
 * existing `voteState.goodVotes` reads stay the same yet become reactive under
 * zoneless (see `SortStateService` for the getter-signal rationale).
 *
 * `toast$` stays a `Subject`: it is a fire-once *event* (an undo/redo fired),
 * not retained state, so a signal would be the wrong shape. Its consumer
 * reacts imperatively and writes its own signals, which schedule CD.
 */
@Injectable({ providedIn: 'root' })
export class VoteStateService implements OnDestroy {
  private sortingApi = inject(SortingApiService);
  private mediasApi = inject(MediasApiService);

  private readonly _goodVotes = signal<Set<number>>(new Set());
  private readonly _badVotes = signal<Set<number>>(new Set());
  private readonly _verifiedIds = signal<Set<number>>(new Set());
  private readonly _clickTimes = signal<Record<string, number>>({});
  private readonly _learnedScores = signal<Record<string, number>>({});
  private readonly _labelsetGoodCount = signal(0);
  private readonly _labelsetBadCount = signal(0);
  private readonly _goodRegionBoxes = signal<Record<string, number[]>>({});
  private readonly destroy$ = new Subject<void>();
  private readonly stopPolling$ = new Subject<void>();
  private polling = false;
  /**
   * Monotonic issue-order id for ``/api/votes`` reads, with the highest id whose
   * response has been applied.  ``loadVotes()`` (fired on every vote) and the 2s
   * poll both GET ``/api/votes`` independently, so under rapid "sit and vote"
   * several reads are in flight at once and can resolve **out of order**: a poll
   * GET that read the server *before* a vote committed can land *after* the
   * post-vote ``loadVotes()``, overwriting the fresh state with stale server
   * data — which wiped a just-verified id back out of the right pile until a
   * manual browser refresh (the find-verification staleness report).  We stamp
   * each read with an issue id and drop any response older than the newest one
   * already applied, so a late stale read can no longer clobber newer state.
   */
  private votesSeq = 0;
  private lastAppliedVotesSeq = 0;
  /**
   * Find-mode flag.  In Find the detector flood-fills *every* item into
   * ``goodVotes`` / ``badVotes`` (its presumption at the cutoff), so a raw
   * good/bad membership is **not** a human decision — only ``verifiedIds`` is.
   * When this is set, {@link currentState} treats an *unverified* item as
   * ``'none'`` so the big Good/Bad buttons read neutral and a click *verifies*
   * the item (sets an absolute good/bad) instead of toggling the detector's
   * presumption off.  Set by the Find view; left ``false`` in Label/Train.
   */
  private findMode = false;
  /**
   * Optimistic post-vote state per media, used to preserve the user's click
   * across a polling response that raced ahead of the vote POST.  Cleared
   * deterministically on every vote POST *resolution*: on success the
   * server's reply is the authoritative state ({@link reconcileVoteResponse});
   * on failure the entry is rolled back and server state re-read
   * ({@link rollbackOptimistic}).  Either way there is no "permanently
   * stuck" desync if our prediction disagreed with the server (the
   * persistent-desync half of logical-bug-audit H1).
   */
  private pendingOptimistic = new Map<number, { state: VoteState; clickTime: number | null }>();

  /**
   * Optimistic verified state per media (Find mode): ``true`` = just verified,
   * ``false`` = just un-verified.  Merged over the server's ``verified`` array
   * in {@link applyVotes} and cleared once the server confirms, so a 2s poll
   * that raced ahead of the vote POST doesn't flicker the left/right split.
   */
  private pendingVerified = new Map<number, boolean>();

  /**
   * Optimistic per-media region box for an in-flight good vote (the box drawn
   * on an image), or ``null`` when an in-flight vote cleared the box (un-vote /
   * bad / box-less good).  Merged over the server's ``good_region_boxes`` in
   * {@link applyVotes} so the Good pile crops to the voted region immediately,
   * without waiting for the next /api/votes poll.  Each entry is cleared once
   * the server's response agrees.
   */
  private pendingRegionBoxes = new Map<number, number[] | null>();

  /** Past votes available to undo, most-recent last.  Capped at UNDO_STACK_MAX. */
  private past: UndoEntry[] = [];
  /** Votes that have been undone and can be redone via Cmd/Ctrl-Shift-Z. */
  private future: UndoEntry[] = [];
  private readonly toastSubject = new Subject<UndoToast>();

  /** Emits a short message every time an undo or redo executes. */
  readonly toast$ = this.toastSubject.asObservable();

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
    this.stopPolling$.next();
    this.stopPolling$.complete();
  }

  get goodVotes(): Set<number> {
    return this._goodVotes();
  }

  get badVotes(): Set<number> {
    return this._badVotes();
  }

  /** Find mode: ids the human has explicitly verified (acted on). */
  get verifiedIds(): Set<number> {
    return this._verifiedIds();
  }

  /**
   * Toggle Find-mode verification gating (see {@link findMode}).  The Find
   * view sets this on enter and clears it on leave so Label/Train keep the
   * plain "good membership = voted good" behaviour.
   */
  setFindMode(on: boolean): void {
    this.findMode = on;
  }

  /**
   * Good/bad state for display, honouring Find-mode verification: an
   * unverified Find item reads as neither good nor bad (its membership is the
   * detector's presumption, not a human vote).  Outside Find these mirror raw
   * ``goodVotes`` / ``badVotes`` membership.
   */
  effectiveGood(id: number): boolean {
    return this.currentState(id) === 'good';
  }

  effectiveBad(id: number): boolean {
    return this.currentState(id) === 'bad';
  }

  /**
   * Optimistically record that *id* is now verified (good/bad vote landed in
   * Find mode) or no longer verified (un-voted).  The server marks the same
   * transition on the vote POST; this just makes the left→right move feel
   * instant.  Reconciled by {@link applyVotes} on the next ``/api/votes`` read.
   */
  setOptimisticVerified(id: number, verified: boolean): void {
    this.pendingVerified.set(id, verified);
    const next = new Set(this._verifiedIds());
    if (verified) next.add(id);
    else next.delete(id);
    this._verifiedIds.set(next);
  }

  get clickTimes(): Record<string, number> {
    return this._clickTimes();
  }

  get learnedScores(): Record<string, number> {
    return this._learnedScores();
  }

  /**
   * Number of "good" labels in the active detector's saved labelset (across
   * all datasets the detector has been used with).  Falls back to current
   * dataset's good vote count when no detector is loaded.
   */
  get labelsetGoodCount(): number {
    return this._labelsetGoodCount();
  }

  /** Bad-label counterpart of {@link labelsetGoodCount}. */
  get labelsetBadCount(): number {
    return this._labelsetBadCount();
  }

  get goodRegionBoxes(): Record<string, number[]> {
    return this._goodRegionBoxes();
  }

  /**
   * True when the active detector has at least one good and one bad label
   * available for training (i.e. `/api/learned-sort` would succeed).
   */
  get learnedSortAvailable(): boolean {
    return this.labelsetGoodCount > 0 && this.labelsetBadCount > 0;
  }

  /**
   * Current local polarity for *id*, or ``'none'`` when not voted.
   *
   * In Find mode an *unverified* item reads as ``'none'`` regardless of its
   * flood-filled good/bad membership: that membership is the detector's
   * presumption, not a human decision (see {@link findMode}).  This is what
   * makes the big Good/Bad buttons verify (set an absolute good/bad) on an
   * unverified item rather than toggle the presumption off.
   */
  private currentState(id: number): VoteState {
    if (this.findMode && !this._verifiedIds().has(id)) return 'none';
    if (this._goodVotes().has(id)) return 'good';
    if (this._badVotes().has(id)) return 'bad';
    return 'none';
  }

  /**
   * Translate a clicked direction into an absolute target by the local
   * toggle rule (clicking the current polarity un-votes; anything else sets
   * to the clicked polarity).  Pure; does not mutate local state.
   */
  toggleTargetFor(id: number, clickedDirection: 'good' | 'bad'): VoteState {
    return this.currentState(id) === clickedDirection ? 'none' : clickedDirection;
  }

  /**
   * Snapshot the polarity and good-vote crop *id* carries before a click, for
   * the undo stack.  Must be read before the optimistic flip overwrites them.
   *
   * The polarity comes from the find-aware {@link currentState}, **not** raw
   * ``goodVotes`` / ``badVotes`` membership: in Find every item is flood-filled
   * with the detector's presumption, so an unverified item's membership is not
   * a human decision.  Snapshotting it as one is what made undo POST the
   * presumption back as an explicit vote — which the server marks *verified*,
   * silently writing a label the human never gave — and inverted redo whenever
   * the clicked direction happened to match the presumption (redo computed
   * ``'none'`` and un-verified the item the user had just re-verified).
   * Outside Find this is identical to the raw membership it replaced.
   */
  private capturePrevious(id: number): {
    previousPolarity: 'good' | 'bad' | null;
    previousRegionBox: number[] | null;
  } {
    const state = this.currentState(id);
    const previousPolarity = state === 'none' ? null : state;
    const prevBox = this._goodRegionBoxes()[String(id)];
    return {
      previousPolarity,
      previousRegionBox: previousPolarity === 'good' && prevBox ? [...prevBox] : null,
    };
  }

  /**
   * Submit a click on a media with the local-view toggle rule.
   *
   * Computes the target from current local state, optimistically applies
   * it, POSTs ``target`` to ``/api/medias/<id>/vote``, and reconciles the
   * local view from the server's response when it arrives.  All callers in
   * the components (centre / right / find / label views) funnel through
   * here so the toggle-to-target conversion lives in exactly one place.
   *
   * @param regionBox  Optional good-vote region.  Honoured only when the
   *                   computed target is ``'good'``.
   */
  submitToggleVote(
    id: number,
    clickedDirection: 'good' | 'bad',
    regionBox?: readonly number[] | null,
  ): Observable<MediaVoteResponse> {
    const target = this.toggleTargetFor(id, clickedDirection);
    this.applyOptimisticState(id, target);
    const effectiveBox = target === 'good' && regionBox && regionBox.length === 4 ? regionBox : null;
    this.applyOptimisticRegionBox(id, effectiveBox);
    return this.mediasApi.vote(id, target, effectiveBox).pipe(
      tap({
        next: (resp) => this.reconcileVoteResponse(id, resp),
        // A failed POST means the server never saw the vote. Without the
        // rollback, the pending entries pin the optimistic state over every
        // subsequent /api/votes poll forever: the item shows as voted while
        // the server (and training) never has it.
        error: () => this.rollbackOptimistic(id),
      }),
    );
  }

  /**
   * Discard every pending optimistic entry for *id* and re-read the server's
   * vote state.  Called when a vote POST fails: the prediction never became
   * server truth, so leaving the entries in place would have
   * {@link applyVotes} re-impose the phantom vote on every poll.
   */
  private rollbackOptimistic(id: number): void {
    this.pendingOptimistic.delete(id);
    this.pendingVerified.delete(id);
    this.pendingRegionBoxes.delete(id);
    this.loadVotes();
  }

  /**
   * Like {@link submitToggleVote}, but also records an undo entry on the
   * past stack, **only after the server confirms the vote**.  Production
   * callers (centre / find / label views) should use this rather than
   * pairing {@link submitToggleVote} with a separate {@link recordVote},
   * because the latter ordering racks an undo entry that may not be
   * reflected on the server (audit bug H26): a failed POST left a phantom
   * entry on the stack, and a subsequent Cmd-Z then issued a "reversal" of
   * a vote that never happened.
   *
   * `previousPolarity` is captured here, synchronously, **before** the
   * optimistic flip; that snapshot is the only piece of state that has to
   * exist pre-POST.  The undo entry itself is only pushed if the POST
   * resolves successfully.
   */
  submitToggleVoteAndRecord(
    id: number,
    clickedDirection: 'good' | 'bad',
    mediaName: string,
    regionBox?: readonly number[] | null,
  ): Observable<MediaVoteResponse> {
    const { previousPolarity, previousRegionBox } = this.capturePrevious(id);
    const target = this.toggleTargetFor(id, clickedDirection);
    const clickedRegionBox =
      target === 'good' && regionBox && regionBox.length === 4 ? [...regionBox] : null;
    return this.submitToggleVote(id, clickedDirection, regionBox).pipe(
      tap(() => {
        this.past.push({
          mediaId: id,
          clickedDirection,
          previousPolarity,
          previousRegionBox,
          clickedRegionBox,
          mediaName,
        });
        if (this.past.length > UNDO_STACK_MAX) this.past.shift();
        this.future = [];
      }),
    );
  }

  /**
   * Optimistically apply an absolute target state without going through the
   * toggle rule.  Used by {@link submitToggleVote} (above) and by the
   * Cmd-Z undo / redo flow, where the desired post-call state is known
   * directly (a polarity to restore, or ``'none'`` to wipe the vote).
   */
  applyOptimisticState(id: number, target: VoteState): void {
    const good = new Set(this._goodVotes());
    const bad = new Set(this._badVotes());
    const times = { ...this._clickTimes() };

    good.delete(id);
    bad.delete(id);

    let optimisticClickTime: number | null = null;
    if (target === 'good') {
      good.add(id);
    } else if (target === 'bad') {
      bad.add(id);
    }
    if (target !== 'none') {
      // Set an optimistic click time so the item sorts correctly immediately,
      // rather than appearing with time=-1 and then jumping when the server responds.
      const maxTime = Object.values(times).reduce((m, t) => Math.max(m, t), 0);
      optimisticClickTime = maxTime + 1;
      times[String(id)] = optimisticClickTime;
    } else {
      delete times[String(id)];
    }

    this.pendingOptimistic.set(id, { state: target, clickTime: optimisticClickTime });

    // Emit all changes together so Angular sees a single consistent state.
    this._goodVotes.set(good);
    this._badVotes.set(bad);
    this._clickTimes.set(times);
  }

  /**
   * Optimistically record the region box for an in-flight good vote (or
   * ``null`` to clear it).  Mirrors {@link applyOptimisticState} for region
   * crops: the Good-pile thumbnail crops to the box immediately, and the entry
   * is reconciled against the server's ``good_region_boxes`` on the next poll.
   */
  private applyOptimisticRegionBox(id: number, box: readonly number[] | null): void {
    this.pendingRegionBoxes.set(id, box ? [...box] : null);
    const boxes = { ...this._goodRegionBoxes() };
    if (box) boxes[String(id)] = [...box];
    else delete boxes[String(id)];
    this._goodRegionBoxes.set(boxes);
  }

  /**
   * Apply the absolute state the server confirmed in its vote response,
   * regardless of what the optimistic prediction was.  Pending optimism is
   * cleared deterministically here; even if our prediction was wrong, the
   * server's reply replaces it so the desync cannot persist.
   */
  reconcileVoteResponse(id: number, resp: MediaVoteResponse): void {
    this.pendingOptimistic.delete(id);
    const good = new Set(this._goodVotes());
    const bad = new Set(this._badVotes());
    const times = { ...this._clickTimes() };

    good.delete(id);
    bad.delete(id);
    if (resp.state === 'good') good.add(id);
    else if (resp.state === 'bad') bad.add(id);

    if (resp.click_time != null) {
      times[String(id)] = resp.click_time;
    } else {
      delete times[String(id)];
    }

    this._goodVotes.set(good);
    this._badVotes.set(bad);
    this._clickTimes.set(times);
  }

  loadVotes(): void {
    const seq = ++this.votesSeq;
    this.sortingApi
      .getVotes()
      .pipe(takeUntil(this.destroy$))
      .subscribe((votes) => this.applyVotesFresh(votes, seq));
  }

  startPolling(intervalMs = 2000): void {
    if (this.polling) return;
    this.polling = true;
    // adaptivePoll never overlaps GETs (a backend slower than the interval no
    // longer has every /api/votes read cancelled by the next tick and freezing
    // the piles), eases to a heartbeat when the votes stop changing, and pauses
    // while the tab is hidden. Errors are absorbed inside adaptivePoll, so a
    // transient failure (502, offline blip, stale X-Dataset-Id after a context
    // switch) skips a single poll without applying anything — applyVotesFresh
    // is never called with a stub, so optimistic state survives a failed tick.
    // The seq is excluded from the change signature so its per-poll increment
    // doesn't defeat the back-off.
    adaptivePoll(
      () => {
        const seq = ++this.votesSeq;
        return this.sortingApi.getVotes().pipe(map((votes) => ({ votes, seq })));
      },
      {
        fastMs: intervalMs,
        slowMs: Math.max(intervalMs, 10000),
        signature: ({ votes }) => JSON.stringify(votes),
      },
    )
      .pipe(takeUntil(this.stopPolling$), takeUntil(this.destroy$))
      .subscribe(({ votes, seq }) => this.applyVotesFresh(votes, seq));
  }

  /**
   * Apply a ``/api/votes`` response only when it is not older than the newest
   * read already applied.  Out-of-order resolutions (a poll GET that read the
   * server before a vote committed, landing after the post-vote loadVotes) are
   * dropped so they cannot overwrite newer state — see {@link votesSeq}.
   */
  private applyVotesFresh(votes: VotesResponse, seq: number): void {
    if (seq < this.lastAppliedVotesSeq) return;
    this.lastAppliedVotesSeq = seq;
    this.applyVotes(votes);
  }

  stopPolling(): void {
    this.stopPolling$.next();
    this.polling = false;
  }

  clear(): void {
    this._goodVotes.set(new Set());
    this._badVotes.set(new Set());
    this._verifiedIds.set(new Set());
    this._clickTimes.set({});
    this._learnedScores.set({});
    this._labelsetGoodCount.set(0);
    this._labelsetBadCount.set(0);
    this._goodRegionBoxes.set({});
    this.pendingOptimistic.clear();
    this.pendingVerified.clear();
    this.pendingRegionBoxes.clear();
    this.past = [];
    this.future = [];
    // Discard any /api/votes read issued before this clear (e.g. an in-flight
    // poll from the previous dataset/detector context): treat everything up to
    // the current issue id as already-superseded so a late pre-clear response
    // can't repopulate the just-cleared piles.
    this.lastAppliedVotesSeq = this.votesSeq;
  }

  /**
   * Snapshot the polarity *before* a vote click and push it onto the undo
   * stack.  Low-level primitive: pushes unconditionally, with no link to a
   * POST result.  Production code should call {@link submitToggleVoteAndRecord}
   * instead, which only records the entry once the server confirms the vote
   * (audit bug H26).
   *
   * Must be called BEFORE the corresponding optimistic flip (otherwise the
   * snapshot would already reflect the toggle).  Any pending redo entries
   * are dropped, matching standard editor undo semantics.
   */
  recordVote(mediaId: number, clickedDirection: 'good' | 'bad', mediaName: string): void {
    const { previousPolarity, previousRegionBox } = this.capturePrevious(mediaId);
    this.past.push({
      mediaId,
      clickedDirection,
      previousPolarity,
      previousRegionBox,
      // This low-level primitive isn't told the click's region box; redo of a
      // good vote recorded this way restores no crop.  Production callers use
      // submitToggleVoteAndRecord, which captures it.
      clickedRegionBox: null,
      mediaName,
    });
    if (this.past.length > UNDO_STACK_MAX) this.past.shift();
    this.future = [];
  }

  /**
   * Mirror the verified transition an undo/redo POST triggers server-side, so
   * the Find left/right split moves with the keystroke instead of lagging a
   * poll behind.  A vote/redo to good|bad verifies the item; a target of
   * ``'none'`` un-verifies it (``_mark_verified_if_find_mode``, votes.py).
   * A no-op outside Find, where nothing is verification-gated.
   */
  private mirrorVerified(id: number, target: VoteState): void {
    if (!this.findMode) return;
    this.setOptimisticVerified(id, target !== 'none');
  }

  canUndo(): boolean {
    return this.past.length > 0;
  }

  canRedo(): boolean {
    return this.future.length > 0;
  }

  /**
   * Reverse the most recent vote.  The inverse target is:
   *   - the saved {@code previousPolarity} (restores prior state, including
   *     polarity flips), or
   *   - ``'none'`` when previousPolarity was null (un-votes).
   *
   * Side effects that aren't reversible (achievements, label_history append,
   * click_counter monotonicity) are accepted; the user really did make the
   * click; we just put the item back where it was.
   */
  undo(): void {
    const entry = this.past.pop();
    if (!entry) return;
    this.future.push(entry);
    const target: VoteState = entry.previousPolarity ?? 'none';
    const box = target === 'good' ? entry.previousRegionBox : null;
    this.applyOptimisticState(entry.mediaId, target);
    this.applyOptimisticRegionBox(entry.mediaId, box);
    this.mirrorVerified(entry.mediaId, target);
    this.mediasApi.vote(entry.mediaId, target, box).subscribe({
      next: (resp) => this.reconcileVoteResponse(entry.mediaId, resp),
      // rollbackOptimistic (not a bare loadVotes): the pending entries set
      // by the optimistic apply above would override the reloaded server
      // state in applyVotes, so they must be dropped first.
      error: () => this.rollbackOptimistic(entry.mediaId),
    });
    this.toastSubject.next({ action: 'undo', mediaName: entry.mediaName });
  }

  /** Re-apply the most recently undone vote: POST the toggled-from-prior target. */
  redo(): void {
    const entry = this.future.pop();
    if (!entry) return;
    this.past.push(entry);
    if (this.past.length > UNDO_STACK_MAX) this.past.shift();
    // The redo target is whichever polarity the original click landed on
    // (clickedDirection unless previousPolarity matched, in which case the
    // click was an un-vote).
    const target: VoteState =
      entry.previousPolarity === entry.clickedDirection ? 'none' : entry.clickedDirection;
    const box = target === 'good' ? entry.clickedRegionBox : null;
    this.applyOptimisticState(entry.mediaId, target);
    this.applyOptimisticRegionBox(entry.mediaId, box);
    this.mirrorVerified(entry.mediaId, target);
    this.mediasApi.vote(entry.mediaId, target, box).subscribe({
      next: (resp) => this.reconcileVoteResponse(entry.mediaId, resp),
      // See undo(): drop the pending entries before reloading server state.
      error: () => this.rollbackOptimistic(entry.mediaId),
    });
    this.toastSubject.next({ action: 'redo', mediaName: entry.mediaName });
  }

  /** True when two region boxes match coordinate-for-coordinate (both 4-tuples). */
  private boxesEqual(a: number[] | null, b: number[] | null): boolean {
    if (a === null || b === null) return a === b;
    return a.length === b.length && a.every((v, i) => v === b[i]);
  }

  private applyVotes(votes: VotesResponse): void {
    const good = new Set(votes.good);
    const bad = new Set(votes.bad);
    const times = { ...votes.click_times };

    // Preserve optimistic votes whose POSTs haven't returned yet.  Once a
    // POST resolves, reconcileVoteResponse() clears the pending entry and
    // applies the server's authoritative state; so any preserved entry
    // here is genuinely in-flight, not a permanent prediction-vs-server
    // desync (the H1 stuck-prediction case is gone).
    for (const [id, opt] of this.pendingOptimistic) {
      good.delete(id);
      bad.delete(id);
      if (opt.state === 'good') good.add(id);
      else if (opt.state === 'bad') bad.add(id);
      if (opt.clickTime != null) {
        times[String(id)] = opt.clickTime;
      } else {
        delete times[String(id)];
      }
    }

    // Verified ids: start from the server set, then apply pending optimistic
    // overrides (clearing each once the server already agrees) so an in-flight
    // verify/un-verify isn't clobbered by a poll that raced the vote POST.
    const verified = new Set(votes.verified ?? []);
    for (const [id, want] of this.pendingVerified) {
      if (verified.has(id) === want) {
        this.pendingVerified.delete(id);
      } else if (want) {
        verified.add(id);
      } else {
        verified.delete(id);
      }
    }

    // Region boxes: start from the server map, then apply pending optimistic
    // overrides (clearing each once the server already agrees) so an in-flight
    // region vote isn't clobbered by a poll that raced the vote POST.
    const regionBoxes: Record<string, number[]> = { ...(votes.good_region_boxes ?? {}) };
    for (const [id, want] of this.pendingRegionBoxes) {
      const key = String(id);
      const server = regionBoxes[key] ?? null;
      const agrees = want === null ? server === null : this.boxesEqual(server, want);
      if (agrees) {
        this.pendingRegionBoxes.delete(id);
      } else if (want) {
        regionBoxes[key] = [...want];
      } else {
        delete regionBoxes[key];
      }
    }

    this._goodVotes.set(good);
    this._badVotes.set(bad);
    this._verifiedIds.set(verified);
    this._clickTimes.set(times);
    this._goodRegionBoxes.set(regionBoxes);
    this._learnedScores.set(votes.learned_scores);
    this._labelsetGoodCount.set(votes.labelset_good_count ?? good.size);
    this._labelsetBadCount.set(votes.labelset_bad_count ?? bad.size);
  }
}
