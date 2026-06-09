import { Injectable, OnDestroy } from '@angular/core';
import { BehaviorSubject, EMPTY, Observable, Subject, timer } from 'rxjs';
import { catchError, switchMap, takeUntil, tap } from 'rxjs/operators';
import type { MediaVoteResponse } from '../generated/api-client/models/media-vote-response';
import { VotesResponse } from '../models/api.models';
import { MediasApiService } from './medias-api.service';
import { SortingApiService } from './sorting-api.service';

/**
 * One vote captured for Cmd/Ctrl-Z undo.  `previousPolarity` is the polarity
 * the media had *before* the click that produced this entry, so undo can
 * restore it with a single inverse POST to /api/medias/<id>/vote.
 */
export interface UndoEntry {
  mediaId: number;
  clickedDirection: 'good' | 'bad';
  previousPolarity: 'good' | 'bad' | null;
  mediaName: string;
}

export interface UndoToast {
  action: 'undo' | 'redo';
  mediaName: string;
}

type VoteState = 'good' | 'bad' | 'none';

const UNDO_STACK_MAX = 20;

@Injectable({ providedIn: 'root' })
export class VoteStateService implements OnDestroy {
  private readonly goodVotesSubject = new BehaviorSubject<Set<number>>(new Set());
  private readonly badVotesSubject = new BehaviorSubject<Set<number>>(new Set());
  private readonly verifiedIdsSubject = new BehaviorSubject<Set<number>>(new Set());
  private readonly clickTimesSubject = new BehaviorSubject<Record<string, number>>({});
  private readonly learnedScoresSubject = new BehaviorSubject<Record<string, number>>({});
  private readonly labelsetGoodCountSubject = new BehaviorSubject<number>(0);
  private readonly labelsetBadCountSubject = new BehaviorSubject<number>(0);
  private readonly destroy$ = new Subject<void>();
  private readonly stopPolling$ = new Subject<void>();
  private polling = false;
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
   * deterministically on every vote POST response; the server's reply IS
   * the authoritative state, so there is no longer a "permanently stuck"
   * desync if our prediction disagreed with the server (the persistent-desync
   * half of logical-bug-audit H1).
   */
  private pendingOptimistic = new Map<number, { state: VoteState; clickTime: number | null }>();

  /**
   * Optimistic verified state per media (Find mode): ``true`` = just verified,
   * ``false`` = just un-verified.  Merged over the server's ``verified`` array
   * in {@link applyVotes} and cleared once the server confirms, so a 2s poll
   * that raced ahead of the vote POST doesn't flicker the left/right split.
   */
  private pendingVerified = new Map<number, boolean>();

  /** Past votes available to undo, most-recent last.  Capped at UNDO_STACK_MAX. */
  private past: UndoEntry[] = [];
  /** Votes that have been undone and can be redone via Cmd/Ctrl-Shift-Z. */
  private future: UndoEntry[] = [];
  private readonly toastSubject = new Subject<UndoToast>();

  readonly goodVotes$ = this.goodVotesSubject.asObservable();
  readonly badVotes$ = this.badVotesSubject.asObservable();
  /** Find mode: ids the human has explicitly verified (acted on). */
  readonly verifiedIds$ = this.verifiedIdsSubject.asObservable();
  readonly clickTimes$ = this.clickTimesSubject.asObservable();
  readonly learnedScores$ = this.learnedScoresSubject.asObservable();
  readonly labelsetGoodCount$ = this.labelsetGoodCountSubject.asObservable();
  readonly labelsetBadCount$ = this.labelsetBadCountSubject.asObservable();
  /** Emits a short message every time an undo or redo executes. */
  readonly toast$ = this.toastSubject.asObservable();

  constructor(
    private sortingApi: SortingApiService,
    private mediasApi: MediasApiService,
  ) {}

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
    this.stopPolling$.next();
    this.stopPolling$.complete();
  }

  get goodVotes(): Set<number> {
    return this.goodVotesSubject.value;
  }

  get badVotes(): Set<number> {
    return this.badVotesSubject.value;
  }

  get verifiedIds(): Set<number> {
    return this.verifiedIdsSubject.value;
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
    const next = new Set(this.verifiedIdsSubject.value);
    if (verified) next.add(id);
    else next.delete(id);
    this.verifiedIdsSubject.next(next);
  }

  get clickTimes(): Record<string, number> {
    return this.clickTimesSubject.value;
  }

  get learnedScores(): Record<string, number> {
    return this.learnedScoresSubject.value;
  }

  /**
   * Number of "good" labels in the active detector's saved labelset (across
   * all datasets the detector has been used with).  Falls back to current
   * dataset's good vote count when no detector is loaded.
   */
  get labelsetGoodCount(): number {
    return this.labelsetGoodCountSubject.value;
  }

  /** Bad-label counterpart of {@link labelsetGoodCount}. */
  get labelsetBadCount(): number {
    return this.labelsetBadCountSubject.value;
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
    if (this.findMode && !this.verifiedIdsSubject.value.has(id)) return 'none';
    if (this.goodVotesSubject.value.has(id)) return 'good';
    if (this.badVotesSubject.value.has(id)) return 'bad';
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
    return this.mediasApi.vote(id, target, target === 'good' ? regionBox : null).pipe(
      tap((resp) => this.reconcileVoteResponse(id, resp)),
    );
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
    const previousPolarity: 'good' | 'bad' | null = this.goodVotesSubject.value.has(id)
      ? 'good'
      : this.badVotesSubject.value.has(id)
        ? 'bad'
        : null;
    return this.submitToggleVote(id, clickedDirection, regionBox).pipe(
      tap(() => {
        this.past.push({ mediaId: id, clickedDirection, previousPolarity, mediaName });
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
    const good = new Set(this.goodVotesSubject.value);
    const bad = new Set(this.badVotesSubject.value);
    const times = { ...this.clickTimesSubject.value };

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
    this.goodVotesSubject.next(good);
    this.badVotesSubject.next(bad);
    this.clickTimesSubject.next(times);
  }

  /**
   * Apply the absolute state the server confirmed in its vote response,
   * regardless of what the optimistic prediction was.  Pending optimism is
   * cleared deterministically here; even if our prediction was wrong, the
   * server's reply replaces it so the desync cannot persist.
   */
  reconcileVoteResponse(id: number, resp: MediaVoteResponse): void {
    this.pendingOptimistic.delete(id);
    const good = new Set(this.goodVotesSubject.value);
    const bad = new Set(this.badVotesSubject.value);
    const times = { ...this.clickTimesSubject.value };

    good.delete(id);
    bad.delete(id);
    if (resp.state === 'good') good.add(id);
    else if (resp.state === 'bad') bad.add(id);

    if (resp.click_time != null) {
      times[String(id)] = resp.click_time;
    } else {
      delete times[String(id)];
    }

    this.goodVotesSubject.next(good);
    this.badVotesSubject.next(bad);
    this.clickTimesSubject.next(times);
  }

  loadVotes(): void {
    this.sortingApi
      .getVotes()
      .pipe(takeUntil(this.destroy$))
      .subscribe((votes) => this.applyVotes(votes));
  }

  startPolling(intervalMs = 2000): void {
    if (this.polling) return;
    this.polling = true;
    timer(0, intervalMs)
      .pipe(
        takeUntil(this.stopPolling$),
        takeUntil(this.destroy$),
        // catchError inside switchMap scopes errors to a single tick; a
        // transient /api/votes failure (502, offline blip, stale
        // X-Dataset-Id after a context switch) would otherwise tear the
        // whole chain down, freeze votes indefinitely, and leave `polling`
        // stuck at true so startPolling() can't re-arm. Emit EMPTY rather
        // than a stub VotesResponse so applyVotes() doesn't clobber
        // optimistic state on a failed tick.
        switchMap(() => this.sortingApi.getVotes().pipe(catchError(() => EMPTY))),
      )
      .subscribe((votes) => this.applyVotes(votes));
  }

  stopPolling(): void {
    this.stopPolling$.next();
    this.polling = false;
  }

  clear(): void {
    this.goodVotesSubject.next(new Set());
    this.badVotesSubject.next(new Set());
    this.verifiedIdsSubject.next(new Set());
    this.clickTimesSubject.next({});
    this.learnedScoresSubject.next({});
    this.labelsetGoodCountSubject.next(0);
    this.labelsetBadCountSubject.next(0);
    this.pendingOptimistic.clear();
    this.pendingVerified.clear();
    this.past = [];
    this.future = [];
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
    const previousPolarity: 'good' | 'bad' | null = this.goodVotesSubject.value.has(mediaId)
      ? 'good'
      : this.badVotesSubject.value.has(mediaId)
        ? 'bad'
        : null;
    this.past.push({ mediaId, clickedDirection, previousPolarity, mediaName });
    if (this.past.length > UNDO_STACK_MAX) this.past.shift();
    this.future = [];
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
    this.applyOptimisticState(entry.mediaId, target);
    this.mediasApi.vote(entry.mediaId, target).subscribe({
      next: (resp) => this.reconcileVoteResponse(entry.mediaId, resp),
      error: () => this.loadVotes(),
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
    this.applyOptimisticState(entry.mediaId, target);
    this.mediasApi.vote(entry.mediaId, target).subscribe({
      next: (resp) => this.reconcileVoteResponse(entry.mediaId, resp),
      error: () => this.loadVotes(),
    });
    this.toastSubject.next({ action: 'redo', mediaName: entry.mediaName });
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

    this.goodVotesSubject.next(good);
    this.badVotesSubject.next(bad);
    this.verifiedIdsSubject.next(verified);
    this.clickTimesSubject.next(times);
    this.learnedScoresSubject.next(votes.learned_scores);
    this.labelsetGoodCountSubject.next(votes.labelset_good_count ?? good.size);
    this.labelsetBadCountSubject.next(votes.labelset_bad_count ?? bad.size);
  }
}
