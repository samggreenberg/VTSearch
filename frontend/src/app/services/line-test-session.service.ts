import { Injectable, computed, inject, signal } from '@angular/core';
import { Observable } from 'rxjs';

import type { LineTestResponse } from '../generated/api-client/models/line-test-response';
import type { LineTestState } from '../generated/api-client/models/line-test-state';
import { LineTestApiService } from './line-test-api.service';
import { MediaMetadataCacheService } from './media-metadata-cache.service';
import { PairScopeService } from './pair-scope.service';
import { VoteStateService } from './vote-state.service';
import type { VoteDirection } from './keyboard.service';
import { apiErrorMessage } from '../utils/api-error';
import { lineTestPhase, type LineTestPhase } from '../utils/line-test';

/**
 * One Find session's test of the line (#4524): the Test autopilot's state as
 * the view and its three panes read it. Component-provided on the Find view,
 * beside `PairScopeService`, so it lives exactly as long as the pair does.
 *
 * The server owns the sample and derives the phase on every read
 * (`/api/line-test`, `vtscore.training.thresholds.line_phase`); this holds the
 * last response and the one thing the server does not report: **the round as
 * dealt**, so the stage's dots can show every pick of the round with its vote,
 * where the wire's `picks` are only the ones still pending. A vote goes out as
 * it is cast (the ranges on the right move per vote, as the plan says), a
 * fast voter's next vote going out while the last is still in flight, and the
 * ↓ key sends an unvote. Responses carry the whole state and can land out of
 * order, so each is stamped with the request's sequence and an older one
 * never overwrites a newer (`apply`).
 */
@Injectable()
export class LineTestSessionService {
  private readonly api = inject(LineTestApiService);
  private readonly metadataCache = inject(MediaMetadataCacheService);
  private readonly voteState = inject(VoteStateService);
  private readonly pairScope = inject(PairScopeService);

  /** The server's last word. */
  readonly response = signal<LineTestResponse | null>(null);
  /** The round's picks as dealt, in draw order (random): the stage's dots. */
  readonly roundPicks = signal<number[]>([]);
  /** The round's votes so far, as the server has them. */
  readonly roundVotes = signal<ReadonlyMap<number, VoteDirection>>(new Map());
  /** Which pick of the round is on screen. */
  readonly index = signal(0);
  /** A request is in flight. Votes still go out; the balance and the exits wait. */
  readonly busy = signal(false);
  /** The last request's refusal, for the stage to show; empty when none. */
  readonly error = signal('');

  readonly test = computed<LineTestState | null>(() => this.response()?.test ?? null);
  readonly phase = computed<LineTestPhase>(() => lineTestPhase(this.response()));
  /** A round is on the table: a running test with picks dealt. */
  readonly running = computed(() => {
    const phase = this.phase();
    return (phase === 'matches' || phase === 'misses') && this.roundPicks().length > 0;
  });
  /** The balance is frozen within a phase (#4524): changing it would move the line and the bands under the picks. */
  readonly locked = computed(() => {
    const phase = this.phase();
    return phase === 'matches' || phase === 'misses';
  });
  readonly currentId = computed(() => this.roundPicks()[this.index()] ?? null);
  readonly currentVote = computed(() => {
    const id = this.currentId();
    return id === null ? null : (this.roundVotes().get(id) ?? null);
  });

  private seq = 0;
  private applied = 0;

  /** Forget everything: a pair switch, or leaving the view. */
  clear(): void {
    this.response.set(null);
    this.roundPicks.set([]);
    this.roundVotes.set(new Map());
    this.index.set(0);
    this.busy.set(false);
    this.error.set('');
    this.seq = 0;
    this.applied = 0;
    this.inFlight = 0;
  }

  /** Read the test the server holds (a running one survives a reload of the view). */
  load(): void {
    this.send(this.api.get(), 'Could not read the test.');
  }

  /**
   * Start a test over the Find pass's frozen scores. One that resumed from the
   * picks the detector keeps (#4526) made them session votes again, so the
   * Review tab's piles are re-read.
   */
  start(): void {
    this.send(this.api.start(), 'Could not start the test.', () => {
      if (this.test()?.kept_at) this.voteState.loadVotes();
    });
  }

  /** Vote the pick on screen and move to the next unvoted one. */
  vote(direction: VoteDirection): void {
    const id = this.currentId();
    if (!this.running() || id === null) return;
    const next = new Map(this.roundVotes());
    next.set(id, direction);
    this.roundVotes.set(next);
    const unvoted = this.nextUnvoted(this.index());
    if (unvoted !== null) this.index.set(unvoted);
    this.send(
      this.api.vote([{ id, label: direction }]),
      'Could not send the vote.',
      // The vote is a session vote: the Review tab's piles hold it now.
      () => this.voteState.loadVotes(),
      // Refused: the pick is unvoted again, and back on screen to vote once more.
      () => {
        const back = new Map(this.roundVotes());
        back.delete(id);
        this.roundVotes.set(back);
        const i = this.roundPicks().indexOf(id);
        if (i >= 0) this.index.set(i);
      },
    );
  }

  /** ↓ steps back to the previous pick and lifts its vote; ↑ goes on to the next unvoted one. */
  navigate(direction: 'back' | 'forward'): void {
    if (!this.running()) return;
    if (direction === 'forward') {
      const unvoted = this.nextUnvoted(this.index());
      if (unvoted !== null) this.index.set(unvoted);
      return;
    }
    const i = Math.max(0, this.index() - 1);
    this.index.set(i);
    const id = this.roundPicks()[i];
    if (id === undefined || !this.roundVotes().has(id)) return;
    const was = this.roundVotes().get(id)!;
    const next = new Map(this.roundVotes());
    next.delete(id);
    this.roundVotes.set(next);
    this.send(
      this.api.unvote(id),
      'Could not take the vote back.',
      () => this.voteState.loadVotes(),
      () => {
        const back = new Map(this.roundVotes());
        back.set(id, was);
        this.roundVotes.set(back);
      },
    );
  }

  /** Show pick *i* of the round. */
  show(i: number): void {
    if (i >= 0 && i < this.roundPicks().length) this.index.set(i);
  }

  /** Drop a running test; its votes stay. A finished one is kept. */
  cancel(): void {
    this.send(this.api.cancel(), 'Could not stop the test.');
  }

  /** Forget the test the detector keeps for this dataset (#4526) and deal a fresh one over the same line. */
  testAfresh(): void {
    this.send(this.api.forget(), 'Could not forget the kept test.', () => this.start());
  }

  private inFlight = 0;

  private send(
    request: Observable<LineTestResponse>,
    failure: string,
    then?: () => void,
    refused?: () => void,
  ): void {
    const seq = ++this.seq;
    this.inFlight++;
    this.busy.set(true);
    this.error.set('');
    const settle = () => {
      this.inFlight--;
      if (this.inFlight <= 0) {
        this.inFlight = 0;
        this.busy.set(false);
      }
    };
    request.pipe(this.pairScope.scoped()).subscribe({
      next: (resp) => {
        settle();
        if (seq < this.applied) return;
        this.applied = seq;
        this.apply(resp);
        then?.();
      },
      error: (err) => {
        settle();
        this.error.set(apiErrorMessage(err, failure));
        refused?.();
      },
    });
  }

  /**
   * Take a response. A new round (a higher round number, or picks the dealt
   * round does not hold) replaces the dealt round; otherwise the dealt round
   * stays with the votes as cast. The server's pending picks are not read
   * back into the round's votes: a reply to one vote lands while the next is
   * still in flight, and would erase it; a refused vote is rolled back by its
   * own request instead.
   */
  private apply(resp: LineTestResponse): void {
    const previous = this.test();
    this.response.set(resp);
    const test = resp.test ?? null;
    const pending = test?.picks ?? [];
    if (!test || !pending.length && !this.roundPicks().length) {
      if (!test || test.phase === 'done' || test.phase === 'nothing') {
        this.roundPicks.set([]);
        this.roundVotes.set(new Map());
        this.index.set(0);
      }
      return;
    }
    const dealt = this.roundPicks();
    const fresh = !previous || test.round !== previous.round || pending.some((id) => !dealt.includes(id));
    if (fresh) {
      this.roundPicks.set([...pending]);
      this.roundVotes.set(new Map());
      this.index.set(0);
      this.metadataCache.ensureLoaded(pending);
      return;
    }
    if (test.phase === 'done' || test.phase === 'nothing') {
      this.roundPicks.set([]);
      this.roundVotes.set(new Map());
      this.index.set(0);
    }
  }

  /** The first unvoted pick after *from*, wrapping round; null when every pick has a vote. */
  private nextUnvoted(from: number): number | null {
    const picks = this.roundPicks();
    const votes = this.roundVotes();
    for (let step = 1; step <= picks.length; step++) {
      const i = (from + step) % picks.length;
      if (!votes.has(picks[i])) return i;
    }
    return null;
  }
}
