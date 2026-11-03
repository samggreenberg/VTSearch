import { Injectable, computed, inject, signal } from '@angular/core';
import { Observable } from 'rxjs';

import type { LineTestResponse } from '../generated/api-client/models/line-test-response';
import type { LineTestState } from '../generated/api-client/models/line-test-state';
import { LineTestApiService } from './line-test-api.service';
import { MediaMetadataCacheService } from './media-metadata-cache.service';
import { PairScopeService } from './pair-scope.service';
import type { NavDirection, VoteDirection } from './keyboard.service';
import { apiErrorMessage } from '../utils/api-error';
import { lineTestPhase, type LineTestPhase } from '../utils/line-test';

/**
 * One Find session's test of the line (#4524): the Tune autopilot's state as
 * the view and its three panes read it. Component-provided on the Find view,
 * beside `PairScopeService`, so it lives exactly as long as the pair does.
 *
 * The server owns the sample and derives the phase on every read
 * (`/api/line-test`, `vtscore.training.thresholds.line_phase`); this holds the
 * last response, **the round as dealt** (the order its picks come on screen,
 * where the wire's `picks` are only the ones still pending), and every vote
 * of the test, which the centre panel lights its buttons from. A vote goes
 * out as it is cast (the ranges on the right move per vote), a fast voter's
 * next vote going out while the last is still in flight. Responses carry the
 * whole state and can land out of order, so each is stamped with the
 * request's sequence and an older one never overwrites a newer (`apply`).
 *
 * The picks are voted as every other vote in the app is (#5502): the centre
 * panel shows them one at a time, and ↓ walks back along the picks already
 * voted, across rounds and from Done too, each with its vote lit; voting one
 * again corrects it, and ↑ returns to the pick waiting for a vote. The round a
 * slip finished does not seal it.
 *
 * A vote here never re-reads the vote piles (`/api/votes`): only the Review
 * tab shows them, so the view reads them when Review opens (#5099). That read
 * was every item's presumed label then, megabytes on a large dataset; Find
 * reads the verified scope now (#5115).
 */
@Injectable()
export class LineTestSessionService {
  private readonly api = inject(LineTestApiService);
  private readonly metadataCache = inject(MediaMetadataCacheService);
  private readonly pairScope = inject(PairScopeService);

  /** The server's last word. */
  readonly response = signal<LineTestResponse | null>(null);
  /** The round's picks as dealt, in draw order (random): the order they come on screen. */
  readonly roundPicks = signal<number[]>([]);
  /** Every vote of the test so far, this round's and every earlier round's, as cast here. */
  readonly votes = signal<ReadonlyMap<number, VoteDirection>>(new Map());
  /** The voted picks in the order first voted: what ↓ walks back along. */
  readonly trail = signal<readonly number[]>([]);
  /** Which pick of the round is the one waiting for a vote. */
  readonly index = signal(0);
  /** Where a ↓ walk stands in {@link trail}; null when not walking back. */
  readonly walk = signal<number | null>(null);
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
  /** The balance is frozen while the test runs, through both phases (#4524): changing it would move the line and the bands under the picks. */
  readonly locked = computed(() => {
    const phase = this.phase();
    return phase === 'matches' || phase === 'misses';
  });
  /** Every pick of the round has its vote, and the next round is a reply away. */
  readonly awaitingRound = computed(() => {
    const votes = this.votes();
    return this.running() && this.roundPicks().every((id) => votes.has(id));
  });
  /** The pick on screen: the one a ↓ walk stands on, else the round's pick waiting for a vote. */
  readonly currentId = computed<number | null>(() => {
    const walk = this.walk();
    if (walk !== null) return this.trail()[walk] ?? null;
    return this.running() ? (this.roundPicks()[this.index()] ?? null) : null;
  });
  readonly currentVote = computed(() => {
    const id = this.currentId();
    return id === null ? null : (this.votes().get(id) ?? null);
  });

  private seq = 0;
  private applied = 0;

  /** Forget everything: a pair switch, or leaving the view. */
  clear(): void {
    this.response.set(null);
    this.roundPicks.set([]);
    this.votes.set(new Map());
    this.trail.set([]);
    this.index.set(0);
    this.walk.set(null);
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
   * picks the detector keeps (#4526) made them session votes again; the
   * Review tab's piles pick them up when it opens.
   */
  start(): void {
    this.send(this.api.start(), 'Could not start the test.');
  }

  /**
   * Vote pick *id* (the one on screen by default). A pick of the round waiting
   * for a vote takes it and the next unvoted one comes on screen; a pick
   * already voted is corrected (#5502), and the walk ends, back at the pick
   * waiting for a vote, as a vote anywhere else in the app moves on.
   */
  vote(direction: VoteDirection, id: number | null = this.currentId()): void {
    if (id === null) return;
    const was = this.votes().get(id);
    if (was !== undefined) {
      this.walk.set(null);
      if (was === direction) return;
      this.setVote(id, direction);
      this.send(this.api.vote([{ id, label: direction }]), 'Could not send the vote.', () => this.setVote(id, was));
      return;
    }
    const i = this.roundPicks().indexOf(id);
    if (!this.running() || i < 0) return;
    this.walk.set(null);
    this.setVote(id, direction);
    this.trail.set([...this.trail(), id]);
    const unvoted = this.nextUnvoted(i);
    if (unvoted !== null) this.index.set(unvoted);
    this.send(
      this.api.vote([{ id, label: direction }]),
      'Could not send the vote.',
      // Refused: the pick is unvoted again, and back on screen to vote once more.
      () => {
        this.setVote(id, null);
        this.trail.set(this.trail().filter((t) => t !== id));
        const at = this.roundPicks().indexOf(id);
        if (at >= 0) this.index.set(at);
        this.walk.set(null);
      },
    );
  }

  /**
   * ↓ walks back one pick along the picks already voted, from any round and
   * from Done, with its vote lit; ↑ ends the walk, back at the pick waiting
   * for a vote (or the verdict), or with no walk goes on to the next unvoted
   * pick. Nothing is sent: a vote on a pick walked back to corrects it.
   */
  navigate(direction: NavDirection): void {
    const trail = this.trail();
    const walk = this.walk();
    if (direction === 'forward') {
      if (walk !== null) {
        this.walk.set(null);
        return;
      }
      if (!this.running()) return;
      const unvoted = this.nextUnvoted(this.index());
      if (unvoted !== null) this.index.set(unvoted);
      return;
    }
    if (walk !== null) {
      if (walk > 0) this.walk.set(walk - 1);
      return;
    }
    // The newest vote, unless it is the pick already on screen (the round's
    // last, waiting on the next round).
    let at = trail.length - 1;
    if (at >= 0 && trail[at] === this.currentId()) at -= 1;
    if (at >= 0) this.walk.set(at);
  }

  /**
   * The next *depth* picks a vote will bring on screen, in that order: the
   * unvoted picks after the one waiting for a vote, wrapping round, as
   * {@link vote} walks them, then the next round's first picks. What the view
   * warms while the user looks at this one (#5099).
   *
   * The next round is known before the vote that deals it (#5116): once one
   * pick is left the server deals that round both ways the pick can go
   * (`test.ahead`). While the pick is unvoted both are upcoming, taken in
   * turns, since either may come (they are the same picks when both ways go
   * to the same band); once it has its vote here, the way it went, so the
   * other's fetch stops as the vote goes out.
   */
  upcoming(depth: number): number[] {
    const picks = this.roundPicks();
    const votes = this.votes();
    const from = this.index();
    const ids: number[] = [];
    for (let step = 1; step < picks.length && ids.length < depth; step++) {
      const id = picks[(from + step) % picks.length];
      if (!votes.has(id)) ids.push(id);
    }
    const next = this.nextRounds();
    for (let i = 0; ids.length < depth && next.some((round) => i < round.length); i++) {
      for (const round of next) {
        const id = round[i];
        if (id !== undefined && ids.length < depth && !ids.includes(id)) ids.push(id);
      }
    }
    return ids;
  }

  /** Drop a running test; its votes stay. A finished one is kept. */
  cancel(): void {
    this.send(this.api.cancel(), 'Could not stop the test.');
  }

  private inFlight = 0;

  /** Set (or, with null, lift) the vote this session holds on *id*. */
  private setVote(id: number, direction: VoteDirection | null | undefined): void {
    const next = new Map(this.votes());
    if (direction) next.set(id, direction);
    else next.delete(id);
    this.votes.set(next);
  }

  private send(
    request: Observable<LineTestResponse>,
    failure: string,
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
   * stays. The test's votes are read back from the server's `votes` only once
   * nothing else is in flight: a reply to one vote lands while the next is
   * still out, and would erase it; a refused vote is rolled back by its own
   * request instead.
   */
  private apply(resp: LineTestResponse): void {
    const previous = this.test();
    this.response.set(resp);
    const test = resp.test ?? null;
    if (!test) {
      this.roundPicks.set([]);
      this.votes.set(new Map());
      this.trail.set([]);
      this.index.set(0);
      this.walk.set(null);
      return;
    }
    if (this.inFlight === 0) {
      const wire = test.votes ?? [];
      this.votes.set(new Map(wire.map((v) => [v.id, v.label as VoteDirection])));
      this.trail.set(wire.map((v) => v.id));
      const walk = this.walk();
      if (walk !== null && walk >= wire.length) this.walk.set(null);
    }
    // The next round, dealt ahead (#5116): its metadata is in hand as it is dealt, like its images.
    if (test.ahead) this.metadataCache.ensureLoaded([...test.ahead.good, ...test.ahead.bad]);
    if (test.phase === 'done' || test.phase === 'nothing') {
      this.roundPicks.set([]);
      this.index.set(0);
      return;
    }
    const pending = test.picks ?? [];
    if (!pending.length) return;
    const dealt = this.roundPicks();
    const fresh = !previous || test.round !== previous.round || pending.some((id) => !dealt.includes(id));
    if (fresh) {
      this.roundPicks.set([...pending]);
      this.index.set(0);
      this.metadataCache.ensureLoaded(pending);
    }
  }

  /**
   * The rounds the vote on this round's last pick can deal, from the
   * server's `ahead`: both while that pick is the one unvoted, the one its
   * vote names once every pick has one, and none while two or more are
   * unvoted.
   */
  private nextRounds(): number[][] {
    const ahead = this.test()?.ahead;
    const picks = this.roundPicks();
    if (!ahead || !picks.includes(ahead.id)) return [];
    const votes = this.votes();
    const unvoted = picks.filter((id) => !votes.has(id));
    const vote = votes.get(ahead.id);
    if (vote) return unvoted.length ? [] : [vote === 'good' ? ahead.good : ahead.bad];
    return unvoted.length === 1 ? [ahead.good, ahead.bad] : [];
  }

  /** The first unvoted pick after *from*, wrapping round; null when every pick has a vote. */
  private nextUnvoted(from: number): number | null {
    const picks = this.roundPicks();
    const votes = this.votes();
    for (let step = 1; step <= picks.length; step++) {
      const i = (from + step) % picks.length;
      if (!votes.has(picks[i])) return i;
    }
    return null;
  }
}
