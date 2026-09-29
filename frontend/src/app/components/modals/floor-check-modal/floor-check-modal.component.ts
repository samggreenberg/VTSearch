import { ChangeDetectionStrategy, Component, DestroyRef, OnInit, computed, inject, output, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { ModalComponent } from '../../modal/modal.component';
import { AudioPlayerComponent } from '../../center-panel/audio-player/audio-player.component';
import { DocumentViewerComponent } from '../../center-panel/document-viewer/document-viewer.component';
import { ImageViewerComponent } from '../../center-panel/image-viewer/image-viewer.component';
import { TextViewerComponent } from '../../center-panel/text-viewer/text-viewer.component';
import { VideoPlayerComponent } from '../../center-panel/video-player/video-player.component';
import { VotingOverlayComponent } from '../../center-panel/voting-overlay/voting-overlay.component';
import { KeyboardService, type NavDirection, type VoteDirection } from '../../../services/keyboard.service';
import { MediaMetadataCacheService } from '../../../services/media-metadata-cache.service';
import { MediaStateService } from '../../../services/media-state.service';
import { SortingApiService } from '../../../services/sorting-api.service';
import type { Media } from '../../../models/api.models';
import type { PrecisionCheckResponse } from '../../../generated/api-client/models/precision-check-response';
import type { PrecisionCheckState } from '../../../generated/api-client/models/precision-check-state';
import { apiErrorMessage } from '../../../utils/api-error';
import { floorPercent, lineFloorFrom, rangePercent, type LineFloor } from '../../../utils/line-floor';

/** Where the step is: drawing the first round, voting on a round, sending it, on the result, or refused. */
type Phase = 'starting' | 'voting' | 'sending' | 'done' | 'error';

/** What a host needs after a round's votes land: they are ordinary votes, and a finished check moves the line. */
export interface FloorCheckVoted {
  /** True when this round ended the check: the line now keeps the set it ended on. */
  finished: boolean;
  response: PrecisionCheckResponse;
}

/**
 * The precision floor's spot check (#4273; the rule is #4272's).
 *
 * Opens from the floor control's "Check 5 picks". The server draws each
 * round's picks uniformly at random from a candidate of the top unvoted items,
 * and the user votes each one Good or Bad. A round that falls short halves the
 * candidate and draws a fresh round, so a check is one to three rounds; it
 * ends on how close the line got - a likely range from the picks alone.
 *
 * - **The picks are a check, not the ranking** (owner, 2026-09-29). They show
 *   one at a time in the order they were drawn, which is random, with no rank
 *   and no score: the progress dots say only how many are left.
 * - **The usual keys.** → Good, ← Bad, ↓ back to the previous pick, ↑ on to
 *   the next unvoted one. They reach this step through a
 *   {@link KeyboardService.captureVoteKeys} claim held for its lifetime, so
 *   the ranked list behind the modal never sees them.
 * - **A round is sent whole.** Votes are held here until every pick in the
 *   round has one, so a slip can be taken back with ↓; the last vote sends the
 *   round. The server records them as ordinary votes (provenance `check`).
 * - **Closing leaves the state as it was.** Cancel, Escape or × on a running
 *   check cancels it server-side; the rounds already sent stay votes, and the
 *   floor keeps its last result. The host re-reads the votes and the line on
 *   `closed`, so nothing the step saw is left half-applied.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-floor-check-modal',
  standalone: true,
  imports: [
    ModalComponent,
    AudioPlayerComponent,
    DocumentViewerComponent,
    ImageViewerComponent,
    TextViewerComponent,
    VideoPlayerComponent,
    VotingOverlayComponent,
  ],
  templateUrl: './floor-check-modal.component.html',
  styleUrl: './floor-check-modal.component.scss',
})
export class FloorCheckModalComponent implements OnInit {
  private readonly sortingApi = inject(SortingApiService);
  private readonly mediaState = inject(MediaStateService);
  private readonly metadataCache = inject(MediaMetadataCacheService);
  private readonly keyboard = inject(KeyboardService);
  private readonly destroyRef = inject(DestroyRef);

  /** The step closed, however it ended. */
  readonly closed = output<void>();
  /** A round's votes landed on the server. */
  readonly voted = output<FloorCheckVoted>();

  readonly phase = signal<Phase>('starting');
  readonly error = signal('');
  /** The check as the server last reported it. */
  readonly check = signal<PrecisionCheckState | null>(null);
  /** The floor's state from the last response: the result, once the check ends. */
  readonly floor = signal<LineFloor | null>(null);
  /** This round's picks, in draw order. */
  readonly picks = signal<number[]>([]);
  /** This round's votes, held until the round is whole. */
  readonly votes = signal<ReadonlyMap<number, VoteDirection>>(new Map());
  /** Which pick is on screen. */
  readonly index = signal(0);
  /** True once a round fell short and a shorter list is being checked. */
  readonly shorter = signal(false);

  readonly currentId = computed(() => this.picks()[this.index()] ?? null);
  readonly currentVote = computed(() => {
    const id = this.currentId();
    return id === null ? null : (this.votes().get(id) ?? null);
  });
  readonly currentMedia = computed<Media | null>(() => {
    const id = this.currentId();
    if (id === null) return null;
    // Read the cache's version so a batch arriving repaints the viewer.
    this.metadataCache.version();
    return this.mediaState.getMedia(id);
  });
  readonly mediaType = computed(() => this.currentMedia()?.media_type ?? '');

  readonly target = computed(() => floorPercent(this.check()?.min_precision ?? this.floor()?.minPrecision ?? 0.5));
  readonly title = computed(() => `Spot check: at least ${this.target()} right?`);

  /** "Round 2 of up to 3: 5 picks drawn at random from the top 64." */
  readonly brief = computed(() => {
    const c = this.check();
    if (!c) return '';
    const n = this.picks().length;
    const drawn = `${n} ${n === 1 ? 'pick' : 'picks'} drawn at random from the top ${c.candidate.toLocaleString()}`;
    return c.rounds > 1 ? `Round ${c.round} of up to ${c.rounds}: ${drawn}.` : `${drawn[0].toUpperCase()}${drawn.slice(1)}.`;
  });

  /** The result's headline: the floor held, or how close it got. A short check names no cause. */
  readonly resultHeadline = computed(() => {
    const f = this.floor();
    if (!f) return '';
    const target = floorPercent(f.minPrecision);
    const r = f.range;
    if (f.status === 'confirmed') {
      return r ? `At least ${target} right: likely ${rangePercent(r)} (checked ${r.labelled}).` : `At least ${target} right.`;
    }
    return r ? `Aimed at ${target}: likely ${rangePercent(r)} right (checked ${r.labelled}).` : `Aimed at ${target}: fell short.`;
  });

  /** What the line keeps now. */
  readonly resultDetail = computed(() => {
    const f = this.floor();
    if (!f) return '';
    const kept = f.count.toLocaleString();
    return f.status === 'confirmed'
      ? `The line keeps these ${kept}. The range is how much of them the picks say is right.`
      : `The line keeps the top ${kept} the check ended on. The range is how much of them the picks say is right.`;
  });

  constructor() {
    const release = this.keyboard.captureVoteKeys({
      vote: (direction) => this.vote(direction),
      navigate: (direction) => this.navigate(direction),
    });
    this.destroyRef.onDestroy(release);
  }

  ngOnInit(): void {
    this.sortingApi.startPrecisionCheck().pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (resp) => this.apply(resp),
      error: (err) => {
        this.error.set(apiErrorMessage(err, 'Could not start the check.'));
        this.phase.set('error');
      },
    });
  }

  /** Vote the pick on screen, then move to the next unvoted one; the round's last vote sends it. */
  vote(direction: VoteDirection): void {
    const id = this.currentId();
    if (this.phase() !== 'voting' || id === null) return;
    const next = new Map(this.votes());
    next.set(id, direction);
    this.votes.set(next);
    const unvoted = this.nextUnvoted(this.index());
    if (unvoted === null) {
      this.send();
      return;
    }
    this.index.set(unvoted);
  }

  /** ↓ steps back to the previous pick (to change its vote); ↑ goes on to the next unvoted one. */
  navigate(direction: NavDirection): void {
    if (this.phase() !== 'voting') return;
    if (direction === 'back') {
      this.index.set(Math.max(0, this.index() - 1));
      return;
    }
    const unvoted = this.nextUnvoted(this.index());
    if (unvoted !== null) this.index.set(unvoted);
  }

  /** Show pick *i* of the round. */
  show(i: number): void {
    if (this.phase() !== 'voting') return;
    if (i >= 0 && i < this.picks().length) this.index.set(i);
  }

  /** Send the round again after a failed POST. */
  retry(): void {
    if (this.phase() === 'voting' && this.picks().every((id) => this.votes().has(id))) this.send();
  }

  /**
   * Close the step. A running check is cancelled so the floor keeps its last
   * result; the rounds already sent stay votes either way. The cancel outlives
   * the step on purpose, and the host re-reads the votes and the line on
   * `closed`, which covers a round that lands after the step has gone.
   */
  close(): void {
    const phase = this.phase();
    if (phase === 'starting' || phase === 'voting' || phase === 'sending') {
      this.sortingApi.cancelPrecisionCheck().subscribe({ error: () => undefined });
    }
    this.closed.emit();
  }

  private send(): void {
    const votes = this.picks()
      .filter((id) => this.votes().has(id))
      .map((id) => ({ id, label: this.votes().get(id)! }));
    this.phase.set('sending');
    this.error.set('');
    // Tied to the step's life: a round still in flight when it closes is the
    // host's to catch up on, which it does on `closed`.
    this.sortingApi.votePrecisionCheck(votes).pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
      next: (resp) => {
        const before = this.check()?.round ?? 1;
        this.apply(resp);
        const finished = resp.check?.status === 'confirmed' || resp.check?.status === 'short';
        if (!finished && (resp.check?.round ?? before) > before) this.shorter.set(true);
        this.voted.emit({ finished, response: resp });
      },
      error: (err) => {
        // The round stays on screen with its votes, to send again.
        this.error.set(apiErrorMessage(err, 'Could not send the votes.'));
        this.phase.set('voting');
      },
    });
  }

  /** Take a response: a round to vote on, or the result. */
  private apply(resp: PrecisionCheckResponse): void {
    const c = resp.check ?? null;
    this.check.set(c);
    this.floor.set(lineFloorFrom(resp.floor));
    if (c?.status === 'running' && c.picks.length > 0) {
      this.picks.set([...c.picks]);
      this.votes.set(new Map());
      this.index.set(0);
      this.metadataCache.ensureLoaded(c.picks);
      this.phase.set('voting');
      return;
    }
    if (c?.status === 'confirmed' || c?.status === 'short') {
      this.phase.set('done');
      return;
    }
    this.error.set('The check ended without a result.');
    this.phase.set('error');
  }

  /** The first unvoted pick after *from*, wrapping round; null when every pick has a vote. */
  private nextUnvoted(from: number): number | null {
    const picks = this.picks();
    const votes = this.votes();
    for (let step = 1; step <= picks.length; step++) {
      const i = (from + step) % picks.length;
      if (!votes.has(picks[i])) return i;
    }
    return null;
  }
}
