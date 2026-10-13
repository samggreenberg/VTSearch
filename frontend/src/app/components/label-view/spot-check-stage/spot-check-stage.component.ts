import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  ElementRef,
  OnInit,
  computed,
  effect,
  inject,
  input,
  output,
  signal,
  viewChild,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { AudioPlayerComponent } from '../../center-panel/audio-player/audio-player.component';
import { DocumentViewerComponent } from '../../center-panel/document-viewer/document-viewer.component';
import { ImageViewerComponent } from '../../center-panel/image-viewer/image-viewer.component';
import { TextViewerComponent } from '../../center-panel/text-viewer/text-viewer.component';
import { VideoPlayerComponent } from '../../center-panel/video-player/video-player.component';
import { type VoteHintId, VotingOverlayComponent } from '../../center-panel/voting-overlay/voting-overlay.component';
import { KeyboardService, type NavDirection, type VoteDirection } from '../../../services/keyboard.service';
import { MediaMetadataCacheService } from '../../../services/media-metadata-cache.service';
import { MediaStateService } from '../../../services/media-state.service';
import { SortingApiService } from '../../../services/sorting-api.service';
import type { Media } from '../../../models/api.models';
import type { PrecisionCheckResponse } from '../../../generated/api-client/models/precision-check-response';
import type { PrecisionCheckState } from '../../../generated/api-client/models/precision-check-state';
import { apiErrorMessage } from '../../../utils/api-error';
import { foundWords, lineBalanceFrom, rangePercent, type LineBalance } from '../../../utils/line-balance';

/** Where the step is: drawing the first round, voting on a round, sending it, on the result, or refused. */
type Phase = 'starting' | 'voting' | 'sending' | 'done' | 'error';

/** What a host needs after a round's votes land: they are ordinary votes, and a finished check moves the line. */
export interface SpotCheckVoted {
  /** True when this round ended the check: the line now keeps the set where the balance peaked. */
  finished: boolean;
  response: PrecisionCheckResponse;
}

/**
 * Autopilot's Spot Check step in the Train view's centre pane (#4413; the
 * step is #4273's, walking the ranking in bands since #4388). It used to be a
 * modal the Threshold's "Check 5 picks" opened; now only Autopilot starts a
 * check, as one of its steps (`AutopilotStateService` phase `spot`): once
 * every quality light is green, before Done, and earlier while the labels
 * separate weakly (#4496). The host renders it in place of the centre panel
 * while the step runs.
 *
 * The server cuts the unvoted ranking into bands (the top 8, the next 8, then
 * 16, 32, ...) and draws each band's 5 picks uniformly at random; the user
 * votes each one Good or Bad. A band at a time, the check walks: deeper while
 * its estimate of the balance (F-beta) does not fall ("Better so far: checking
 * the next 32"), and back to a shorter list once it does ("Past the peak:
 * checking a shorter list"), ending on the set where the estimate peaked. The
 * result is what the picks say about that set: a likely share right, and how
 * many of all the matches it holds. Nothing is met or fallen short of.
 *
 * - **The picks are a check, not the ranking** (owner, 2026-09-29). They show
 *   one at a time in the order they were drawn, which is random, with no rank
 *   and no score: the progress dots say only how many are left. The ranked
 *   list is not on screen in Autopilot, so nothing shows one's rank.
 * - **The usual keys.** → Good, ← Bad, ↓ back to the previous pick, ↑ on to
 *   the next unvoted one. They reach this step through a
 *   {@link KeyboardService.captureVoteKeys} claim held for its lifetime, so
 *   the centre panel hidden behind it never sees them.
 * - **A round is sent whole.** Votes are held here until every pick in the
 *   round has one, so a slip can be taken back with ↓; the last vote sends the
 *   round. The server records them as ordinary votes (provenance `check`).
 * - **Stopping leaves the state as it was.** Stop check, or the step going
 *   away under a running check (Autopilot stopped, the view left), cancels it
 *   server-side; the rounds already sent stay votes, and the balance keeps its
 *   last result. The host re-reads the votes and the line on `closed`, so
 *   nothing the step saw is left half-applied.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-spot-check-stage',
  standalone: true,
  imports: [
    AudioPlayerComponent,
    DocumentViewerComponent,
    ImageViewerComponent,
    TextViewerComponent,
    VideoPlayerComponent,
    VotingOverlayComponent,
  ],
  templateUrl: './spot-check-stage.component.html',
  styleUrl: './spot-check-stage.component.scss',
})
export class SpotCheckStageComponent implements OnInit {
  private readonly sortingApi = inject(SortingApiService);
  private readonly mediaState = inject(MediaStateService);
  private readonly metadataCache = inject(MediaMetadataCacheService);
  private readonly keyboard = inject(KeyboardService);
  private readonly destroyRef = inject(DestroyRef);

  /** Why Autopilot is checking now: before Done, or while the labels separate weakly (#4496). Shown above the picks. */
  readonly intro = input<string | null>(null);
  /** Toasty may say what the step is (#4807): the run trains the detector, as for the phase hints. */
  readonly hintable = input(false);

  /** The step closed, however it ended. */
  readonly closed = output<void>();
  /** A round's votes landed on the server. */
  readonly voted = output<SpotCheckVoted>();

  readonly phase = signal<Phase>('starting');
  readonly error = signal('');
  /** The check as the server last reported it. */
  readonly check = signal<PrecisionCheckState | null>(null);
  /** The balance's state from the last response: the result, once the check ends. */
  readonly balance = signal<LineBalance | null>(null);
  /** This round's picks, in draw order. */
  readonly picks = signal<number[]>([]);
  /** This round's votes, held until the round is whole. */
  readonly votes = signal<ReadonlyMap<number, VoteDirection>>(new Map());
  /** Which pick is on screen. */
  readonly index = signal(0);
  /** What the last band decided, once the walk has moved: deeper, or back to a shorter list. Empty until then. */
  readonly walkNote = signal('');

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

  /** Toasty's hint over Good / Bad (#4807): the check's first pick, before any vote. */
  readonly voteHint = computed<VoteHintId | null>(() =>
    this.hintable() && this.phase() === 'voting' && (this.check()?.labelled ?? 0) === 0 && this.votes().size === 0
      ? 'spot-check'
      : null,
  );

  /** The step has been closed (or cancelled), so its going away cancels nothing more. */
  private closedOnce = false;

  private readonly continueBtn = viewChild<ElementRef<HTMLButtonElement>>('continueBtn');

  /**
   * "5 picks drawn at random from the top 8, checking the top 32." then
   * "5 picks drawn at random from items 33–64 of the list, checking the top 64."
   * The band is named by its place in the list, never the picks: they are a
   * check, not the ranking, and carry no rank of their own.
   */
  readonly brief = computed(() => {
    const c = this.check();
    if (!c) return '';
    const n = this.picks().length;
    const picks = `${n} ${n === 1 ? 'pick' : 'picks'} drawn at random`;
    const band = c.band;
    let from = '';
    if (band) {
      from =
        band.lo === 1
          ? ` from the top ${band.hi.toLocaleString()}`
          : ` from items ${band.lo.toLocaleString()}–${band.hi.toLocaleString()} of the list`;
    }
    const set = `checking the top ${c.candidate.toLocaleString()}`;
    return `${picks[0].toUpperCase()}${picks.slice(1)}${from}, ${set}.`;
  });

  /** The result's headline: what the picks found on the set the walk ended on (under `advisory`, named, since the line keeps another). */
  readonly resultHeadline = computed(() => {
    const b = this.balance();
    if (!b) return '';
    const p = b.precision;
    const r = b.recall;
    const what = b.shape === 'advisory' ? `Checked the top ${(b.audited ?? b.count).toLocaleString()}` : 'Checked';
    return p && r ? `${what}: likely ${rangePercent(p)} right, ${foundWords(r)} (checked ${p.labelled}).` : `${what}.`;
  });

  /** What the line keeps now, and what the ranges describe (#4427: under `advisory` the two differ). */
  readonly resultDetail = computed(() => {
    const b = this.balance();
    if (!b) return '';
    const kept = b.count.toLocaleString();
    if (b.shape === 'advisory') {
      const audited = (b.audited ?? b.count).toLocaleString();
      return (
        `The line keeps its ${kept}, where your labels put it: the check informs the line and does not move it. ` +
        `The ranges are what the picks say about the top ${audited}.`
      );
    }
    return `The line keeps these ${kept}: the set where the check's balance peaked. The ranges are what the picks say about them.`;
  });

  constructor() {
    const release = this.keyboard.captureVoteKeys({
      vote: (direction) => this.vote(direction),
      navigate: (direction) => this.navigate(direction),
    });
    this.destroyRef.onDestroy(release);
    // Gone from under a running check without Stop check (Autopilot stopped,
    // the view left): cancel it, as Stop check would.
    this.destroyRef.onDestroy(() => {
      if (!this.closedOnce) this.cancelIfRunning();
    });
    // On a result or a refusal, Continue takes the focus: Enter carries on.
    effect(() => this.continueBtn()?.nativeElement.focus());
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
   * End the step: Stop check on a running one, Continue on a result or a
   * refusal. A running check is cancelled so the balance keeps its last
   * result; the rounds already sent stay votes either way. The cancel outlives
   * the step on purpose, and the host re-reads the votes and the line on
   * `closed`, which covers a round that lands after the step has gone.
   */
  close(): void {
    if (this.closedOnce) return;
    this.cancelIfRunning();
    this.closedOnce = true;
    this.closed.emit();
  }

  private cancelIfRunning(): void {
    const phase = this.phase();
    if (phase === 'starting' || phase === 'voting' || phase === 'sending') {
      this.sortingApi.cancelPrecisionCheck().subscribe({ error: () => undefined });
    }
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
        const beforeBands = this.check()?.bands ?? 0;
        this.apply(resp);
        const finished = resp.check?.status === 'checked';
        const c = resp.check;
        if (!finished && c && c.bands !== beforeBands) {
          // The walk moved: the set under test grew by a band, or shrank.
          const next = c.band ? c.band.hi - c.band.lo + 1 : 0;
          this.walkNote.set(
            c.direction === 'deeper'
              ? `Better so far: checking the next ${next.toLocaleString()}.`
              : this.balance()?.shape === 'trim'
                ? 'Trimming: checking a shorter list.'
                : 'Past the peak: checking a shorter list.',
          );
        }
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
    this.balance.set(lineBalanceFrom(resp.balance));
    if (c?.status === 'running' && c.picks.length > 0) {
      this.picks.set([...c.picks]);
      this.votes.set(new Map());
      this.index.set(0);
      this.metadataCache.ensureLoaded(c.picks);
      this.phase.set('voting');
      return;
    }
    if (c?.status === 'checked') {
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
