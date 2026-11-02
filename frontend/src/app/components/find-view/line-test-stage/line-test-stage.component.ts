import { ChangeDetectionStrategy, Component, DestroyRef, computed, inject, output } from '@angular/core';
import { DecimalPipe } from '@angular/common';

import { AudioPlayerComponent } from '../../center-panel/audio-player/audio-player.component';
import { DocumentViewerComponent } from '../../center-panel/document-viewer/document-viewer.component';
import { ImageViewerComponent } from '../../center-panel/image-viewer/image-viewer.component';
import { TextViewerComponent } from '../../center-panel/text-viewer/text-viewer.component';
import { VideoPlayerComponent } from '../../center-panel/video-player/video-player.component';
import { VoteHintId, VotingOverlayComponent } from '../../center-panel/voting-overlay/voting-overlay.component';
import { IconComponent } from '../../icon/icon.component';
import { ToastyHintComponent } from '../../toasty-hint/toasty-hint.component';
import { KeyboardService } from '../../../services/keyboard.service';
import { LineTestSessionService } from '../../../services/line-test-session.service';
import { MediaMetadataCacheService } from '../../../services/media-metadata-cache.service';
import { MediaStateService } from '../../../services/media-state.service';
import type { Media } from '../../../models/api.models';
import { BALANCE_PRESETS } from '../../../utils/line-balance';
import { estimatePercent } from '../../../utils/line-test';

/** One row of the verdict's form: a label and the value measured for it. */
export interface VerdictField {
  label: string;
  value: string;
  /** What the value measures, on hover over its label. */
  title: string;
}

/**
 * The Tune autopilot's centre pane (#4524): the current pick with Good and
 * Bad, the round's dots, the phase's one-line prompt, and ↓ to go back one
 * pick, as the view rather than a modal over a list. Before a test exists it
 * says what the pane is waiting on.
 *
 * - **At Done it is the verdict** (#5100): the ranges as a form, one labelled
 *   value per row (#5503: a measured value reads as a field, never as words in
 *   a paragraph), and its three exits, **Move to AutoFind**, **Add Corrections and retrain**
 *   and **Lean the Threshold** (the ranges each preset would ship,
 *   re-estimated from the picks already taken). They sit where the user just
 *   voted, rather than under the numbers in the right pane, which keeps the
 *   result's detail and can be folded away.
 *
 * - **The picks are a sample, not the ranking** (#4267). They show one at a
 *   time in the order they were drawn, which is random, with no rank and no
 *   score: the dots say only how many are left.
 * - **The usual keys.** → Good, ← Bad, ↓ back a pick (and its vote is taken
 *   back), ↑ on to the next unvoted one, through a
 *   `KeyboardService.captureVoteKeys` claim held while the stage is on screen.
 *   The Review tab's centre panel is not rendered then, so nothing else hears
 *   them.
 * - **A vote goes out as it is cast.** The ranges on the right move per vote;
 *   the session service (`LineTestSessionService`) owns the round and the
 *   requests, and this pane only shows and asks.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-line-test-stage',
  standalone: true,
  imports: [
    DecimalPipe,
    AudioPlayerComponent,
    DocumentViewerComponent,
    ImageViewerComponent,
    TextViewerComponent,
    VideoPlayerComponent,
    VotingOverlayComponent,
    IconComponent,
    ToastyHintComponent,
  ],
  templateUrl: './line-test-stage.component.html',
  styleUrl: './line-test-stage.component.scss',
})
export class LineTestStageComponent {
  readonly session = inject(LineTestSessionService);
  private readonly mediaState = inject(MediaStateService);
  private readonly metadataCache = inject(MediaMetadataCacheService);
  private readonly keyboard = inject(KeyboardService);
  private readonly destroyRef = inject(DestroyRef);

  /** Move the detector to AutoFind. */
  readonly moveToAutoFind = output<void>();
  /** Lean the Threshold to a preset (a beta). */
  readonly lean = output<number>();
  /** Fold the session's corrections into the detector and retrain. */
  readonly addCorrections = output<void>();
  /** Test the line as it stands now (after it moved). */
  readonly testAgain = output<void>();

  readonly currentMedia = computed<Media | null>(() => {
    const id = this.session.currentId();
    if (id === null) return null;
    // Read the cache's version so a batch arriving repaints the viewer.
    this.metadataCache.version();
    return this.mediaState.getMedia(id);
  });
  readonly mediaType = computed(() => this.currentMedia()?.media_type ?? '');

  /** Toasty's hint over Good / Bad (#4807): the test's first pick, before any vote. */
  readonly voteHint = computed<VoteHintId | null>(() =>
    this.session.running() && (this.session.test()?.labelled ?? 0) === 0 && this.session.roundVotes().size === 0
      ? 'test-vote'
      : null,
  );

  /** The phase's one-line prompt over the pick. */
  readonly prompt = computed(() => {
    const test = this.session.test();
    const phase = this.session.phase();
    const n = this.session.roundPicks().length;
    const band = test?.band ?? null;
    const where = band
      ? band.side === 'above'
        ? band.lo === 1
          ? `the top ${band.hi.toLocaleString()}`
          : `items ${band.lo.toLocaleString()}–${band.hi.toLocaleString()} of the list`
        : `items ${band.lo.toLocaleString()}–${band.hi.toLocaleString()}, below the line`
      : '';
    const picks = `${n} ${n === 1 ? 'pick' : 'picks'} drawn at random${where ? ` from ${where}` : ''}`;
    if (phase === 'matches') return `Checking the matches: ${picks}. Is each one a match?`;
    if (phase === 'misses') return `Checking the misses: ${picks}. Is each one a match the line missed?`;
    return '';
  });

  // --- The verdict, at Done ---
  readonly moved = computed(() => this.session.response()?.moved ?? false);
  readonly stale = computed(() => this.session.response()?.stale ?? false);
  /** How many the line keeps now, for the note when it moved since the test. */
  readonly lineCount = computed(() => this.session.response()?.line_count ?? 0);

  /**
   * The verdict as a form (#5503): one labelled value per row, so each
   * measurement reads at a glance rather than from inside a sentence. A
   * reading of the ranges, not a threshold the app enforces.
   */
  readonly verdictFields = computed<VerdictField[]>(() => {
    const t = this.session.test();
    const e = t?.estimates ?? null;
    if (!e || !t) return [];
    // A structural or document detector has no class model, so its recall is read in words only (#4542).
    const classModel = t.class_model ?? true;
    return [
      {
        label: 'Line Keeps',
        value: t.line_count.toLocaleString(),
        title: 'How many items the line keeps on this collection: the top of the ranking, down to the Threshold.',
      },
      {
        label: 'Fraction Correct',
        value: estimatePercent(e.precision),
        title: 'Of what the line keeps, the share that is right: a likely range from random picks above the line.',
      },
      {
        label: 'Fraction Found',
        value: classModel ? estimatePercent(e.recall) : e.found.charAt(0).toUpperCase() + e.found.slice(1),
        title: classModel
          ? "Of all the matches in the collection, the share the line keeps: a likely range from random picks below the line, read against the detector's own count of what the unchecked tail holds."
          : 'Of all the matches in the collection, the share the line keeps, in words: this detector has no class model, so nothing counts the matches below where the picks reached.',
      },
    ];
  });

  /** The line each preset would draw, from the picks already taken, left to right as the control offers them. */
  readonly presetRows = computed(() => {
    const presets = this.session.response()?.presets ?? [];
    const current = this.session.test()?.beta ?? null;
    return BALANCE_PRESETS.map((preset) => {
      const p = presets.find((row) => row.beta === preset.value) ?? null;
      return { beta: preset.value, hint: preset.hint, current: current === preset.value, estimate: p };
    }).filter((row) => row.estimate !== null);
  });

  /** A likely range as the page reads every one: "96 ± 5%" (#5503). */
  readonly percent = estimatePercent;

  presetTitle(row:{ hint: string; estimate: { count: number; precision: { lo: number; hi: number }; recall: { lo: number; hi: number } } | null }): string {
    const e = row.estimate;
    if (!e) return row.hint;
    return `${row.hint}. The line would keep the top ${e.count.toLocaleString()}: likely ${estimatePercent(e.precision)} right, ${estimatePercent(e.recall)} of all the matches found.`;
  }

  constructor() {
    const release = this.keyboard.captureVoteKeys({
      vote: (direction) => this.session.vote(direction),
      navigate: (direction) => this.session.navigate(direction),
    });
    this.destroyRef.onDestroy(release);
  }
}
