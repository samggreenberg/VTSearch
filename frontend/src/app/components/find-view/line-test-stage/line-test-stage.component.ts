import { ChangeDetectionStrategy, Component, DestroyRef, computed, inject } from '@angular/core';

import { AudioPlayerComponent } from '../../center-panel/audio-player/audio-player.component';
import { DocumentViewerComponent } from '../../center-panel/document-viewer/document-viewer.component';
import { ImageViewerComponent } from '../../center-panel/image-viewer/image-viewer.component';
import { TextViewerComponent } from '../../center-panel/text-viewer/text-viewer.component';
import { VideoPlayerComponent } from '../../center-panel/video-player/video-player.component';
import { VotingOverlayComponent } from '../../center-panel/voting-overlay/voting-overlay.component';
import { IconComponent } from '../../icon/icon.component';
import { KeyboardService } from '../../../services/keyboard.service';
import { LineTestSessionService } from '../../../services/line-test-session.service';
import { MediaMetadataCacheService } from '../../../services/media-metadata-cache.service';
import { MediaStateService } from '../../../services/media-state.service';
import type { Media } from '../../../models/api.models';

/**
 * The Test autopilot's centre pane (#4524): the current pick with Good and
 * Bad, the round's dots, the phase's one-line prompt, and ↓ to go back one
 * pick, as the spot-check modal shows a round, but as the view rather than a
 * modal over a list. Before a test exists, and after Done, it says what the
 * pane is waiting on.
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
    AudioPlayerComponent,
    DocumentViewerComponent,
    ImageViewerComponent,
    TextViewerComponent,
    VideoPlayerComponent,
    VotingOverlayComponent,
    IconComponent,
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

  readonly currentMedia = computed<Media | null>(() => {
    const id = this.session.currentId();
    if (id === null) return null;
    // Read the cache's version so a batch arriving repaints the viewer.
    this.metadataCache.version();
    return this.mediaState.getMedia(id);
  });
  readonly mediaType = computed(() => this.currentMedia()?.media_type ?? '');

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

  constructor() {
    const release = this.keyboard.captureVoteKeys({
      vote: (direction) => this.session.vote(direction),
      navigate: (direction) => this.session.navigate(direction),
    });
    this.destroyRef.onDestroy(release);
  }
}
