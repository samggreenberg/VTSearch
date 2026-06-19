import { Component, Input, OnChanges, OnDestroy, SimpleChanges, ViewChild, effect, inject, input, output, signal, untracked } from '@angular/core';
import { KeyValuePipe } from '@angular/common';
import { Subscription } from 'rxjs';
import { EmbedderInfo, Media } from '../../models/api.models';
import { MediasApiService } from '../../services/medias-api.service';
import { KeyboardService } from '../../services/keyboard.service';
import { VoteStateService } from '../../services/vote-state.service';
import { SettingsStateService } from '../../services/settings-state.service';
import { SortStateService } from '../../services/sort-state.service';
import { DatasetsListingsApiService } from '../../services/datasets-listings-api.service';
import { AudioPlayerComponent } from './audio-player/audio-player.component';
import { ImageViewerComponent, RegionBox } from './image-viewer/image-viewer.component';
import { VideoPlayerComponent } from './video-player/video-player.component';
import { TextViewerComponent } from './text-viewer/text-viewer.component';
import { DocumentViewerComponent } from './document-viewer/document-viewer.component';
import { VotingOverlayComponent } from './voting-overlay/voting-overlay.component';
import { prefersReducedMotion } from '../../utils/reduced-motion';

@Component({
  selector: 'vt-center-panel',
  standalone: true,
  imports: [
    KeyValuePipe,
    AudioPlayerComponent,
    ImageViewerComponent,
    VideoPlayerComponent,
    TextViewerComponent,
    DocumentViewerComponent,
    VotingOverlayComponent,
  ],
  templateUrl: './center-panel.component.html',
  styleUrl: './center-panel.component.scss',
})
export class CenterPanelComponent implements OnChanges, OnDestroy {
  private mediasApi = inject(MediasApiService);
  private keyboard = inject(KeyboardService);
  voteState = inject(VoteStateService);
  private settingsState = inject(SettingsStateService);
  private sortState = inject(SortStateService);
  private datasetsListingsApi = inject(DatasetsListingsApiService);

  @Input() media: Media | null = null;
  readonly disabled = input(false);
  readonly mediaVoted = output<{
    id: number;
    vote: 'good' | 'bad';
}>();

  @ViewChild(AudioPlayerComponent) audioPlayer?: AudioPlayerComponent;
  @ViewChild(ImageViewerComponent) imageViewer?: ImageViewerComponent;
  @ViewChild(VideoPlayerComponent) videoPlayer?: VideoPlayerComponent;

  // These fields are written from non-bound callbacks — the keyboard-shortcut
  // dispatch (`KeyboardService.action$`), HTTP/vote subscriptions, and timers —
  // and read in the template, so under zoneless CD they must be signals: a plain
  // field write from those callbacks would not notify the scheduler and the view
  // would silently go stale. (zoneless-migration.md, Phase 1.2 / Recipe B & F.)
  readonly isVoting = signal(false);
  readonly volume = signal(1);
  readonly audioPlaying = signal(true);
  readonly showAnimations = signal(true);
  readonly showMetadata = signal(true);
  readonly swipeClass = signal('');
  readonly spinningVote = signal<'good' | 'bad' | null>(null);

  /** Persisted dismissal of the zero-votes first-vote hint. Initialised
   *  to ``true`` so the hint never flashes before settings load resolves;
   *  loadSettings() flips it to ``false`` only when the server confirms
   *  the user has never dismissed it. */
  private readonly labelHintDismissed = signal(true);

  /** Transient text shown after Cmd/Ctrl-Z; auto-cleared after a short delay. */
  readonly undoToastText = signal<string | null>(null);
  private undoToastTimer: ReturnType<typeof setTimeout> | null = null;

  // Bad-vote-with-box state: drawing a box is real work, so a stray ← shouldn't
  // throw it away. The first ← arms a sticky discard-confirm state (no timeout);
  // the second ← throws the box away and votes no. Esc, a mousedown on the box,
  // a Shift-drag-redraw, or navigating to another item all clear the armed
  // state without voting and without discarding the box.
  // Public so the template can bind it into the image viewer.
  currentRegionBox: RegionBox | null = null;
  readonly pendingBadConfirm = signal(false);

  /** Embedder capability listings, loaded once in init(). Used to decide
   *  whether the active dataset's embedder is patch-region-aware, which gates
   *  the Highlight toggle in the image-view controls. */
  private embedderInfos: EmbedderInfo[] = [];

  private spinTimer: ReturnType<typeof setTimeout> | null = null;

  private _pausedByVisibility = false;
  private subs: Subscription[] = [];

  constructor() {
    effect(() => {
      const settings = this.settingsState.settingsSignal();
      if (!settings) return;
      this.volume.set(settings.volume ?? 1);
      this.audioPlaying.set(settings.audio_playing !== false);
      this.showAnimations.set(settings.show_animations !== false);
      this.showMetadata.set(settings.show_metadata !== false);
      this.labelHintDismissed.set(settings.label_hint_dismissed === true);
    });

    // Any vote in any pane (center buttons, keyboard, hover-vote) retires the
    // first-vote hint for this user. VoteStateService is signal-backed, so an
    // effect tracking the vote sets covers every channel without each call site
    // knowing about the hint. The dismiss logic runs `untracked` because it both
    // reads and writes `labelHintDismissed` — tracking that read would loop the
    // effect (zoneless-migration.md, Phase 2.5).
    effect(() => {
      this.voteState.goodVotes;
      this.voteState.badVotes;
      untracked(() => this.maybeDismissLabelHint());
    });
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['media']) {
      this.swipeClass.set('');
      // Navigating to another item clears the armed bad-vote-confirm state.
      // ImageViewer also clears its own regionBox on media change and will emit null;
      // resetting eagerly here keeps state coherent across the swap.
      this.pendingBadConfirm.set(false);
      this.currentRegionBox = null;
    }
  }

  ngOnDestroy(): void {
    this.stopPlayback();
    this.keyboard.stop();
    this.subs.forEach((s) => s.unsubscribe());
    if (this.spinTimer) clearTimeout(this.spinTimer);
    if (this.undoToastTimer) clearTimeout(this.undoToastTimer);
    document.removeEventListener('visibilitychange', this.onVisibilityChange);
  }

  /** Stop all media playback (used on navigation away). */
  stopPlayback(): void {
    if (this.audioPlayer) {
      const audio = this.audioPlayer.audioRef?.nativeElement;
      if (audio) {
        audio.pause();
      }
    }
    if (this.videoPlayer) {
      const video = this.videoPlayer.videoRef?.nativeElement;
      if (video) {
        video.pause();
      }
    }
  }

  /** Initialize: load settings, start keyboard listener, listen for tab visibility. */
  init(): void {
    this.loadSettings();
    this.keyboard.start();
    this.subs.push(
      this.datasetsListingsApi.getEmbedders().subscribe({
        next: (embedders) => (this.embedderInfos = embedders),
      }),
    );
    document.addEventListener('visibilitychange', this.onVisibilityChange);
    this.subs.push(
      this.keyboard.action$.subscribe((action) => {
        switch (action.type) {
          case 'vote':
            if (this.media && action.direction && !this.disabled()) {
              this.castVote(action.direction);
            }
            break;
          case 'volume':
            this.adjustVolume(action.volumeDelta ?? 0);
            break;
          case 'playback':
            this.togglePlayback();
            break;
          case 'zoom':
            if (this.imageViewer && action.zoomDirection) {
              if (action.zoomDirection === 'in') this.imageViewer.zoomIn();
              else this.imageViewer.zoomOut();
            }
            break;
          case 'rotate':
            if (this.imageViewer && action.rotateDirection) {
              if (action.rotateDirection === 'left') this.imageViewer.rotateLeft();
              else this.imageViewer.rotateRight();
            }
            break;
          case 'undo':
            if (!this.disabled() && !this.isVoting()) this.voteState.undo();
            break;
          case 'redo':
            if (!this.disabled() && !this.isVoting()) this.voteState.redo();
            break;
        }
      }),
      this.voteState.toast$.subscribe((t) => this.showUndoToast(t.action, t.mediaName)),
    );
  }

  /** Persist the first-vote hint as dismissed once the first vote lands. */
  private maybeDismissLabelHint(): void {
    if (this.labelHintDismissed()) return;
    if (this.voteState.goodVotes.size === 0 && this.voteState.badVotes.size === 0) return;
    this.labelHintDismissed.set(true);
    this.settingsState.update({ label_hint_dismissed: true }).subscribe();
  }

  private showUndoToast(action: 'undo' | 'redo', mediaName: string): void {
    const verb = action === 'undo' ? 'Undid vote on' : 'Redid vote on';
    this.undoToastText.set(`${verb} ${mediaName}`);
    if (this.undoToastTimer) clearTimeout(this.undoToastTimer);
    this.undoToastTimer = setTimeout(() => {
      this.undoToastText.set(null);
      this.undoToastTimer = null;
    }, 2000);
  }

  get mediaType(): string {
    return this.media?.media_type || 'audio';
  }

  /** Whether the active dataset's embedder produces patch regions. Gates the
   *  Highlight toggle: only patch-region-aware embedders (DINOv2/v3, EUPE) emit
   *  a best-match region, so the button is hidden entirely otherwise. Defaults
   *  to false when the embedder is unknown (embedders not yet loaded, or the
   *  media carries no embedder field) so a dead toggle never appears. */
  get patchCapable(): boolean {
    const name = this.media?.embedder;
    if (!name || this.embedderInfos.length === 0) return false;
    const info = this.embedderInfos.find((e) => e.name === name);
    return info?.supports_patch_regions === true;
  }

  /** The focused media's best-match region (the argmax patch region from the
   *  most recent sort/train), looked up by id from the in-memory sort results.
   *  ``null`` when the media wasn't scored or carries no region. Passed to the
   *  image viewer, which draws it only while Highlight is toggled on. */
  get highlightBox(): RegionBox | null {
    if (!this.media) return null;
    const id = this.media.id;
    const box = this.sortState.sortOrder?.find((s) => s.id === id)?.bestRegion;
    if (!box || box.length !== 4) return null;
    return [box[0], box[1], box[2], box[3]];
  }

  get isGood(): boolean {
    return this.media ? this.voteState.effectiveGood(this.media.id) : false;
  }

  get isBad(): boolean {
    return this.media ? this.voteState.effectiveBad(this.media.id) : false;
  }

  /** True when the labeling session is fresh (no votes yet across either
   *  polarity) and the user has not previously dismissed the hint. The
   *  hint dismisses on the first vote in this session and persists. */
  get showFirstVoteHint(): boolean {
    if (this.labelHintDismissed()) return false;
    return this.voteState.goodVotes.size === 0 && this.voteState.badVotes.size === 0;
  }

  get customMetadata(): Record<string, unknown> {
    if (!this.media) return {};
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return (this.media as any)['custom_metadata'] as Record<string, unknown> || {};
  }

  /** Human-readable label for an item used in undo toasts. */
  private mediaDisplayName(media: Media): string {
    return media.filename || media.origin_name || `#${media.id}`;
  }

  formatMetadataValue(label: string, value: unknown): string {
    if (label === 'File Size' && typeof value === 'number') {
      return (value / 1024).toFixed(1) + ' KB';
    }
    if (label === 'Duration' && typeof value === 'number') {
      return value.toFixed(1) + 's';
    }
    if (label === 'Frequency' && typeof value === 'number') {
      return value + ' Hz';
    }
    return String(value);
  }

  toggleMetadata(): void {
    this.showMetadata.set(!this.showMetadata());
    this.settingsState.update({ show_metadata: this.showMetadata() }).subscribe();
  }

  castVote(vote: 'good' | 'bad'): void {
    if (!this.media || this.isVoting()) return;

    // Region annotations only attach to yes-votes (salient-area semantics).
    // A no-vote with a box drawn arms a sticky discard-confirm state; the first
    // ← shake-pulses the box and surfaces a hint; only a second ← while armed
    // throws the box away and votes no. The state has no timeout: a time-based
    // modal would expire silently and surprise the user. Esc, mouse-on-box, a
    // fresh Shift-drag, or item navigation clear armed without voting.
    if (vote === 'bad' && this.currentRegionBox && !this.pendingBadConfirm()) {
      this.pendingBadConfirm.set(true);
      this.imageViewer?.pulseRegionBox();
      return;
    }

    const regionBox = vote === 'good' ? this.currentRegionBox : null;
    this.pendingBadConfirm.set(false);
    this.isVoting.set(true);

    this.voteState
      .submitToggleVoteAndRecord(this.media.id, vote, this.mediaDisplayName(this.media), regionBox)
      .subscribe({
        next: () => {
          const animate = this.showAnimations() && !!this.media && !prefersReducedMotion();
          if (animate) {
            this.swipeClass.set(vote === 'good' ? 'swipe-right' : 'swipe-left');
            this.spinningVote.set(vote);
            if (this.spinTimer) clearTimeout(this.spinTimer);
            this.spinTimer = setTimeout(() => this.spinningVote.set(null), 300);
            setTimeout(() => {
              this.mediaVoted.emit({ id: this.media!.id, vote });
              this.isVoting.set(false);
            }, 180);
          } else {
            this.mediaVoted.emit({ id: this.media!.id, vote });
            this.isVoting.set(false);
          }
        },
        error: () => {
          this.isVoting.set(false);
        },
      });
  }

  onRegionBoxChange(box: RegionBox | null): void {
    this.currentRegionBox = box;
    // Clearing the box also clears any pending bad-vote confirmation;
    // there's nothing left to confirm against.
    if (!box) this.pendingBadConfirm.set(false);
  }

  /** Esc-while-armed or mouse interaction with the box: cancel armed, keep the box. */
  onArmedConfirmCanceled(): void {
    this.pendingBadConfirm.set(false);
  }

  private loadSettings(): void {
    this.settingsState.load();
  }

  private adjustVolume(delta: number): void {
    this.volume.set(Math.max(0, Math.min(1, this.volume() + delta)));
    if (this.audioPlayer) {
      this.audioPlayer.adjustVolume(delta);
    }
    if (this.videoPlayer) {
      this.videoPlayer.adjustVolume(delta);
    }
    this.settingsState.update({ volume: this.volume() }).subscribe();
  }

  onPlayingChanged(playing: boolean): void {
    if (this._pausedByVisibility) return;
    this.audioPlaying.set(playing);
    this.settingsState.update({ audio_playing: this.audioPlaying() }).subscribe();
  }

  private onVisibilityChange = (): void => {
    if (document.hidden) {
      if (this.audioPlaying()) {
        this._pausedByVisibility = true;
        this.stopPlayback();
      }
    } else {
      if (this._pausedByVisibility) {
        this._pausedByVisibility = false;
        this.resumePlayback();
      }
    }
  };

  private resumePlayback(): void {
    if (this.audioPlayer) {
      const audio = this.audioPlayer.audioRef?.nativeElement;
      if (audio) audio.play().catch(() => {});
    }
    if (this.videoPlayer) {
      const video = this.videoPlayer.videoRef?.nativeElement;
      if (video) video.play().catch(() => {});
    }
  }

  private togglePlayback(): void {
    this.audioPlaying.set(!this.audioPlaying());
    if (this.audioPlayer) {
      this.audioPlayer.togglePlayback();
    }
    if (this.videoPlayer) {
      this.videoPlayer.togglePlayback();
    }
    this.settingsState.update({ audio_playing: this.audioPlaying() }).subscribe();
  }
}
