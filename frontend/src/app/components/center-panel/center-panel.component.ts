import { Component, EventEmitter, Input, OnChanges, OnDestroy, Output, SimpleChanges, ViewChild } from '@angular/core';
import { KeyValuePipe } from '@angular/common';
import { Subscription } from 'rxjs';
import { Media } from '../../models/api.models';
import { MediasApiService } from '../../services/medias-api.service';
import { KeyboardService } from '../../services/keyboard.service';
import { VoteStateService } from '../../services/vote-state.service';
import { SettingsStateService } from '../../services/settings-state.service';
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
  @Input() media: Media | null = null;
  @Input() disabled = false;
  @Output() mediaVoted = new EventEmitter<{ id: number; vote: 'good' | 'bad' }>();

  @ViewChild(AudioPlayerComponent) audioPlayer?: AudioPlayerComponent;
  @ViewChild(ImageViewerComponent) imageViewer?: ImageViewerComponent;
  @ViewChild(VideoPlayerComponent) videoPlayer?: VideoPlayerComponent;

  isVoting = false;
  volume = 1;
  audioPlaying = true;
  swipeAnimation = true;
  showMetadata = true;
  swipeClass = '';
  spinningVote: 'good' | 'bad' | null = null;

  /** Persisted dismissal of the zero-votes first-vote hint. Initialised
   *  to ``true`` so the hint never flashes before settings load resolves;
   *  loadSettings() flips it to ``false`` only when the server confirms
   *  the user has never dismissed it. */
  private labelHintDismissed = true;

  /** Transient text shown after Cmd/Ctrl-Z; auto-cleared after a short delay. */
  undoToastText: string | null = null;
  private undoToastTimer: ReturnType<typeof setTimeout> | null = null;

  // Bad-vote-with-box state: drawing a box is real work, so a stray ← shouldn't
  // throw it away. The first ← arms a sticky discard-confirm state (no timeout);
  // the second ← throws the box away and votes no. Esc, a mousedown on the box,
  // a Shift-drag-redraw, or navigating to another item all clear the armed
  // state without voting and without discarding the box.
  // Public so the template can bind it into the image viewer.
  currentRegionBox: RegionBox | null = null;
  pendingBadConfirm = false;

  private spinTimer: ReturnType<typeof setTimeout> | null = null;

  private _pausedByVisibility = false;
  private subs: Subscription[] = [];

  constructor(
    private mediasApi: MediasApiService,
    private keyboard: KeyboardService,
    public voteState: VoteStateService,
    private settingsState: SettingsStateService,
  ) {}

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['media']) {
      this.swipeClass = '';
      // Navigating to another item clears the armed bad-vote-confirm state.
      // ImageViewer also clears its own regionBox on media change and will emit null;
      // resetting eagerly here keeps state coherent across the swap.
      this.pendingBadConfirm = false;
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
    document.addEventListener('visibilitychange', this.onVisibilityChange);
    this.subs.push(
      this.keyboard.action$.subscribe((action) => {
        switch (action.type) {
          case 'vote':
            if (this.media && action.direction && !this.disabled) {
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
            if (!this.disabled && !this.isVoting) this.voteState.undo();
            break;
          case 'redo':
            if (!this.disabled && !this.isVoting) this.voteState.redo();
            break;
        }
      }),
      this.voteState.toast$.subscribe((t) => this.showUndoToast(t.action, t.mediaName)),
      // Any vote in any pane (center buttons, keyboard, hover-vote) retires
      // the first-vote hint for this user. Watching the vote sets covers
      // every channel without each call site needing to know about the hint.
      this.voteState.goodVotes$.subscribe(() => this.maybeDismissLabelHint()),
      this.voteState.badVotes$.subscribe(() => this.maybeDismissLabelHint()),
    );
  }

  /** Persist the first-vote hint as dismissed once the first vote lands. */
  private maybeDismissLabelHint(): void {
    if (this.labelHintDismissed) return;
    if (this.voteState.goodVotes.size === 0 && this.voteState.badVotes.size === 0) return;
    this.labelHintDismissed = true;
    this.settingsState.update({ label_hint_dismissed: true }).subscribe();
  }

  private showUndoToast(action: 'undo' | 'redo', mediaName: string): void {
    const verb = action === 'undo' ? 'Undid vote on' : 'Redid vote on';
    this.undoToastText = `${verb} ${mediaName}`;
    if (this.undoToastTimer) clearTimeout(this.undoToastTimer);
    this.undoToastTimer = setTimeout(() => {
      this.undoToastText = null;
      this.undoToastTimer = null;
    }, 2000);
  }

  get mediaType(): string {
    return this.media?.media_type || 'audio';
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
    if (this.labelHintDismissed) return false;
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
    this.showMetadata = !this.showMetadata;
    this.settingsState.update({ show_metadata: this.showMetadata }).subscribe();
  }

  castVote(vote: 'good' | 'bad'): void {
    if (!this.media || this.isVoting) return;

    // Region annotations only attach to yes-votes (salient-area semantics).
    // A no-vote with a box drawn arms a sticky discard-confirm state; the first
    // ← shake-pulses the box and surfaces a hint; only a second ← while armed
    // throws the box away and votes no. The state has no timeout: a time-based
    // modal would expire silently and surprise the user. Esc, mouse-on-box, a
    // fresh Shift-drag, or item navigation clear armed without voting.
    if (vote === 'bad' && this.currentRegionBox && !this.pendingBadConfirm) {
      this.pendingBadConfirm = true;
      this.imageViewer?.pulseRegionBox();
      return;
    }

    const regionBox = vote === 'good' ? this.currentRegionBox : null;
    this.pendingBadConfirm = false;
    this.isVoting = true;

    this.voteState
      .submitToggleVoteAndRecord(this.media.id, vote, this.mediaDisplayName(this.media), regionBox)
      .subscribe({
        next: () => {
          const animate = this.swipeAnimation && !!this.media && !prefersReducedMotion();
          if (animate) {
            this.swipeClass = vote === 'good' ? 'swipe-right' : 'swipe-left';
            this.spinningVote = vote;
            if (this.spinTimer) clearTimeout(this.spinTimer);
            this.spinTimer = setTimeout(() => (this.spinningVote = null), 300);
            setTimeout(() => {
              this.mediaVoted.emit({ id: this.media!.id, vote });
              this.isVoting = false;
            }, 180);
          } else {
            this.mediaVoted.emit({ id: this.media!.id, vote });
            this.isVoting = false;
          }
        },
        error: () => {
          this.isVoting = false;
        },
      });
  }

  onRegionBoxChange(box: RegionBox | null): void {
    this.currentRegionBox = box;
    // Clearing the box also clears any pending bad-vote confirmation;
    // there's nothing left to confirm against.
    if (!box) this.pendingBadConfirm = false;
  }

  /** Esc-while-armed or mouse interaction with the box: cancel armed, keep the box. */
  onArmedConfirmCanceled(): void {
    this.pendingBadConfirm = false;
  }

  private loadSettings(): void {
    this.settingsState.load();
    this.subs.push(
      this.settingsState.settings$.subscribe((settings) => {
        if (!settings) return;
        this.volume = settings.volume ?? 1;
        this.audioPlaying = settings.audio_playing !== false;
        this.swipeAnimation = settings.swipe_animation !== false;
        this.showMetadata = settings.show_metadata !== false;
        this.labelHintDismissed = settings.label_hint_dismissed === true;
      }),
    );
  }

  private adjustVolume(delta: number): void {
    this.volume = Math.max(0, Math.min(1, this.volume + delta));
    if (this.audioPlayer) {
      this.audioPlayer.adjustVolume(delta);
    }
    if (this.videoPlayer) {
      this.videoPlayer.adjustVolume(delta);
    }
    this.settingsState.update({ volume: this.volume }).subscribe();
  }

  onPlayingChanged(playing: boolean): void {
    if (this._pausedByVisibility) return;
    this.audioPlaying = playing;
    this.settingsState.update({ audio_playing: this.audioPlaying }).subscribe();
  }

  private onVisibilityChange = (): void => {
    if (document.hidden) {
      if (this.audioPlaying) {
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
    this.audioPlaying = !this.audioPlaying;
    if (this.audioPlayer) {
      this.audioPlayer.togglePlayback();
    }
    if (this.videoPlayer) {
      this.videoPlayer.togglePlayback();
    }
    this.settingsState.update({ audio_playing: this.audioPlaying }).subscribe();
  }
}
