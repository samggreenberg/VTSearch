import { Component, EventEmitter, Input, OnChanges, OnDestroy, Output, SimpleChanges, ViewChild } from '@angular/core';
import { KeyValuePipe } from '@angular/common';
import { Subscription } from 'rxjs';
import { MediaItem } from '../../models/api.models';
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
  @Input() media: MediaItem | null = null;
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

  private currentRegionBox: RegionBox | null = null;
  private pendingBadConfirm = false;
  private badConfirmTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly BAD_CONFIRM_MS = 2000;

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
      this.cancelBadConfirm();
      // ImageViewer also clears its own regionBox on media change and will emit null;
      // resetting eagerly here keeps state coherent across the swap.
      this.currentRegionBox = null;
    }
  }

  ngOnDestroy(): void {
    this.stopPlayback();
    this.keyboard.stop();
    this.subs.forEach((s) => s.unsubscribe());
    if (this.spinTimer) clearTimeout(this.spinTimer);
    if (this.badConfirmTimer) clearTimeout(this.badConfirmTimer);
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
        }
      }),
    );
  }

  get mediaType(): string {
    return this.media?.type || 'audio';
  }

  get isGood(): boolean {
    return this.media ? this.voteState.goodVotes.has(this.media.id) : false;
  }

  get isBad(): boolean {
    return this.media ? this.voteState.badVotes.has(this.media.id) : false;
  }

  get customMetadata(): Record<string, unknown> {
    if (!this.media) return {};
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return (this.media as any)['custom_metadata'] as Record<string, unknown> || {};
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
    // A no-vote with a box drawn requires a confirming second press so a stray
    // ArrowLeft can't throw away real work the user did drawing the box.
    if (vote === 'bad' && this.currentRegionBox && !this.pendingBadConfirm) {
      this.pendingBadConfirm = true;
      this.imageViewer?.pulseRegionBox();
      if (this.badConfirmTimer) clearTimeout(this.badConfirmTimer);
      this.badConfirmTimer = setTimeout(() => this.cancelBadConfirm(), this.BAD_CONFIRM_MS);
      return;
    }

    const regionBox = vote === 'good' ? this.currentRegionBox : null;
    this.cancelBadConfirm();
    this.isVoting = true;

    this.mediasApi.vote(this.media.id, vote, regionBox).subscribe({
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
    // Clearing the box also clears any pending bad-vote confirmation —
    // there's nothing left to confirm against.
    if (!box) this.cancelBadConfirm();
  }

  private cancelBadConfirm(): void {
    this.pendingBadConfirm = false;
    if (this.badConfirmTimer) {
      clearTimeout(this.badConfirmTimer);
      this.badConfirmTimer = null;
    }
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
