import { Component, EventEmitter, Input, OnChanges, OnDestroy, Output, SimpleChanges, ViewChild } from '@angular/core';
import { KeyValuePipe } from '@angular/common';
import { Subscription } from 'rxjs';
import { MediaItem } from '../../models/api.models';
import { MediasApiService } from '../../services/medias-api.service';
import { KeyboardService } from '../../services/keyboard.service';
import { VoteStateService } from '../../services/vote-state.service';
import { SettingsStateService } from '../../services/settings-state.service';
import { AudioPlayerComponent } from './audio-player/audio-player.component';
import { ImageViewerComponent } from './image-viewer/image-viewer.component';
import { VideoPlayerComponent } from './video-player/video-player.component';
import { TextViewerComponent } from './text-viewer/text-viewer.component';
import { DocumentViewerComponent } from './document-viewer/document-viewer.component';
import { VotingOverlayComponent } from './voting-overlay/voting-overlay.component';

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
    }
  }

  ngOnDestroy(): void {
    this.keyboard.stop();
    this.subs.forEach((s) => s.unsubscribe());
  }

  /** Initialize: load settings, start keyboard listener. */
  init(): void {
    this.loadSettings();
    this.keyboard.start();
    this.subs.push(
      this.keyboard.action$.subscribe((action) => {
        switch (action.type) {
          case 'vote':
            if (this.media && action.direction) {
              this.castVote(action.direction);
            }
            break;
          case 'volume':
            this.adjustVolume(action.volumeDelta ?? 0);
            break;
          case 'playback':
            this.togglePlayback();
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
    this.isVoting = true;

    this.mediasApi.vote(this.media.id, vote).subscribe({
      next: () => {
        if (this.swipeAnimation && this.media) {
          this.swipeClass = vote === 'good' ? 'swipe-right' : 'swipe-left';
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
    this.audioPlaying = playing;
    this.settingsState.update({ audio_playing: this.audioPlaying }).subscribe();
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
