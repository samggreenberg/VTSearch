import { ChangeDetectionStrategy, Component, effect, ElementRef, inject, input, OnDestroy, output, signal, untracked, viewChild } from '@angular/core';

import { Media } from '../../../models/api.models';
import { ActiveContextService } from '../../../services/active-context.service';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-video-player',
  standalone: true,
  imports: [],
  templateUrl: './video-player.component.html',
  styleUrl: './video-player.component.scss',
})
export class VideoPlayerComponent implements OnDestroy {
  private activeContext = inject(ActiveContextService);

  readonly media = input.required<Media>();
  readonly volume = input(1);
  readonly audioPlaying = input(true);
  readonly playingChanged = output<boolean>();

  // Public: CenterPanelComponent reaches through this to pause/resume the
  // native element on navigation / tab-visibility changes.
  readonly videoRef = viewChild<ElementRef<HTMLVideoElement>>('videoEl');

  // Signals: written from the media-change effect below (videoSrc) and read in
  // the template, so plain fields would leave the view stale under zoneless.
  readonly videoSrc = signal('');
  readonly videoError = signal(false);

  // Active clip window bounds while enforcement is on, else null. Enforcement is
  // driven by the <video> element's (timeupdate) event rather than a polling
  // timer, so it only runs while the clip is actually playing and progressing.
  private clipBounds: { start: number; end: number } | null = null;
  // See ImageViewerComponent.lastMediaId; guards against metadata-enrichment
  // effect cycles rebuilding videoSrc (and yanking playback) for the
  // same id.
  private lastMediaId: number | null = null;
  // Whether (loadedmetadata) has fired for the current videoSrc. Clip bounds
  // (clip_start/clip_end) often arrive via batch hydration *after* the video
  // has already loaded, on a later media change with the same media id; in that
  // case (loadedmetadata) does not fire again, so we (re)apply the bounds here.
  private metadataLoaded = false;

  constructor() {
    // Point the element at the new clip when the media id changes; re-apply
    // clip bounds on same-id metadata enrichment. The src rides a template
    // binding, so no view-query dependency is needed here.
    effect(() => {
      const media = this.media();
      if (media.id !== this.lastMediaId) {
        this.lastMediaId = media.id;
        this.videoError.set(false);
        this.metadataLoaded = false;
        this.stopClipEnforcement();
        this.videoSrc.set(this.activeContext.mediaUrl(`/api/medias/${media.id}/video`));
      } else if (this.metadataLoaded) {
        // Same media id, but metadata (e.g. clip_start/clip_end) may have just
        // arrived via batch hydration. The video already loaded, so
        // (loadedmetadata) won't fire again; apply the clip window now.
        untracked(() => this.applyClipBounds());
      }
    });

    // Push volume changes onto the element. The element is read untracked so
    // this runs only when the volume actually changes, mirroring the old
    // `changes['volume']` guard; the initial volume is applied by
    // (loadedmetadata), as before.
    effect(() => {
      const vol = this.volume();
      const video = untracked(this.videoRef)?.nativeElement;
      if (video) video.volume = vol;
    });

    // Play/pause the element when the parent toggles `audioPlaying`. Untracked
    // element read for the same reason: a media swap must not re-trigger this
    // (the old code's `!changes['media']` guard) — the post-load sync happens
    // in (loadedmetadata) instead.
    effect(() => {
      this.audioPlaying();
      if (untracked(this.videoRef)) untracked(() => this.syncPlaybackState());
    });
  }

  ngOnDestroy(): void {
    this.stopClipEnforcement();
    const video = this.videoRef()?.nativeElement;
    if (video) {
      video.pause();
      video.removeAttribute('src');
      video.load();
    }
  }

  onPlay(): void {
    if (!this.audioPlaying()) {
      this.playingChanged.emit(true);
    }
  }

  onError(): void {
    this.videoError.set(true);
  }

  onPause(): void {
    if (this.audioPlaying()) {
      this.playingChanged.emit(false);
    }
  }

  togglePlayback(): void {
    const video = this.videoRef()?.nativeElement;
    if (!video) return;
    if (video.paused) {
      video.play().catch(() => {});
    } else {
      video.pause();
    }
  }

  adjustVolume(delta: number): void {
    const video = this.videoRef()?.nativeElement;
    if (!video) return;
    video.volume = Math.max(0, Math.min(1, video.volume + delta));
  }

  onLoadedMetadata(): void {
    const video = this.videoRef()?.nativeElement;
    if (!video) return;
    this.metadataLoaded = true;
    video.volume = this.volume();
    this.applyClipBounds();
    this.syncPlaybackState();
  }

  // Seek into the clip window and (re)start boundary enforcement when the media
  // carries clip extents; otherwise tear enforcement down. Safe to call both on
  // (loadedmetadata) and on later metadata-enrichment effect cycles.
  private applyClipBounds(): void {
    const video = this.videoRef()?.nativeElement;
    if (!video) return;

    const media = this.media();
    if (media.clip_start != null) {
      const clipStart = media.clip_start;
      const clipEnd = media.clip_end;
      // Snap into the window only when currently outside it. This handles the
      // initial seek and the case where the full video already started playing
      // before clip extents arrived, without yanking a video already looping
      // correctly within its window.
      if (video.currentTime < clipStart || (clipEnd != null && video.currentTime >= clipEnd)) {
        video.currentTime = clipStart;
      }
      this.startClipEnforcement();
    } else {
      this.stopClipEnforcement();
    }
  }

  private startClipEnforcement(): void {
    const media = this.media();
    if (media.clip_start == null || media.clip_end == null) {
      this.clipBounds = null;
      return;
    }
    // Arm enforcement; the actual boundary check runs in (timeupdate), which the
    // <video> element fires as playback advances (no timer while paused).
    this.clipBounds = { start: media.clip_start, end: media.clip_end };
  }

  private stopClipEnforcement(): void {
    this.clipBounds = null;
  }

  // Enforce the clip window as playback advances: when the current time leaves
  // [start, end), loop back to the window start. Driven by the element's
  // (timeupdate) event instead of a 100ms polling interval.
  onTimeUpdate(): void {
    const bounds = this.clipBounds;
    if (!bounds) return;
    const video = this.videoRef()?.nativeElement;
    if (!video || video.paused) return;
    if (video.currentTime >= bounds.end || video.currentTime < bounds.start) {
      video.currentTime = bounds.start;
    }
  }

  private syncPlaybackState(): void {
    const video = this.videoRef()?.nativeElement;
    if (!video) return;
    const audioPlaying = this.audioPlaying();
    if (audioPlaying && video.paused) {
      video.play().catch(() => {});
    } else if (!audioPlaying && !video.paused) {
      video.pause();
    }
  }
}
