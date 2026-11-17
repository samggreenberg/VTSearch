import { ChangeDetectionStrategy, Component, computed, effect, ElementRef, inject, input, OnDestroy, output, signal, untracked, viewChild } from '@angular/core';

import { SharedResizeObserver } from '../../../directives/blur.directive';
import { Media, PayloadVariant } from '../../../models/api.models';
import { ActiveContextService } from '../../../services/active-context.service';
import { ActiveDatasetService } from '../../../services/active-dataset.service';
import { BLUR_RADIUS_MAX_PX, blurRadiusFor } from '../../../utils/blur-radius';
import { clipBoxViewBox } from '../../../utils/clip-box';
import { armClipWindow, ClipBounds, enforceClipWindow } from '../../../utils/clip-window';
import { focusedBlur } from '../../../utils/focused-blur';
import { VideoControlsComponent } from '../video-controls/video-controls.component';

/**
 * The focused video. A video the server rules sensitive plays blurred
 * (#5698), as a focused image is drawn blurred: with the same peek, timed
 * here, and the same Blur / Unblur button, which the centre panel puts under
 * it. A blur on the element blurs its native controls with it, so while the
 * video is blurred they are hidden and `vt-video-controls` is drawn over the
 * bottom of the frame, outside the blur, in their place.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-video-player',
  standalone: true,
  imports: [VideoControlsComponent],
  templateUrl: './video-player.component.html',
  styleUrl: './video-player.component.scss',
})
export class VideoPlayerComponent implements OnDestroy {
  private activeContext = inject(ActiveContextService);
  private readonly activeDataset = inject(ActiveDatasetService);
  private readonly sizes = inject(SharedResizeObserver);

  readonly media = input.required<Media>();
  /**
   * Which payload to show: `''` (canonical) or `'original'` (the pre-clean
   * snapshot of an item a cleaner rewrote at load time).  Driven by the
   * parent's Clean/Original toggle.
   */
  readonly variant = input<PayloadVariant>('');
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
  /** The current source's first frame is on screen ((loadeddata)): the peek waits for it. */
  private readonly frameShown = signal(false);

  /**
   * The item's blur (`utils/focused-blur.ts`), in the active dataset. The peek
   * waits for the first frame, and starts over for each new video, so one
   * voted past under a resting pointer does not arrive sharp.
   */
  private readonly itemBlur = focusedBlur({
    itemId: () => this.media().id,
    datasetId: () => this.activeDataset.activeId(),
    ready: () => this.frameShown(),
    shown: () => this.videoSrc(),
  });
  /** Whether the Blur / Unblur button has an item to act on. */
  readonly blurrable = this.itemBlur.blurrable;
  /** Whether the video plays blurred (see `FocusedBlur.blurred`). */
  readonly blurred = this.itemBlur.blurred;
  /** The pointer has rested on the blurred video for a second: its blur fades out (`.video-peek`). */
  readonly peeking = this.itemBlur.peeking;

  /** The radius for the element's last measured size, or null before it was measured. */
  private readonly blurRadius = signal<number | null>(null);
  /**
   * The blur's radius while {@link blurred}: the shared share of the video's
   * shorter side, as every image gets, and the most any site gets before the
   * element is measured.
   */
  readonly blurRadiusCss = computed<string | null>(() => {
    if (!this.blurred()) return null;
    return `${this.blurRadius() ?? BLUR_RADIUS_MAX_PX}px`;
  });

  /**
   * The window playback is held in, for the blurred video's seek bar: the
   * same bounds {@link armClipWindow} enforces, or null for the whole file.
   */
  readonly clipWindow = computed<ClipBounds | null>(() => {
    const { clip_start: start, clip_end: end } = this.media();
    return start != null && end != null ? { start, end } : null;
  });

  // The element's natural frame size, once (loadedmetadata) reports it for the
  // current videoSrc; null before that. The unit's clip_box is scaled into it
  // and clamped against it, so no crop can be computed until it is known.
  private readonly frameSize = signal<{ width: number; height: number } | null>(null);

  /**
   * The unit's letterbox crop as an `object-view-box` value, or null when
   * none applies. A letterbox cleaner records the crop as `clip_box` rather
   * than re-encoding the file, and the embedder and grid thumbnail honour it;
   * this makes the player show the same picture instead of the full padded
   * frame. Recomputed when batch hydration delivers `clip_box` after load.
   * `clip_box_frame` is the frame size the box was measured on, which differs
   * from the natural size on an anamorphic video.
   */
  readonly viewBox = computed(() => {
    const size = this.frameSize();
    if (!size) return null;
    const media = this.media();
    return clipBoxViewBox(media.clip_box, size.width, size.height, media.clip_box_frame);
  });

  // Active clip window bounds while enforcement is on, else null. Enforcement is
  // driven by the <video> element's (timeupdate) event rather than a polling
  // timer, so it only runs while the clip is actually playing and progressing.
  private clipBounds: ClipBounds | null = null;
  // See ImageViewerComponent.lastMediaId; guards against metadata-enrichment
  // effect cycles rebuilding videoSrc (and yanking playback) for the
  // same id.
  private lastMediaId: number | null = null;
  // Same guard for the payload variant: a Clean/Original flip keeps the media
  // id, so the effect has to notice the variant changed to reload the source.
  private lastVariant: PayloadVariant = '';
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
      const variant = this.variant();
      if (media.id !== this.lastMediaId || variant !== this.lastVariant) {
        this.lastMediaId = media.id;
        this.lastVariant = variant;
        this.videoError.set(false);
        this.frameShown.set(false);
        this.metadataLoaded = false;
        this.frameSize.set(null);
        this.stopClipEnforcement();
        this.videoSrc.set(this.activeContext.mediaUrl(`/api/medias/${media.id}/video`, { variant }));
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

    // Measure the element only while it is blurred: a sharp video has no
    // radius to keep current.
    effect((onCleanup) => {
      const video = this.videoRef()?.nativeElement;
      if (!video || !this.blurred()) return;
      onCleanup(
        untracked(() =>
          this.sizes.observe(video, (width, height) => this.blurRadius.set(blurRadiusFor(width, height))),
        ),
      );
    });

    // A blurred video plays only here, under its blur: picture-in-picture
    // would show it sharp, and fullscreen would show it without its controls.
    effect(() => {
      const video = this.videoRef()?.nativeElement;
      if (!video || !this.blurred()) return;
      if (document.pictureInPictureElement === video) document.exitPictureInPicture().catch(() => {});
      if (document.fullscreenElement === video) document.exitFullscreen().catch(() => {});
    });
  }

  onPointerEnter(): void {
    this.itemBlur.pointerEnter();
  }

  onPointerLeave(): void {
    this.itemBlur.pointerLeave();
  }

  /**
   * The Blur / Unblur button and its `b` key: flip the video's state for
   * good, in the active dataset, so every view of it (its thumbnail in the
   * list among them) follows and the choice outlasts the blur rules.
   */
  toggleBlur(): void {
    this.itemBlur.toggle();
  }

  onLoadedData(): void {
    this.frameShown.set(true);
  }

  /**
   * A click on the picture plays or pauses it, as it does on the native
   * controls, while those are hidden for the blur. With them shown, the
   * element does it itself.
   */
  onVideoClick(): void {
    if (this.blurred()) this.togglePlayback();
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
    this.frameSize.set({ width: video.videoWidth, height: video.videoHeight });
    video.volume = this.volume();
    this.applyClipBounds();
    this.syncPlaybackState();
  }

  // Seek into the clip window and (re)arm boundary enforcement when the media
  // carries clip extents; otherwise tear enforcement down. Safe to call both on
  // (loadedmetadata) and on later metadata-enrichment effect cycles.
  private applyClipBounds(): void {
    const video = this.videoRef()?.nativeElement;
    if (!video) return;
    this.clipBounds = armClipWindow(video, this.media());
  }

  private stopClipEnforcement(): void {
    this.clipBounds = null;
  }

  onTimeUpdate(): void {
    const video = this.videoRef()?.nativeElement;
    if (video) enforceClipWindow(video, this.clipBounds);
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
