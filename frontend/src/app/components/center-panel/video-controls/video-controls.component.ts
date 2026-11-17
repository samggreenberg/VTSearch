import { ChangeDetectionStrategy, Component, computed, effect, input, signal, untracked } from '@angular/core';

import { ClipBounds } from '../../../utils/clip-window';
import { IconComponent } from '../../icon/icon.component';

/** What the bar follows on its element: each of these can move what it shows. */
const MEDIA_EVENTS = [
  'play',
  'pause',
  'ended',
  'timeupdate',
  'seeking',
  'seeked',
  'durationchange',
  'loadedmetadata',
  'emptied',
  'volumechange',
] as const;

/**
 * The control bar a blurred video plays under (#5698): play / pause, a seek
 * bar with the time, and mute with a volume slider.
 *
 * A CSS blur on a `<video>` blurs its native controls with the picture, so
 * while the video is blurred `vt-video-player` hides them and draws this bar
 * outside the blur instead. It drives the element directly, as the native
 * controls do, so the player hears the same `play` / `pause` / `volumechange`
 * events either way. The seek bar spans the clip window when the item has one,
 * since playback is held inside it.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-video-controls',
  standalone: true,
  imports: [IconComponent],
  host: { role: 'group', 'aria-label': 'Video controls' },
  templateUrl: './video-controls.component.html',
  styleUrl: './video-controls.component.scss',
})
export class VideoControlsComponent {
  /** The element the bar drives. */
  readonly video = input.required<HTMLVideoElement>();
  /** The window playback is held in, or null for the whole file. */
  readonly clip = input<ClipBounds | null>(null);

  // The element's state, copied on each of its MEDIA_EVENTS.
  readonly paused = signal(true);
  readonly time = signal(0);
  private readonly duration = signal(Number.NaN);
  readonly volume = signal(1);
  readonly muted = signal(false);

  /** What the seek bar spans: the clip window, or the whole file once its length is known. */
  readonly range = computed<ClipBounds | null>(() => {
    const clip = this.clip();
    if (clip) return clip;
    const duration = this.duration();
    return Number.isFinite(duration) && duration > 0 ? { start: 0, end: duration } : null;
  });

  /** "0:03 / 0:10", measured from the start of {@link range}. */
  readonly timeLabel = computed(() => {
    const range = this.range();
    if (!range) return '0:00 / 0:00';
    const span = range.end - range.start;
    const at = Math.max(0, Math.min(span, this.time() - range.start));
    return `${formatTime(at)} / ${formatTime(span)}`;
  });

  /** The speaker shows muted when the element is muted or at no volume. */
  readonly silent = computed(() => this.muted() || this.volume() === 0);

  constructor() {
    effect((onCleanup) => {
      const video = this.video();
      const sync = (): void => {
        this.paused.set(video.paused);
        this.time.set(video.currentTime);
        this.duration.set(video.duration);
        this.volume.set(video.volume);
        this.muted.set(video.muted);
      };
      untracked(sync);
      for (const type of MEDIA_EVENTS) video.addEventListener(type, sync);
      onCleanup(() => {
        for (const type of MEDIA_EVENTS) video.removeEventListener(type, sync);
      });
    });
  }

  togglePlay(): void {
    const video = this.video();
    if (video.paused) {
      video.play().catch(() => {});
    } else {
      video.pause();
    }
  }

  onSeek(event: Event): void {
    const video = this.video();
    video.currentTime = Number((event.target as HTMLInputElement).value);
    this.time.set(video.currentTime);
  }

  toggleMute(): void {
    const video = this.video();
    video.muted = !video.muted;
  }

  onVolume(event: Event): void {
    const video = this.video();
    const volume = Number((event.target as HTMLInputElement).value);
    video.volume = volume;
    // Raising the slider is asking to hear it, as on the native controls.
    if (volume > 0) video.muted = false;
  }
}

/** *seconds* as `m:ss`, or `h:mm:ss` from an hour. */
export function formatTime(seconds: number): string {
  const whole = Math.max(0, Math.floor(Number.isFinite(seconds) ? seconds : 0));
  const h = Math.floor(whole / 3600);
  const m = Math.floor((whole % 3600) / 60);
  const s = String(whole % 60).padStart(2, '0');
  return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${s}` : `${m}:${s}`;
}
