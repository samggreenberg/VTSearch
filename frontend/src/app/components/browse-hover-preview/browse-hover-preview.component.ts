import { ChangeDetectionStrategy, Component, effect, ElementRef, inject, input, OnChanges, OnDestroy, signal, SimpleChanges, ViewChild } from '@angular/core';

import { ActiveContextService } from '../../services/active-context.service';
import { MediaMetadataCacheService } from '../../services/media-metadata-cache.service';
import { applyClipWindow, clearClipWindow } from '../../utils/clip-window';
import type { HexHoverEvent } from '../browse-canvas/browse-canvas.component';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-browse-hover-preview',
  standalone: true,
  imports: [],
  templateUrl: './browse-hover-preview.component.html',
  styleUrl: './browse-hover-preview.component.scss',
})
export class BrowseHoverPreviewComponent implements OnChanges, OnDestroy {
  private activeContext = inject(ActiveContextService);
  private metadataCache = inject(MediaMetadataCacheService);

  readonly hover = input<HexHoverEvent | null>(null);
  readonly mediaType = input('');
  /** Preview-audio volume (0–1), driven by the Browse toolbar's volume control. */
  readonly volume = input(1);
  @ViewChild('audioEl') audioRef?: ElementRef<HTMLAudioElement>;

  constructor() {
    // Keep the live element in sync when the toolbar slider moves mid-playback;
    // ``playAudio`` also seeds it so a freshly-started clip honours the level.
    effect(() => {
      const el = this.audioRef?.nativeElement;
      if (el) el.volume = this.volume();
    });
  }

  visible = false;
  left = 0;
  top = 0;
  audioSrc = '';
  // Written both from `ngOnChanges` (a CD context) and from the async text
  // `fetch().then()` continuation, so a signal so the late write repaints the
  // popup body under zoneless.
  readonly textContent = signal('');
  count = 0;
  private textLoadAbort: AbortController | null = null;

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['hover']) {
      const hover = this.hover();
      if (hover) {
        this.show(hover);
      } else {
        this.hide();
      }
    }
  }

  ngOnDestroy(): void {
    this.stopAudio();
    this.textLoadAbort?.abort();
  }

  private show(event: HexHoverEvent): void {
    const representativeId = event.cell.rep_id;

    // Image and video paint their thumbnail directly onto the hex (see
    // browse-canvas); nothing happens on hover, so suppress the pop-up.
    const mediaType = this.mediaType();
    if (mediaType === 'image' || mediaType === 'video') {
      this.hide();
      return;
    }

    this.visible = true;
    this.left = event.screenX + 16;
    this.top = event.screenY - 8;
    this.count = event.cell.count;

    switch (mediaType) {
      case 'audio':
        this.textContent.set('');
        this.playAudio(representativeId);
        break;
      case 'text':
        this.stopAudio();
        this.loadText(representativeId);
        break;
      default:
        this.stopAudio();
        this.textContent.set(`Item #${representativeId}`);
    }
  }

  private hide(): void {
    this.visible = false;
    this.stopAudio();
    this.textLoadAbort?.abort();
    this.textLoadAbort = null;
    this.textContent.set('');
  }

  private playAudio(mediaId: number): void {
    const src = this.activeContext.mediaUrl(`/api/medias/${mediaId}/audio`);
    if (this.audioSrc === src) return;
    this.audioSrc = src;

    // Kick off metadata hydration so the clip window (clip_start/clip_end) is
    // available by the time the audio's loadedmetadata fires; the fetch runs
    // while the element loads.
    this.metadataCache.ensureLoaded([mediaId]);

    setTimeout(() => {
      const el = this.audioRef?.nativeElement;
      if (!el) return;
      el.volume = this.volume();
      // Windowed archive-member clips serve the whole file: seek to clip_start
      // and loop within [clip_start, clip_end] instead of playing it all.
      applyClipWindow(el, () => this.metadataCache.get(mediaId));
      el.load();
      el.play().catch(() => {});
    });
  }

  private stopAudio(): void {
    const el = this.audioRef?.nativeElement;
    if (el) {
      clearClipWindow(el);
      el.pause();
      el.currentTime = 0;
    }
    this.audioSrc = '';
  }

  private loadText(mediaId: number): void {
    this.textContent.set('Loading...');
    // Cancel any in-flight text load so a slow earlier response can't clobber
    // the preview after the cursor has moved to another hex.
    this.textLoadAbort?.abort();
    const abort = new AbortController();
    this.textLoadAbort = abort;
    const url = this.activeContext.mediaUrl(`/api/medias/${mediaId}/paragraph`);
    fetch(url, { signal: abort.signal })
      .then((r) => r.json())
      .then((data) => {
        if (this.hover()?.cell.rep_id === mediaId) {
          const text: string = data.content || '';
          this.textContent.set(text.length > 300 ? text.slice(0, 300) + '...' : text);
        }
      })
      .catch((err) => {
        if (err?.name === 'AbortError') return;
        if (this.hover()?.cell.rep_id === mediaId) {
          this.textContent.set(`Item #${mediaId}`);
        }
      });
  }
}
