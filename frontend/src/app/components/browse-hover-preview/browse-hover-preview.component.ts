import { ChangeDetectionStrategy, Component, effect, inject, input, OnDestroy, output, signal, untracked } from '@angular/core';

import { ActiveContextService } from '../../services/active-context.service';
import { MediaMetadataCacheService } from '../../services/media-metadata-cache.service';
import { BrowseAudioAudition, type NowPlaying } from '../../utils/browse-audio-audition';
import type { HexHoverEvent } from '../browse-canvas/browse-canvas.component';

/**
 * The VTSBrowse hover preview.
 *
 * For **audio** it starts auditioning the hovered bin's clip after a short
 * dwell. There is no on-canvas UI of its own: the bin itself lifts (the
 * canvas's hover-enlarge, shared with every thumbnail type — see
 * `usesThumbnails()`), and the now-playing waveform + volume control anchored
 * at the top-left of the canvas (`browse-view.component`) is the only visible
 * feedback that sound is playing. Hover clearing stops it immediately.
 *
 * For **text** it shows the small paragraph popup; **image/video** paint
 * their thumbnail on the tile, so nothing pops up.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-browse-hover-preview',
  standalone: true,
  imports: [],
  templateUrl: './browse-hover-preview.component.html',
  styleUrl: './browse-hover-preview.component.scss',
})
export class BrowseHoverPreviewComponent implements OnDestroy {
  private activeContext = inject(ActiveContextService);
  private metadataCache = inject(MediaMetadataCacheService);

  readonly hover = input<HexHoverEvent | null>(null);
  readonly mediaType = input('');
  /** Preview-audio volume (0–1), driven by the Browse toolbar's volume control. */
  readonly volume = input(1);
  /** The clip now auditioning (for the top-left now-playing indicator), or
   *  ``null`` once the hover clears. */
  readonly nowPlaying = output<NowPlaying | null>();

  /** The dwell debounce, buffering tri-state and playhead sweep, shared with the
   *  selection panel and the bin popup. */
  private readonly audition = new BrowseAudioAudition({
    mediaUrl: (path) => this.activeContext.mediaUrl(path),
    lookup: (id) => this.metadataCache.get(id),
    ensureLoaded: (id) => this.metadataCache.ensureLoaded([id]),
    emit: (state) => this.nowPlaying.emit(state),
  });

  constructor() {
    // The hover input drives the whole preview: show over a cell, hide on
    // null. (The old ngOnChanges arm — signal inputs don't fire it.) The body
    // runs untracked so its mediaType/metadata reads don't widen the trigger
    // beyond the hover itself.
    effect(() => {
      const hover = this.hover();
      untracked(() => (hover ? this.show(hover) : this.hide()));
    });
    // Keep the live element's volume in sync when the toolbar slider moves
    // mid-playback; starting a clip also seeds it.
    effect(() => {
      this.audition.setVolume(this.volume());
    });
  }

  // --- Text / count popup (text + any other non-thumbnail type) --------------
  visible = false;
  left = 0;
  top = 0;
  count = 0;
  // Written from the async paragraph `fetch().then()` continuation, so a signal
  // so the late write repaints the popup body under zoneless.
  readonly textContent = signal('');
  private textLoadAbort: AbortController | null = null;

  ngOnDestroy(): void {
    this.audition.destroy();
    this.textLoadAbort?.abort();
  }

  private show(event: HexHoverEvent): void {
    const mediaType = this.mediaType();
    // Image and video paint their thumbnail directly onto the tile; nothing
    // pops up on hover.
    if (mediaType === 'image' || mediaType === 'video') {
      this.hide();
      return;
    }

    if (mediaType === 'audio') {
      this.audition.hover(event.cell.rep_id);
      return;
    }

    // text / other: the lightweight text-or-count popup (immediate, no dwell).
    this.visible = true;
    this.left = event.screenX + 16;
    this.top = event.screenY - 8;
    this.count = event.cell.count;
    const rep = event.cell.rep_id;
    if (mediaType === 'text') {
      this.loadText(rep);
    } else {
      this.textContent.set(`Item #${rep}`);
    }
  }

  private hide(): void {
    // Text popup: hide immediately.
    this.visible = false;
    this.textLoadAbort?.abort();
    this.textLoadAbort = null;
    this.textContent.set('');

    // Audio: cancel a pending audition and stop anything already playing. There
    // is no panel to bridge the cursor onto, so hover-off silences it at once.
    this.audition.stop();
  }

  private loadText(mediaId: number): void {
    this.textContent.set('Loading…');
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
          this.textContent.set(text.length > 300 ? text.slice(0, 300) + '…' : text);
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
