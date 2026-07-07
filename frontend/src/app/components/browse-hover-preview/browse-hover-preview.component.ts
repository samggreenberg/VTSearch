import { ChangeDetectionStrategy, Component, effect, ElementRef, inject, input, OnChanges, OnDestroy, signal, SimpleChanges, ViewChild } from '@angular/core';

import { ActiveContextService } from '../../services/active-context.service';
import { MediaMetadataCacheService } from '../../services/media-metadata-cache.service';
import { applyClipWindow, clearClipWindow } from '../../utils/clip-window';
import type { HexHoverEvent } from '../browse-canvas/browse-canvas.component';

/** The open anchored audio player (``null`` when none is shown). */
interface AudioPanel {
  mediaId: number;
  /** The bin's waveform PNG — the same thumbnail painted on the tile. */
  waveUrl: string;
  audioSrc: string;
  left: number;
  top: number;
  count: number;
}

/** How long (ms) the cursor must rest on an audio bin before its player opens,
 *  so sweeping across bins doesn't spawn a burst of players. */
const AUDIO_DWELL_MS = 200;
/** Grace (ms) after the bin-hover clears before the player closes, giving the
 *  cursor time to travel from the tile onto the player without it vanishing —
 *  the classic hover-bridge. */
const AUDIO_HIDE_GRACE_MS = 140;

/**
 * The VTSBrowse hover preview.
 *
 * For **audio** it opens an anchored player next to the hovered bin — the bin's
 * waveform plus native ``<audio>`` controls (play/pause, volume, scrubber /
 * play-point) — so the audition is visible and controllable instead of "sound
 * from nowhere". The player opens on a short dwell (debounced against sweeps)
 * and stays open while the cursor is on it (so the controls are reachable).
 *
 * For **text** it shows the small paragraph popup; **image/video** paint their
 * thumbnail on the tile, so nothing pops up.
 */
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
    // Keep the live element's volume in sync when the toolbar slider moves
    // mid-playback; opening a player also seeds it so a fresh clip honours it.
    effect(() => {
      const el = this.audioRef?.nativeElement;
      if (el) el.volume = this.volume();
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

  // --- Anchored audio player -------------------------------------------------
  /** The open audio player, or ``null``. A signal so the dwell / hide timers
   *  (which fire outside a CD context) repaint under zoneless. */
  readonly player = signal<AudioPanel | null>(null);
  private dwellTimer: ReturnType<typeof setTimeout> | null = null;
  private hideTimer: ReturnType<typeof setTimeout> | null = null;
  private pointerInPanel = false;
  private pendingTarget: Omit<AudioPanel, 'waveUrl' | 'audioSrc'> | null = null;

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
    this.clearTimers();
    this.stopAudio();
    this.player.set(null);
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
      this.scheduleAudioPlayer(event);
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

    // Audio player: cancel a pending open, then close after a grace period so
    // the cursor can bridge from the tile onto the player. If it's already on
    // the player, leave it open — the panel's mouseleave closes it instead.
    this.cancelDwell();
    this.pendingTarget = null;
    if (this.player() && !this.pointerInPanel) {
      this.scheduleHide();
    }
  }

  // --- Audio player state machine --------------------------------------------

  private scheduleAudioPlayer(event: HexHoverEvent): void {
    this.cancelHide();
    const target = {
      mediaId: event.cell.rep_id,
      left: event.screenX + 16,
      top: event.screenY - 8,
      count: event.cell.count,
    };
    // Already showing this exact clip: keep it (don't restart the audition).
    if (this.player()?.mediaId === target.mediaId) return;

    // Debounce: (re)arm the dwell so a sweep across bins keeps resetting it and
    // only the bin the cursor settles on opens.
    this.pendingTarget = target;
    this.cancelDwell();
    this.dwellTimer = setTimeout(() => {
      this.dwellTimer = null;
      const t = this.pendingTarget;
      this.pendingTarget = null;
      if (t) this.openAudioPlayer(t);
    }, AUDIO_DWELL_MS);
  }

  private openAudioPlayer(target: Omit<AudioPanel, 'waveUrl' | 'audioSrc'>): void {
    // Replacing an open player: stop the old element before its src changes.
    this.stopAudio();
    const mediaId = target.mediaId;
    this.player.set({
      ...target,
      waveUrl: this.activeContext.mediaUrl(`/api/medias/${mediaId}/thumbnail`),
      audioSrc: this.activeContext.mediaUrl(`/api/medias/${mediaId}/audio`),
    });
    // Hydrate clip extents (clip_start/clip_end) so windowed clips loop within
    // their window; applyClipWindow reads them lazily as they land.
    this.metadataCache.ensureLoaded([mediaId]);
    // The <audio> mounts with the player signal; grab it after the render the
    // signal write schedules, then start the audition.
    setTimeout(() => {
      const el = this.audioRef?.nativeElement;
      if (!el || this.player()?.mediaId !== mediaId) return;
      el.volume = this.volume();
      applyClipWindow(el, () => this.metadataCache.get(mediaId));
      el.load();
      el.play().catch(() => {});
    });
  }

  /** The cursor entered the player panel — keep it open and reachable. */
  onPanelEnter(): void {
    this.pointerInPanel = true;
    this.cancelHide();
  }

  /** The cursor left the player panel — close it (unless it lands back on a bin,
   *  whose hover event will reopen the right one). */
  onPanelLeave(): void {
    this.pointerInPanel = false;
    this.scheduleHide();
  }

  private scheduleHide(): void {
    this.cancelHide();
    this.hideTimer = setTimeout(() => {
      this.hideTimer = null;
      if (this.pointerInPanel) return;
      this.stopAudio();
      this.player.set(null);
    }, AUDIO_HIDE_GRACE_MS);
  }

  private cancelDwell(): void {
    if (this.dwellTimer) {
      clearTimeout(this.dwellTimer);
      this.dwellTimer = null;
    }
  }

  private cancelHide(): void {
    if (this.hideTimer) {
      clearTimeout(this.hideTimer);
      this.hideTimer = null;
    }
  }

  private clearTimers(): void {
    this.cancelDwell();
    this.cancelHide();
  }

  private stopAudio(): void {
    const el = this.audioRef?.nativeElement;
    if (el) {
      clearClipWindow(el);
      el.pause();
      el.currentTime = 0;
    }
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
