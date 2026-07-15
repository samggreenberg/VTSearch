import { ChangeDetectionStrategy, Component, effect, inject, input, OnDestroy, OnInit, output, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subscription } from 'rxjs';
import { BrowseSelectionService } from '../../services/browse-selection.service';
import { MediaMetadataCacheService } from '../../services/media-metadata-cache.service';
import { ActiveContextService } from '../../services/active-context.service';
import { SettingsStateService } from '../../services/settings-state.service';
import { MediaTypeCapabilityService } from '../../services/media-type-capability.service';
import { ViewControlsComponent } from '../view-controls/view-controls.component';
import { NoFocusStealDirective } from '../../directives/no-focus-steal.directive';
import { IconComponent } from '../icon/icon.component';
import { CopyDetailButtonComponent } from '../copy-detail-button/copy-detail-button.component';
import { iconSizeToGoalWidth } from '../../utils/grid-icon-size';
import { applyClipWindow, clearClipWindow } from '../../utils/clip-window';
import type { NowPlaying } from '../browse-hover-preview/browse-hover-preview.component';

/** Ordering for the selected-item list. No detector confidence in browse, so
 *  the choices are recency (selection order), name, and id. */
type SelectionSortMode = 'time-desc' | 'time-asc' | 'name-asc' | 'name-desc' | 'id-asc';

/** How long (ms) the cursor must rest on an audio entry before its clip starts
 *  auditioning, so sweeping down a large multi-select list doesn't fire a burst
 *  of plays. Matches the canvas hover-preview's dwell (`AUDIO_DWELL_MS`). */
const AUDIO_DWELL_MS = 200;

interface SelectionEntry {
  id: number;
  name: string;
  /** Position in the selection's insertion order (recency proxy). */
  order: number;
}

/**
 * The VTSBrowse selection panel: a docked, always-visible list of every media
 * item currently selected on the canvas, with the same thumbnail-size control
 * as the Find view's labeled-item lists. Modeled on the Find "Good" panel — but
 * where that panel lists votes, this lists the live canvas selection, and
 * clicking an item removes it from the selection.
 *
 * Names and thumbnails are resolved lazily through
 * {@link MediaMetadataCacheService} (the browse view never loads the full media
 * list), so the panel renders ids immediately and fills in detail as the cache
 * hydrates. The thumbnail-size choice reuses the shared
 * ``grid_icon_size_right`` per-media-type setting, so it stays in step with the
 * Find right panel.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-browse-selection-panel',
  standalone: true,
  imports: [CommonModule, FormsModule, ViewControlsComponent, NoFocusStealDirective, IconComponent, CopyDetailButtonComponent],
  templateUrl: './browse-selection-panel.component.html',
  styleUrl: './browse-selection-panel.component.scss',
})
export class BrowseSelectionPanelComponent implements OnInit, OnDestroy {
  private selection = inject(BrowseSelectionService);
  private metadataCache = inject(MediaMetadataCacheService);
  private activeContext = inject(ActiveContextService);
  private settingsState = inject(SettingsStateService);
  private mediaTypeCaps = inject(MediaTypeCapabilityService);

  /** Active media type, used to resolve the per-type view-mode + size prefs. */
  readonly mediaType = input('');

  /**
   * Whether to offer the verify actions. Only meaningful when browsing a Find
   * run's positives (subset mode): "Verified Good" / "Verified Bad" mark the
   * selected items good/bad in the detector's labels *and* verify them, so they
   * leave the unverified set and drop from the browse. The browse view owns the
   * actual mutation + re-projection.
   */
  readonly canVerify = input(false);

  /** Emitted when the user marks the selection Verified Good. */
  readonly verifyGood = output<void>();

  /** Emitted when the user marks the selection Verified Bad. */
  readonly verifyBad = output<void>();

  /**
   * Emitted when the header checkbox asks for a select-all-in-view (the [ ]/[-]
   * → [x] click). The actual selection lives on the canvas — only it knows the
   * viewport and which bins sit fully inside it — so the browse view forwards
   * this to {@link BrowseCanvasComponent.selectAllInView}.
   */
  readonly selectAllInView = output<void>();

  /** Preview-audio volume (0–1), driven by the Browse toolbar's volume control,
   *  so hover audition here stays in lockstep with the canvas + bin-popup. */
  readonly volume = input(1);

  /** The clip now auditioning from a hovered audio entry (for the top-left
   *  now-playing indicator), or ``null`` once the hover clears — the same shared
   *  output the canvas and bin-popup drive. */
  readonly nowPlaying = output<NowPlaying | null>();

  // These are template-bound and written from the selection-refresh and
  // settings `effect()`s and the metadata-cache `version$` subscribe — none of
  // which schedule CD for a plain field under zoneless — so they are signals.
  // (`sortMode` stays plain: it is only written from the bound `(ngModelChange)`.)
  readonly count = signal(0);
  sortMode: SelectionSortMode = 'time-desc';
  readonly gridGoalWidth = signal(80);
  readonly sortedEntries = signal<SelectionEntry[]>([]);

  private ids: number[] = [];
  private gridIconSizeRightDict: Record<string, string> = {};
  private readonly thumbnailFailedUrls = new Set<string>();
  private readonly subs: Subscription[] = [];

  // --- Audio audition state machine (mirrors browse-hover-preview) -----------
  /** Never mounted in the DOM — audio here is heard, not shown; the top-left
   *  now-playing indicator is the only visual feedback that it's playing. */
  private readonly audioEl = new Audio();
  /** Waveform PNG of the clip currently auditioning, kept so the buffering
   *  listeners can re-emit the now-playing state without recomputing it. */
  private nowPlayingWaveUrl = '';
  private dwellTimer: ReturnType<typeof setTimeout> | null = null;
  private pendingMediaId: number | null = null;
  private playingMediaId: number | null = null;

  constructor() {
    // Keep the live element's volume in sync when the toolbar slider moves
    // mid-playback; starting a clip also seeds it.
    effect(() => {
      this.audioEl.volume = this.volume();
    });
    // Buffering feedback: while the clip is fetching/decoding (or stalls to
    // rebuffer), the widget shows a spinner; once it's actually sounding, the
    // spinner clears. Listeners are attached once to the reused element and
    // guarded by ``playingMediaId`` so stray events after a stop are ignored.
    this.audioEl.addEventListener('waiting', () => this.emitNowPlaying(true));
    this.audioEl.addEventListener('playing', () => this.emitNowPlaying(false));
    this.audioEl.addEventListener('canplay', () => this.emitNowPlaying(false));

    effect(() => {
      const settings = this.settingsState.settingsSignal();
      if (!settings) return;
      const sizeDict = settings.grid_icon_size_right;
      if (sizeDict && typeof sizeDict === 'object') {
        this.gridIconSizeRightDict = sizeDict as Record<string, string>;
      }
      this.applyViewPrefs();
    });
    // Rebuild the list whenever the selection changes. An effect on the signal
    // (rather than a `changed$` subscription) covers both the initial fill and
    // every later mutation, and schedules the refresh under zoneless from any
    // context. The first run (post-construction) does the initial refresh.
    effect(() => {
      this.selection.version();
      this.refreshSelection();
    });
  }

  ngOnInit(): void {
    this.subs.push(
      this.metadataCache.version$.subscribe(() => {
        this.sortedEntries.set(this.buildSortedEntries());
      }),
    );
    this.settingsState.load();
  }

  ngOnDestroy(): void {
    for (const sub of this.subs) sub.unsubscribe();
    this.cancelDwell();
    this.stopAudio();
  }

  private refreshSelection(): void {
    this.ids = this.selection.ids();
    this.count.set(this.ids.length);
    this.metadataCache.ensureLoaded(this.ids);
    this.sortedEntries.set(this.buildSortedEntries());
  }

  private applyViewPrefs(): void {
    const mediaType = this.mediaType();
    if (!mediaType) return;
    this.gridGoalWidth.set(iconSizeToGoalWidth(this.gridIconSizeRightDict[mediaType] ?? 'M'));
  }

  private buildSortedEntries(): SelectionEntry[] {
    const entries = this.ids.map((id, order) => ({
      id,
      name: this.lookupName(id),
      order,
    }));
    return this.sortEntries(entries);
  }

  private lookupName(id: number): string {
    return this.metadataCache.get(id)?.filename || `Clip #${id}`;
  }

  private sortEntries(entries: SelectionEntry[]): SelectionEntry[] {
    const sorted = [...entries];
    switch (this.sortMode) {
      case 'time-desc':
        sorted.sort((a, b) => b.order - a.order);
        break;
      case 'time-asc':
        sorted.sort((a, b) => a.order - b.order);
        break;
      case 'name-asc':
        sorted.sort((a, b) => a.name.localeCompare(b.name));
        break;
      case 'name-desc':
        sorted.sort((a, b) => b.name.localeCompare(a.name));
        break;
      case 'id-asc':
        sorted.sort((a, b) => a.id - b.id);
        break;
    }
    return sorted;
  }

  onSortChange(mode: SelectionSortMode): void {
    this.sortMode = mode;
    this.sortedEntries.set(this.buildSortedEntries());
  }

  /**
   * Tri-state of the header select-all checkbox: `none` when nothing is
   * selected ([ ]), `all` when a select-all-in-view is active and untouched
   * ([x] — see {@link BrowseSelectionService.allSelected}), or `some` for any
   * other, partial selection ([-]). Both reads are signals, so the template
   * binding re-evaluates when either the count or the latch changes.
   */
  checkState(): 'none' | 'some' | 'all' {
    if (this.count() === 0) return 'none';
    return this.selection.allSelected() ? 'all' : 'some';
  }

  /**
   * The header checkbox click cycle: from [ ] or [-], select everything in view
   * ([x]); from [x], clear the whole selection ([ ]). Select-all is the canvas's
   * job (it owns the viewport), so we emit for the view to forward; clearing is
   * a direct selection mutation.
   */
  onToggleCheckbox(): void {
    if (this.checkState() === 'all') {
      this.selection.clear();
    } else {
      this.selectAllInView.emit();
    }
  }

  /** Ask the browse view to mark the selected items Verified Good and drop them. */
  onVerifyGood(): void {
    if (this.count() === 0) return;
    this.verifyGood.emit();
  }

  /** Ask the browse view to mark the selected items Verified Bad and drop them. */
  onVerifyBad(): void {
    if (this.count() === 0) return;
    this.verifyBad.emit();
  }

  /** Clicking a selected item drops it from the selection. */
  onEntryClick(id: number): void {
    // The entry is about to leave the list (its element unmounts, so no
    // mouseleave will fire): silence it now if it's the one auditioning.
    if (this.playingMediaId === id || this.pendingMediaId === id) this.stopAudio();
    this.selection.remove(id);
  }

  onEntryKeydown(event: KeyboardEvent, id: number): void {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      this.onEntryClick(id);
    }
  }

  // --- Audio audition on hover (audio only) ----------------------------------

  /**
   * Cursor entered an entry: for audio datasets, arm the dwell debounce so the
   * clip starts auditioning once the cursor settles, matching the canvas
   * hover-preview. A no-op for every other media type — Browse only plays sound
   * for audio.
   */
  onEntryEnter(id: number): void {
    if (this.mediaType() !== 'audio') return;
    this.scheduleAudio(id);
  }

  /** Cursor left an entry: cancel a pending audition and stop anything playing.
   *  There is no pane to bridge the cursor onto, so hover-off silences at once
   *  (the same as the canvas path). */
  onEntryLeave(): void {
    this.cancelDwell();
    this.stopAudio();
  }

  private scheduleAudio(mediaId: number): void {
    // Already auditioning this exact clip: keep it (don't restart).
    if (this.playingMediaId === mediaId) return;

    // Debounce: (re)arm the dwell so a sweep down the list keeps resetting it and
    // only the entry the cursor settles on actually plays.
    this.pendingMediaId = mediaId;
    this.cancelDwell();
    this.dwellTimer = setTimeout(() => {
      this.dwellTimer = null;
      const id = this.pendingMediaId;
      this.pendingMediaId = null;
      if (id != null) this.playAudio(id);
    }, AUDIO_DWELL_MS);
  }

  private playAudio(mediaId: number): void {
    this.playingMediaId = mediaId;
    const waveUrl = this.activeContext.mediaUrl(`/api/medias/${mediaId}/thumbnail`);
    this.nowPlayingWaveUrl = waveUrl;
    // Hydrate clip extents (clip_start/clip_end) so windowed clips loop within
    // their window; applyClipWindow reads them lazily as they land.
    this.metadataCache.ensureLoaded([mediaId]);
    this.audioEl.volume = this.volume();
    applyClipWindow(this.audioEl, () => this.metadataCache.get(mediaId));
    this.audioEl.src = this.activeContext.mediaUrl(`/api/medias/${mediaId}/audio`);
    this.audioEl.load();
    this.audioEl.play().catch(() => {});
    // Starts loading: show the spinner until a ``playing``/``canplay`` event
    // (wired in the constructor) clears it.
    this.nowPlaying.emit({ mediaId, waveUrl, loading: true });
  }

  /** Re-emit the now-playing state with a fresh ``loading`` flag, from the
   *  buffering listeners. A no-op once nothing is auditioning, so events that
   *  fire after a stop don't resurrect the widget. */
  private emitNowPlaying(loading: boolean): void {
    if (this.playingMediaId == null) return;
    this.nowPlaying.emit({ mediaId: this.playingMediaId, waveUrl: this.nowPlayingWaveUrl, loading });
  }

  private stopAudio(): void {
    this.pendingMediaId = null;
    if (this.playingMediaId == null) return;
    this.playingMediaId = null;
    clearClipWindow(this.audioEl);
    this.audioEl.pause();
    this.audioEl.currentTime = 0;
    this.nowPlaying.emit(null);
  }

  private cancelDwell(): void {
    if (this.dwellTimer) {
      clearTimeout(this.dwellTimer);
      this.dwellTimer = null;
    }
  }

  // --- Thumbnails (mirrors the Find label-list treatment) ------------------

  hasThumbnailUrl(id: number): boolean {
    const url = this.thumbnailUrl(id);
    if (url && this.thumbnailFailedUrls.has(url)) return false;
    const media = this.metadataCache.get(id);
    return !!media && this.mediaTypeCaps.usesThumbnails(media.media_type);
  }

  thumbnailUrl(id: number): string {
    return this.activeContext.mediaUrl(`/api/medias/${id}/thumbnail`);
  }

  /** True when this item's thumbnail is an audio waveform — a theme-agnostic
   *  alpha-mask PNG (issue #2369) tinted via a CSS mask, not a plain <img>. */
  isAudioThumbnail(id: number): boolean {
    return this.metadataCache.get(id)?.media_type === 'audio';
  }

  onThumbnailError(url: string): void {
    if (url) this.thumbnailFailedUrls.add(url);
  }

  placeholderIcon(id: number): string | null {
    if (this.hasThumbnailUrl(id)) return null;
    const media = this.metadataCache.get(id);
    if (!media) return '□';
    if (media.media_type === 'audio') return '♫';
    if (media.media_type === 'text') return '¶';
    return '□';
  }

  trackById(_index: number, entry: SelectionEntry): number {
    return entry.id;
  }
}
