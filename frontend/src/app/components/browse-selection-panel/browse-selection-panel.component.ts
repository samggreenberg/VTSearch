import { ChangeDetectionStrategy, Component, effect, inject, input, OnDestroy, OnInit, output, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subscription } from 'rxjs';
import { BrowseSelectionService } from '../../services/browse-selection.service';
import { MediaMetadataCacheService } from '../../services/media-metadata-cache.service';
import { ActiveContextService } from '../../services/active-context.service';
import { SettingsStateService } from '../../services/settings-state.service';
import { ViewControlsComponent } from '../view-controls/view-controls.component';
import { NoFocusStealDirective } from '../../directives/no-focus-steal.directive';
import { IconComponent } from '../icon/icon.component';
import { iconSizeToGoalWidth } from '../../utils/grid-icon-size';

/** Ordering for the selected-item list. No detector confidence in browse, so
 *  the choices are recency (selection order), name, and id. */
type SelectionSortMode = 'time-desc' | 'time-asc' | 'name-asc' | 'name-desc' | 'id-asc';

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
  imports: [CommonModule, FormsModule, ViewControlsComponent, NoFocusStealDirective, IconComponent],
  templateUrl: './browse-selection-panel.component.html',
  styleUrl: './browse-selection-panel.component.scss',
})
export class BrowseSelectionPanelComponent implements OnInit, OnDestroy {
  private selection = inject(BrowseSelectionService);
  private metadataCache = inject(MediaMetadataCacheService);
  private activeContext = inject(ActiveContextService);
  private settingsState = inject(SettingsStateService);

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

  constructor() {
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
    this.selection.remove(id);
  }

  onEntryKeydown(event: KeyboardEvent, id: number): void {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      this.selection.remove(id);
    }
  }

  // --- Thumbnails (mirrors the Find label-list treatment) ------------------

  hasThumbnailUrl(id: number): boolean {
    const url = this.thumbnailUrl(id);
    if (url && this.thumbnailFailedUrls.has(url)) return false;
    const media = this.metadataCache.get(id);
    return (
      !!media &&
      (media.media_type === 'image' ||
        media.media_type === 'video' ||
        media.media_type === 'document' ||
        media.media_type === 'audio')
    );
  }

  thumbnailUrl(id: number): string {
    return this.activeContext.mediaUrl(`/api/medias/${id}/thumbnail`);
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
