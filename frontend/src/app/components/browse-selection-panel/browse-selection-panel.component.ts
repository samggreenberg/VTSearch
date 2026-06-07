import { Component, Input, OnDestroy, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subscription } from 'rxjs';
import { BrowseSelectionService } from '../../services/browse-selection.service';
import { MediaMetadataCacheService } from '../../services/media-metadata-cache.service';
import { ActiveContextService } from '../../services/active-context.service';
import { SettingsStateService } from '../../services/settings-state.service';
import { ViewControlsComponent } from '../view-controls/view-controls.component';
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
 * item currently selected on the canvas, with the same Grid/List + size
 * controls as the Find view's labeled-item lists. Modeled on the Find "Good"
 * panel — but where that panel lists votes, this lists the live canvas
 * selection, and clicking an item removes it from the selection.
 *
 * Names and thumbnails are resolved lazily through
 * {@link MediaMetadataCacheService} (the browse view never loads the full media
 * list), so the panel renders ids immediately and fills in detail as the cache
 * hydrates. The Grid/List + thumbnail-size choice reuses the shared
 * ``view_mode_right`` / ``grid_icon_size_right`` per-media-type settings, so it
 * stays in step with the Find right panel.
 */
@Component({
  selector: 'vt-browse-selection-panel',
  standalone: true,
  imports: [CommonModule, FormsModule, ViewControlsComponent],
  templateUrl: './browse-selection-panel.component.html',
  styleUrl: './browse-selection-panel.component.scss',
})
export class BrowseSelectionPanelComponent implements OnInit, OnDestroy {
  /** Active media type, used to resolve the per-type view-mode + size prefs. */
  @Input() mediaType = '';

  count = 0;
  sortMode: SelectionSortMode = 'time-desc';
  viewMode: 'grid' | 'list' = 'grid';
  gridGoalWidth = 80;
  sortedEntries: SelectionEntry[] = [];

  private ids: number[] = [];
  private viewModeRightDict: Record<string, 'grid' | 'list'> = {};
  private gridIconSizeRightDict: Record<string, string> = {};
  private readonly thumbnailFailedUrls = new Set<string>();
  private readonly subs: Subscription[] = [];

  constructor(
    private selection: BrowseSelectionService,
    private metadataCache: MediaMetadataCacheService,
    private activeContext: ActiveContextService,
    private settingsState: SettingsStateService,
  ) {}

  ngOnInit(): void {
    this.refreshSelection();
    this.subs.push(
      this.selection.changed$.subscribe(() => this.refreshSelection()),
      this.metadataCache.version$.subscribe(() => {
        this.sortedEntries = this.buildSortedEntries();
      }),
      this.settingsState.settings$.subscribe((settings) => {
        if (!settings) return;
        const viewDict = settings.view_mode_right;
        if (viewDict && typeof viewDict === 'object') {
          this.viewModeRightDict = viewDict as Record<string, 'grid' | 'list'>;
        }
        const sizeDict = settings.grid_icon_size_right;
        if (sizeDict && typeof sizeDict === 'object') {
          this.gridIconSizeRightDict = sizeDict as Record<string, string>;
        }
        this.applyViewPrefs();
      }),
    );
    this.settingsState.load();
  }

  ngOnDestroy(): void {
    for (const sub of this.subs) sub.unsubscribe();
  }

  private refreshSelection(): void {
    this.ids = this.selection.ids();
    this.count = this.ids.length;
    this.metadataCache.ensureLoaded(this.ids);
    this.sortedEntries = this.buildSortedEntries();
  }

  private applyViewPrefs(): void {
    if (!this.mediaType) return;
    this.viewMode = this.viewModeRightDict[this.mediaType] ?? 'grid';
    this.gridGoalWidth = iconSizeToGoalWidth(this.gridIconSizeRightDict[this.mediaType] ?? 'M');
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
    this.sortedEntries = this.buildSortedEntries();
  }

  get isGrid(): boolean {
    return this.viewMode === 'grid';
  }

  clear(): void {
    this.selection.clear();
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
    return this.activeContext.mediaUrl(`/api/medias/${id}/image`);
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
