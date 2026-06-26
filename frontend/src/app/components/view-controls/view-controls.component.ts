import { ChangeDetectionStrategy, Component, computed, effect, inject, input, OnChanges, OnInit, signal, SimpleChanges } from '@angular/core';

import { SettingsStateService } from '../../services/settings-state.service';
import type { AppSettings } from '../../generated/api-client/models/app-settings';
import type { SettingsUpdate } from '../../generated/api-client/models/settings-update';
import { IconComponent } from '../icon/icon.component';

const ICON_SIZES = ['XS', 'S', 'M', 'L', 'XL'] as const;
type IconSize = (typeof ICON_SIZES)[number];

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-view-controls',
  standalone: true,
  imports: [IconComponent],
  templateUrl: './view-controls.component.html',
  styleUrl: './view-controls.component.scss',
})
export class ViewControlsComponent implements OnInit, OnChanges {
  private settingsState = inject(SettingsStateService);

  readonly side = input<'left' | 'right' | 'popup'>('left');
  readonly currentMediaType = input('');
  /**
   * Whether to show the click/hover focus toggle. The VTSBrowse selection
   * panel reuses this control for its Grid/List + size buttons but has no
   * good/bad voting, so it hides the focus group with ``[showFocus]="false"``.
   */
  readonly showFocus = input(true);
  /**
   * Whether the thumbnail-size buttons stay active in list mode. The
   * left/right Find panels keep the historic behavior (size applies to grid
   * only, so the buttons disable in list mode), but the VTSBrowse bin popup
   * scales its list-mode thumbnails too, so it opts in with
   * ``[allowSizeInList]="true"``.
   */
  readonly allowSizeInList = input(false);
  /**
   * Whether to show the Grid/List toggle. The VTSBrowse bin popup is grid-only
   * (it pairs the grid with a large hover preview), so it hides the toggle with
   * ``[showViewMode]="false"`` while keeping the thumbnail-size buttons.
   */
  readonly showViewMode = input(true);

  // These are signals (not plain fields) so the OnPush template re-renders when
  // the constructor effect's `refresh()` updates them after a settings change.
  // With plain fields the controls would not repaint until the next unrelated
  // change-detection pass, which is what made the Grid/List toggle take two
  // clicks and left the thumbnail-size buttons looking stuck-disabled.
  readonly viewMode = signal<'grid' | 'list'>('list');
  readonly focusMode = signal<'click' | 'hover'>('click');
  readonly gridIconSize = signal<IconSize>('M');

  private settings: AppSettings = { volume: 50 };

  constructor() {
    effect(() => {
      const settings = this.settingsState.settingsSignal();
      if (!settings) return;
      this.settings = settings;
      this.refresh();
    });
  }

  ngOnInit(): void {
    this.settingsState.load();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['currentMediaType'] || changes['side']) {
      this.refresh();
    }
  }

  private refresh(): void {
    const type = this.currentMediaType();
    this.viewMode.set((this.getDict('view_mode')[type] as 'grid' | 'list') ?? this.defaultViewMode());
    this.focusMode.set((this.getDict('focus_mode')[type] as 'click' | 'hover') ?? 'click');
    const size = this.getDict('grid_icon_size')[type] as IconSize | undefined;
    this.gridIconSize.set((size && ICON_SIZES.includes(size)) ? size : 'M');
  }

  private defaultViewMode(): 'grid' | 'list' {
    // Left panel reads as a list by default; the right panel and the
    // VTSBrowse bin popup default to a thumbnail grid.
    return this.side() === 'left' ? 'list' : 'grid';
  }

  private key(prefix: 'view_mode' | 'focus_mode' | 'grid_icon_size'): keyof AppSettings {
    return `${prefix}_${this.side()}` as keyof AppSettings;
  }

  private getDict(prefix: 'view_mode' | 'focus_mode' | 'grid_icon_size'): Record<string, string> {
    const value = this.settings[this.key(prefix)];
    return (value && typeof value === 'object') ? (value as Record<string, string>) : {};
  }

  setViewMode(mode: 'grid' | 'list'): void {
    if (!this.currentMediaType() || this.viewMode() === mode) return;
    // Reflect the change immediately so the toggle (and the thumbnail-size
    // buttons it gates) update on the first click; `refresh()` reconciles with
    // the persisted settings once the round-trip lands.
    this.viewMode.set(mode);
    this.save('view_mode', mode);
  }

  setFocusMode(mode: 'click' | 'hover'): void {
    if (!this.currentMediaType() || this.focusMode() === mode) return;
    this.focusMode.set(mode);
    this.save('focus_mode', mode);
  }

  bumpSize(delta: 1 | -1): void {
    if (!this.currentMediaType() || (this.viewMode() === 'list' && !this.allowSizeInList())) return;
    const idx = ICON_SIZES.indexOf(this.gridIconSize());
    const next = ICON_SIZES[Math.max(0, Math.min(ICON_SIZES.length - 1, idx + delta))];
    if (next === this.gridIconSize()) return;
    this.gridIconSize.set(next);
    this.save('grid_icon_size', next);
  }

  readonly atMaxSize = computed(() => this.gridIconSize() === ICON_SIZES[ICON_SIZES.length - 1]);

  readonly atMinSize = computed(() => this.gridIconSize() === ICON_SIZES[0]);

  private save(prefix: 'view_mode' | 'focus_mode' | 'grid_icon_size', value: string): void {
    const key = this.key(prefix);
    const updated = { ...this.getDict(prefix), [this.currentMediaType()]: value };
    this.settingsState.update({ [key]: updated } as SettingsUpdate).subscribe();
  }
}
