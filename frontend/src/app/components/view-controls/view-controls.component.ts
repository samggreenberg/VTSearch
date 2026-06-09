import { Component, Input, OnChanges, OnDestroy, OnInit, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { SettingsStateService } from '../../services/settings-state.service';
import type { AppSettings } from '../../generated/api-client/models/app-settings';
import type { SettingsUpdate } from '../../generated/api-client/models/settings-update';
import { IconComponent } from '../icon/icon.component';

const ICON_SIZES = ['XS', 'S', 'M', 'L', 'XL'] as const;
type IconSize = (typeof ICON_SIZES)[number];

@Component({
  selector: 'vt-view-controls',
  standalone: true,
  imports: [CommonModule, IconComponent],
  templateUrl: './view-controls.component.html',
  styleUrl: './view-controls.component.scss',
})
export class ViewControlsComponent implements OnInit, OnChanges, OnDestroy {
  @Input() side: 'left' | 'right' | 'popup' = 'left';
  @Input() currentMediaType = '';
  /**
   * Whether to show the click/hover focus toggle. The VTSBrowse selection
   * panel reuses this control for its Grid/List + size buttons but has no
   * good/bad voting, so it hides the focus group with ``[showFocus]="false"``.
   */
  @Input() showFocus = true;

  viewMode: 'grid' | 'list' = 'list';
  focusMode: 'click' | 'hover' = 'click';
  gridIconSize: IconSize = 'M';

  private settings: AppSettings = { volume: 50 };
  private destroy$ = new Subject<void>();

  constructor(private settingsState: SettingsStateService) {}

  ngOnInit(): void {
    this.settingsState.load();
    this.settingsState.settings$
      .pipe(takeUntil(this.destroy$))
      .subscribe(settings => {
        if (!settings) return;
        this.settings = settings;
        this.refresh();
      });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['currentMediaType'] || changes['side']) {
      this.refresh();
    }
  }

  private refresh(): void {
    const type = this.currentMediaType;
    this.viewMode = (this.getDict('view_mode')[type] as 'grid' | 'list') ?? this.defaultViewMode();
    this.focusMode = (this.getDict('focus_mode')[type] as 'click' | 'hover') ?? 'click';
    const size = this.getDict('grid_icon_size')[type] as IconSize | undefined;
    this.gridIconSize = (size && ICON_SIZES.includes(size)) ? size : 'M';
  }

  private defaultViewMode(): 'grid' | 'list' {
    // Left panel reads as a list by default; the right panel and the
    // VTSBrowse bin popup default to a thumbnail grid.
    return this.side === 'left' ? 'list' : 'grid';
  }

  private key(prefix: 'view_mode' | 'focus_mode' | 'grid_icon_size'): keyof AppSettings {
    return `${prefix}_${this.side}` as keyof AppSettings;
  }

  private getDict(prefix: 'view_mode' | 'focus_mode' | 'grid_icon_size'): Record<string, string> {
    const value = this.settings[this.key(prefix)];
    return (value && typeof value === 'object') ? (value as Record<string, string>) : {};
  }

  setViewMode(mode: 'grid' | 'list'): void {
    if (!this.currentMediaType || this.viewMode === mode) return;
    this.save('view_mode', mode);
  }

  setFocusMode(mode: 'click' | 'hover'): void {
    if (!this.currentMediaType || this.focusMode === mode) return;
    this.save('focus_mode', mode);
  }

  bumpSize(delta: 1 | -1): void {
    if (!this.currentMediaType || this.viewMode === 'list') return;
    const idx = ICON_SIZES.indexOf(this.gridIconSize);
    const next = ICON_SIZES[Math.max(0, Math.min(ICON_SIZES.length - 1, idx + delta))];
    if (next === this.gridIconSize) return;
    this.save('grid_icon_size', next);
  }

  get atMaxSize(): boolean {
    return this.gridIconSize === ICON_SIZES[ICON_SIZES.length - 1];
  }

  get atMinSize(): boolean {
    return this.gridIconSize === ICON_SIZES[0];
  }

  private save(prefix: 'view_mode' | 'focus_mode' | 'grid_icon_size', value: string): void {
    const key = this.key(prefix);
    const updated = { ...this.getDict(prefix), [this.currentMediaType]: value };
    this.settingsState.update({ [key]: updated } as SettingsUpdate).subscribe();
  }
}
