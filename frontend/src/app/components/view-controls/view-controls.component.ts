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
   * panel reuses this control for its size buttons but has no good/bad voting,
   * so it hides the focus group with ``[showFocus]="false"``.
   */
  readonly showFocus = input(true);

  // These are signals (not plain fields) so the OnPush template re-renders when
  // the constructor effect's `refresh()` updates them after a settings change.
  // With plain fields the controls would not repaint until the next unrelated
  // change-detection pass, which left the thumbnail-size buttons looking
  // stuck-disabled.
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
    this.focusMode.set((this.getDict('focus_mode')[type] as 'click' | 'hover') ?? 'click');
    const size = this.getDict('grid_icon_size')[type] as IconSize | undefined;
    this.gridIconSize.set((size && ICON_SIZES.includes(size)) ? size : 'M');
  }

  private key(prefix: 'focus_mode' | 'grid_icon_size'): keyof AppSettings {
    return `${prefix}_${this.side()}` as keyof AppSettings;
  }

  private getDict(prefix: 'focus_mode' | 'grid_icon_size'): Record<string, string> {
    const value = this.settings[this.key(prefix)];
    return (value && typeof value === 'object') ? (value as Record<string, string>) : {};
  }

  setFocusMode(mode: 'click' | 'hover'): void {
    if (!this.currentMediaType() || this.focusMode() === mode) return;
    this.focusMode.set(mode);
    this.save('focus_mode', mode);
  }

  bumpSize(delta: 1 | -1): void {
    if (!this.currentMediaType()) return;
    const idx = ICON_SIZES.indexOf(this.gridIconSize());
    const next = ICON_SIZES[Math.max(0, Math.min(ICON_SIZES.length - 1, idx + delta))];
    if (next === this.gridIconSize()) return;
    this.gridIconSize.set(next);
    this.save('grid_icon_size', next);
  }

  readonly atMaxSize = computed(() => this.gridIconSize() === ICON_SIZES[ICON_SIZES.length - 1]);

  readonly atMinSize = computed(() => this.gridIconSize() === ICON_SIZES[0]);

  private save(prefix: 'focus_mode' | 'grid_icon_size', value: string): void {
    const key = this.key(prefix);
    const updated = { ...this.getDict(prefix), [this.currentMediaType()]: value };
    this.settingsState.update({ [key]: updated } as SettingsUpdate).subscribe();
  }
}
