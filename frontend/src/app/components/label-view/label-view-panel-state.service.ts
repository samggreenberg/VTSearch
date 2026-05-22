import { Injectable } from '@angular/core';
import type { AppSettings } from '../../generated/api-client/models/app-settings';
import { SettingsStateService } from '../../services/settings-state.service';
import { iconSizeToGoalWidth } from '../../utils/grid-icon-size';

type ViewMode = 'grid' | 'list';
type FocusMode = 'click' | 'hover';

/**
 * Per-media-type panel display preferences for `vt-label-view`.
 *
 * Tracks the six dicts (left view mode, left grid icon size, left/right focus
 * mode, left/right saved panel widths) that used to live inline on
 * `LabelViewComponent` and exposes them as computed getters keyed on the
 * currently-active media type.  `loadFromSettings` hydrates the dicts from
 * the loaded `AppSettings` blob; `savePanelPx` persists a width change for
 * the active media type back through `SettingsStateService`.
 *
 * The component still owns the layout math and the CSS-var writes — this
 * service only owns the per-media-type lookup tables and the persistence
 * round-trip.
 */
@Injectable()
export class LabelViewPanelStateService {
  private viewModeLeftDict: Record<string, ViewMode> = {};
  private gridIconSizeLeftDict: Record<string, string> = {};
  private focusModeLeftDict: Record<string, FocusMode> = {};
  private focusModeRightDict: Record<string, FocusMode> = {};
  private panelPxLeftDict: Record<string, number> = {};
  private panelPxRightDict: Record<string, number> = {};

  private _currentMediaType = '';

  constructor(private settingsState: SettingsStateService) {}

  setMediaType(mediaType: string): void {
    this._currentMediaType = mediaType;
  }

  get currentMediaType(): string {
    return this._currentMediaType;
  }

  get viewModeLeft(): ViewMode {
    return this.viewModeLeftDict[this._currentMediaType] ?? 'list';
  }

  get gridGoalWidthLeft(): number {
    return iconSizeToGoalWidth(this.gridIconSizeLeftDict[this._currentMediaType] ?? 'M');
  }

  get focusModeLeft(): FocusMode {
    return this.focusModeLeftDict[this._currentMediaType] ?? 'click';
  }

  get focusModeRight(): FocusMode {
    return this.focusModeRightDict[this._currentMediaType] ?? 'click';
  }

  /** Saved width for `side` at the current media type, or null if none. */
  getPanelPx(side: 'left' | 'right'): number | null {
    const dict = side === 'left' ? this.panelPxLeftDict : this.panelPxRightDict;
    return dict[this._currentMediaType] ?? null;
  }

  /** Persist `px` for the active media type. No-op if media type is empty. */
  savePanelPx(side: 'left' | 'right', px: number): void {
    if (!this._currentMediaType) return;
    const dict = side === 'left' ? this.panelPxLeftDict : this.panelPxRightDict;
    dict[this._currentMediaType] = px;
    const key = side === 'left' ? 'panel_pct_left' : 'panel_pct_right';
    this.settingsState.update({ [key]: { ...dict } }).subscribe();
  }

  loadFromSettings(settings: AppSettings): void {
    const vm = settings.view_mode_left;
    if (vm && typeof vm === 'object') this.viewModeLeftDict = vm as Record<string, ViewMode>;
    const gs = settings.grid_icon_size_left;
    if (gs && typeof gs === 'object') this.gridIconSizeLeftDict = gs as Record<string, string>;
    const fl = settings.focus_mode_left;
    if (fl && typeof fl === 'object') this.focusModeLeftDict = fl as Record<string, FocusMode>;
    const fr = settings.focus_mode_right;
    if (fr && typeof fr === 'object') this.focusModeRightDict = fr as Record<string, FocusMode>;
    const pl = settings.panel_pct_left;
    if (pl && typeof pl === 'object') this.panelPxLeftDict = pl as Record<string, number>;
    const pr = settings.panel_pct_right;
    if (pr && typeof pr === 'object') this.panelPxRightDict = pr as Record<string, number>;
  }
}
