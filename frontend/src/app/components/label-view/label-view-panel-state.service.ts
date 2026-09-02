import { Injectable, Signal, computed, inject, signal } from '@angular/core';
import {
  SettingsStateService,
  type PerMediaTypePref,
} from '../../services/settings-state.service';
import { iconSizeToGoalWidth } from '../../utils/grid-icon-size';

type FocusMode = 'click' | 'hover';

/** Accept only the two real focus modes; anything else falls back. */
function coerceFocusMode(raw: unknown): FocusMode | undefined {
  return raw === 'click' || raw === 'hover' ? raw : undefined;
}

/** Accept only a finite number of pixels; anything else falls back to "unset". */
function coercePx(raw: unknown): number | undefined {
  return typeof raw === 'number' && Number.isFinite(raw) ? raw : undefined;
}

/**
 * Per-media-type panel display preferences for `vt-label-view`.
 *
 * Owns the five `{media_type: value}` settings dicts (left grid icon size,
 * left/right focus mode, left/right saved panel widths) keyed on the active
 * media type. The component still owns the layout math and the CSS-var writes;
 * this service only owns the lookup and the persistence round-trip.
 *
 * Each preference is a {@link PerMediaTypePref} built by
 * `SettingsStateService.perMediaType`, so the values are `computed`s over the
 * settings signal rather than fields mirrored out of it. That matters for two
 * reasons beyond brevity:
 *
 * - The public getters below are read from `label-view.component.html` through
 *   the component's pass-through getters. They used to return **plain fields**
 *   written from an HTTP continuation, with no signal anywhere in the chain —
 *   not the getter-over-signal shape `docs/FRONTEND.md` section 5 sanctions.
 *   They rendered correctly only because a co-located `effect()` in
 *   `LabelViewComponent` happened to dirty the same view. Now the dependency
 *   is real and the binding repaints on its own.
 * - There is no longer a hydration step (`loadFromSettings`) or a shadow dict
 *   that can outlive the value it mirrored.
 */
@Injectable()
export class LabelViewPanelStateService {
  private settingsState = inject(SettingsStateService);

  private readonly _currentMediaType = signal('');
  /** The active media type as a signal, for `computed`s built on top of it. */
  readonly mediaType: Signal<string> = this._currentMediaType.asReadonly();

  private readonly gridIconSizeLeft = this.settingsState.perMediaType<string>(
    'grid_icon_size_left',
    this.mediaType,
    { fallback: 'M' },
  );
  private readonly focusLeft = this.settingsState.perMediaType<FocusMode>(
    'focus_mode_left',
    this.mediaType,
    { fallback: 'click', coerce: coerceFocusMode },
  );
  private readonly focusRight = this.settingsState.perMediaType<FocusMode>(
    'focus_mode_right',
    this.mediaType,
    { fallback: 'click', coerce: coerceFocusMode },
  );
  private readonly panelPx: Record<'left' | 'right', PerMediaTypePref<number | null>> = {
    left: this.settingsState.perMediaType<number | null>('panel_pct_left', this.mediaType, {
      fallback: null,
      coerce: coercePx,
    }),
    right: this.settingsState.perMediaType<number | null>('panel_pct_right', this.mediaType, {
      fallback: null,
      coerce: coercePx,
    }),
  };

  private readonly _gridGoalWidthLeft = computed(() =>
    iconSizeToGoalWidth(this.gridIconSizeLeft.value()),
  );

  setMediaType(mediaType: string): void {
    this._currentMediaType.set(mediaType);
  }

  get currentMediaType(): string {
    return this._currentMediaType();
  }

  get gridGoalWidthLeft(): number {
    return this._gridGoalWidthLeft();
  }

  get focusModeLeft(): FocusMode {
    return this.focusLeft.value();
  }

  get focusModeRight(): FocusMode {
    return this.focusRight.value();
  }

  /** Saved width for `side` at the current media type, or null if none. */
  getPanelPx(side: 'left' | 'right'): number | null {
    return this.panelPx[side].value();
  }

  /** Persist `px` for the active media type. No-op if media type is empty. */
  savePanelPx(side: 'left' | 'right', px: number): void {
    this.panelPx[side].set(px)?.subscribe();
  }
}
