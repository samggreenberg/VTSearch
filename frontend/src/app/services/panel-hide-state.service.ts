import { Injectable, Signal, computed, inject, signal } from '@angular/core';
import { SettingsStateService } from './settings-state.service';

export type PanelSide = 'left' | 'right';

const SETTING_KEY = {
  left: 'hide_left_panel',
  right: 'hide_right_panel',
} as const;

/**
 * Whether Train and Test fold each side panel to its strip (#4673): the
 * `hide_left_panel` / `hide_right_panel` settings, one per side, shared by
 * both views.
 *
 * `SettingsStateService.update` writes the store only once the PUT returns, so
 * a strip bound straight to the setting would sit unchanged for the length of
 * the round trip. A click is held here until the server echoes it, the way
 * `vt-view-controls` keeps a local optimistic signal over its preference; the
 * sequence number keeps an older write's echo from clearing a newer click.
 *
 * Before settings land a side reads hidden, the default, so a fresh install
 * never flashes its panels open on the way in.
 */
@Injectable({ providedIn: 'root' })
export class PanelHideStateService {
  private readonly settingsState = inject(SettingsStateService);

  private readonly pending = {
    left: signal<boolean | null>(null),
    right: signal<boolean | null>(null),
  };
  private readonly seq = { left: 0, right: 0 };

  readonly left: Signal<boolean> = computed(
    () => this.pending.left() ?? this.settingsState.settingsSignal()?.hide_left_panel ?? true,
  );
  readonly right: Signal<boolean> = computed(
    () => this.pending.right() ?? this.settingsState.settingsSignal()?.hide_right_panel ?? true,
  );

  hidden(side: PanelSide): Signal<boolean> {
    return side === 'left' ? this.left : this.right;
  }

  /** Fold or open `side` now, and remember it. */
  set(side: PanelSide, hidden: boolean): void {
    const n = ++this.seq[side];
    this.pending[side].set(hidden);
    const settle = () => {
      if (this.seq[side] === n) this.pending[side].set(null);
    };
    this.settingsState.update({ [SETTING_KEY[side]]: hidden }).subscribe({ next: settle, error: settle });
  }

  toggle(side: PanelSide): void {
    this.set(side, !this.hidden(side)());
  }
}
