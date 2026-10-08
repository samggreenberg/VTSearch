import { Injectable, inject, signal } from '@angular/core';
import { SettingsStateService } from './settings-state.service';

/**
 * Every hint Toasty gives (#4680), by the id the `hidden_hints` setting stores.
 * A new hint adds its id here; the server keeps whatever ids it is sent.
 */
export const HINT_IDS = ['add-dataset', 'add-detector', 'train'] as const;
export type HintId = (typeof HINT_IDS)[number];

/**
 * Whether each of Toasty's hints may show, backed by two per-user settings:
 * `hide_all_hints` (a bubble's "Hide all hints" box, Settings' Hide All) and
 * `hidden_hints` (the ids hidden one by one with "Hide this hint"). Settings'
 * Show All clears both; that write goes through the Settings modal's own draft,
 * not through here.
 *
 * Whether a hint is *needed* is the caller's business (the Dashboard shows
 * "add a dataset" only while there are none); this service answers only
 * whether the user still wants to see it.
 */
@Injectable({ providedIn: 'root' })
export class HintsService {
  private readonly settingsState = inject(SettingsStateService);

  // Hides already asked for whose PUT has not come back. A box hides its
  // bubble on the click rather than a round-trip later, and a second hide
  // sent before the first lands still carries the first one's id. Each entry
  // is dropped once its write settles: on success the settings say the same
  // thing, and on failure the hint comes back.
  private readonly pendingHidden = signal<ReadonlySet<HintId>>(new Set());
  private readonly pendingHideAll = signal(false);

  /**
   * True when hint `id` may show. False until settings load, so a user who
   * hid the hints never sees one flash up first.
   */
  isShown(id: HintId): boolean {
    const settings = this.settingsState.settingsSignal();
    if (!settings || settings.hide_all_hints || this.pendingHideAll()) return false;
    return !this.pendingHidden().has(id) && !(settings.hidden_hints ?? []).includes(id);
  }

  /** "Hide this hint": stop showing hint `id` to this user. */
  hide(id: HintId): void {
    const stored = this.settingsState.settingsSignal()?.hidden_hints ?? [];
    const hidden = [...new Set([...stored, ...this.pendingHidden(), id])];
    this.pendingHidden.update((ids) => new Set(ids).add(id));
    const settle = () =>
      this.pendingHidden.update((ids) => {
        const next = new Set(ids);
        next.delete(id);
        return next;
      });
    this.settingsState.update({ hidden_hints: hidden }).subscribe({ next: settle, error: settle });
  }

  /** "Hide all hints": stop showing every hint to this user. */
  hideAll(): void {
    this.pendingHideAll.set(true);
    const settle = () => this.pendingHideAll.set(false);
    this.settingsState.update({ hide_all_hints: true }).subscribe({ next: settle, error: settle });
  }
}
