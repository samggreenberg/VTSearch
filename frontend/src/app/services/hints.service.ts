import { Injectable, inject, signal } from '@angular/core';
import { ConnectionStateService } from './connection-state.service';
import { SettingsStateService } from './settings-state.service';

/**
 * Every hint Toasty gives (#4680), by the id the `hidden_hints` setting stores.
 * One id is one bubble with one text: a hint that would say something else in
 * another state is a second hint, so hiding one never hides words the user has
 * not read. A new hint adds its id here and a row to `docs/hints.md` (a test
 * holds the two together); the server keeps whatever ids it is sent. At most
 * one hint is on screen at a time: see `ToastyHintComponent` (#4807).
 */
export const HINT_IDS = [
  // Anywhere, over everything else, while the server is unreachable (#5149).
  'offline',
  // Dashboard, in the order a new user meets them.
  'add-dataset',
  'select-dataset',
  'mixed-datasets',
  'add-detector',
  'empty-detector-tab',
  'select-detector',
  'mismatch',
  'wait-for-import',
  'train',
  'train-one-dataset',
  'untrained-detector',
  'trained-here',
  'test-or-find',
  'find',
  // Train view.
  'start-voting',
  'fix-votes',
  'revote',
  'find-goods',
  'find-bads',
  'find-more',
  'keep-voting',
  'resort-prompt',
  'autopilot-done',
  'all-labeled',
  // Tune view.
  'test-vote',
  'test-done',
  'review',
  // Browse view: a ladder, each rung up until the first use of the feature
  // it teaches (#5360).
  'browse-pan-zoom',
  'browse-thumbnail-size',
  'browse-tile-details',
  'browse-select',
  'browse-verify',
  // Dialogs on the way: what to do in each.
  'import-dataset',
  'new-detector',
  'use-example',
  'find-results',
  'export-results',
] as const;
export type HintId = (typeof HINT_IDS)[number];

/**
 * The features a hint teaches, by the id the `used_features` setting stores
 * (#5360). Such a hint stays up until the user first uses its feature, on any
 * visit, and then never shows again: the next one takes over. Each feature is
 * named after the hint that teaches it.
 */
export const FEATURE_IDS = [
  'browse-pan-zoom',
  'browse-thumbnail-size',
  'browse-tile-details',
  'browse-select',
  'browse-verify',
] as const;
export type FeatureId = (typeof FEATURE_IDS)[number];

/**
 * Whether each of Toasty's hints may show, backed by two per-user settings:
 * `hide_all_hints` (a bubble's "Hide all hints" box, Settings' Hide All) and
 * `hidden_hints` (the ids hidden one by one with "Hide this hint"). A third,
 * `used_features`, records the features the user has tried, which retire the
 * hints that teach them (#5360). Settings' Show All clears all three; that
 * write goes through the Settings modal's own draft, not through here.
 *
 * Whether a hint is *needed* is the caller's business (the Dashboard shows
 * "add a dataset" only while there are none); this service answers whether
 * the user still wants to see it, and whether `offline` outranks it: while the
 * server is unreachable that is the only hint, since no step another one
 * points at can work (#5149).
 */
@Injectable({ providedIn: 'root' })
export class HintsService {
  private readonly settingsState = inject(SettingsStateService);
  private readonly connection = inject(ConnectionStateService);

  // Hides already asked for whose PUT has not come back. A box hides its
  // bubble on the click rather than a round-trip later, and a second hide
  // sent before the first lands still carries the first one's id. Each entry
  // is dropped once its write settles: on success the settings say the same
  // thing, and on failure the hint comes back.
  private readonly pendingHidden = signal<ReadonlySet<HintId>>(new Set());
  private readonly pendingHideAll = signal(false);
  // Likewise for first uses: the hint retires on the gesture, and a second
  // use before the first write lands sends nothing.
  private readonly pendingUsed = signal<ReadonlySet<FeatureId>>(new Set());

  /**
   * True when hint `id` may show. False until settings load, so a user who
   * hid the hints never sees one flash up first.
   */
  isShown(id: HintId): boolean {
    const settings = this.settingsState.settingsSignal();
    if (!settings || settings.hide_all_hints || this.pendingHideAll()) return false;
    if (id !== 'offline' && this.connection.status() === 'offline') return false;
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

  /** True once this user has used feature `id`, on this visit or an earlier one. */
  hasUsed(id: FeatureId): boolean {
    return this.pendingUsed().has(id) || (this.settingsState.settingsSignal()?.used_features ?? []).includes(id);
  }

  /**
   * The user just used feature `id`: record it, the first time only. Nothing
   * is sent before settings load, since the write replaces the stored list.
   */
  markUsed(id: FeatureId): void {
    const settings = this.settingsState.settingsSignal();
    if (!settings || this.hasUsed(id)) return;
    const used = [...new Set([...(settings.used_features ?? []), ...this.pendingUsed(), id])];
    this.pendingUsed.update((ids) => new Set(ids).add(id));
    const settle = () =>
      this.pendingUsed.update((ids) => {
        const next = new Set(ids);
        next.delete(id);
        return next;
      });
    this.settingsState.update({ used_features: used }).subscribe({ next: settle, error: settle });
  }
}
