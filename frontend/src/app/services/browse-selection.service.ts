import { Injectable, signal } from '@angular/core';

/**
 * Tracks which media items are selected in the VTSBrowse canvas.
 *
 * Selection is held at the *item* (media id) granularity, not the bin: bins
 * split and merge as you zoom, so a bin's display state — none, partial, or
 * fully selected — is *derived* from how many of its members are in this set,
 * and a selection made at one zoom level stays coherent at every other. The
 * set is in-memory and session-scoped: the service is provided per
 * {@link BrowseViewComponent}, so navigating away drops it, and the canvas
 * clears it when the underlying projection changes (media-type switch /
 * rebuild).
 *
 * {@link version} bumps on every mutation so the canvas can memoize each cell's
 * derived state and only recompute it when the selection actually changed,
 * keeping the per-frame redraw O(visible cells) rather than O(members) during a
 * pan.
 */
@Injectable()
export class BrowseSelectionService {
  private selected = new Set<number>();
  private readonly _version = signal(0);
  private readonly _allSelected = signal(false);

  /** How many items are currently selected. */
  get size(): number {
    return this.selected.size;
  }

  /** A snapshot array of the currently-selected media ids (insertion order). */
  ids(): number[] {
    return Array.from(this.selected);
  }

  /** Remove a single id from the selection. */
  remove(id: number): void {
    if (this.selected.delete(id)) this.bump();
  }

  /**
   * A monotonically increasing token that changes on every mutation. Read it
   * in a template or an `effect` (e.g. the canvas redraw, the selection panel
   * refresh, the bin popup re-highlight) to react to selection changes; it
   * bumps on every add/remove/clear. Because it is a signal, a write from any
   * context — including a raw canvas event handler — schedules change detection
   * under zoneless without an `NgZone.run` re-entry.
   */
  readonly version = this._version.asReadonly();

  /** Whether a specific media id is selected. */
  has(id: number): boolean {
    return this.selected.has(id);
  }

  /** How many of *ids* are currently selected (the bin's selected-member count). */
  selectedCountIn(ids: number[]): number {
    let n = 0;
    for (const id of ids) if (this.selected.has(id)) n++;
    return n;
  }

  /**
   * Toggle a bin's contents per the browse rule: clicking a bin with *no*
   * members selected fully selects it; clicking a partially- or fully-selected
   * bin fully clears it.
   */
  toggleBin(ids: number[]): void {
    if (ids.length === 0) return;
    if (this.selectedCountIn(ids) > 0) {
      this.removeAll(ids);
    } else {
      this.addAll(ids);
    }
  }

  /**
   * Whether the current selection is a *select-all-in-view* that hasn't been
   * manually edited since — the latch behind the selection panel's tri-state
   * checkbox's "all" ([x]) state.
   *
   * It is a latch, not a live geometric fact: {@link selectAllInView} turns it
   * on, and the next selection mutation of any other kind ({@link add},
   * {@link remove}, a marquee, a bin toggle, {@link clear}) turns it off via
   * {@link bump}. Deliberately, panning and zooming — which never touch the
   * set — leave it on, so the checkbox reads "your select-all is intact" rather
   * than flickering to [-] the moment a new bin scrolls into view. That keeps
   * the control cheap (no per-pan viewport recompute) and its meaning stable.
   */
  readonly allSelected = this._allSelected.asReadonly();

  /**
   * Add *ids* (the bins fully in view, computed by the canvas) and latch
   * {@link allSelected} on, so the panel checkbox shows [x]. Mirrors ctrl-A;
   * an empty view is a no-op that leaves the current state untouched.
   */
  selectAllInView(ids: number[]): void {
    if (ids.length === 0) return;
    this.addAll(ids); // bump() drops the latch; we re-arm it right after
    this._allSelected.set(true);
  }

  /** Add every id in *ids* to the selection (marquee union). */
  addAll(ids: number[]): void {
    let changed = false;
    for (const id of ids) {
      if (!this.selected.has(id)) {
        this.selected.add(id);
        changed = true;
      }
    }
    if (changed) this.bump();
  }

  /** Remove every id in *ids* from the selection. */
  removeAll(ids: number[]): void {
    let changed = false;
    for (const id of ids) {
      if (this.selected.delete(id)) changed = true;
    }
    if (changed) this.bump();
  }

  /**
   * Arm a one-shot exemption from the canvas's clear-on-new-projection rule.
   *
   * Selection is id-based, so it survives any relayout of the *same* items —
   * a Re-project re-fits UMAP over what's already on screen and only moves
   * points around. The view marks the selection before kicking off such a
   * build; the canvas consumes the mark when the new projection arrives and
   * keeps the selection instead of dropping it. Genuine new projections
   * (media-type switch, rebuild over different items) never set the mark, so
   * they still clear.
   */
  markSurviveProjectionChange(): void {
    this.surviveProjectionChange = true;
  }

  /** Consume the one-shot survive mark, returning whether it was set. */
  consumeSurviveProjectionChange(): boolean {
    const survive = this.surviveProjectionChange;
    this.surviveProjectionChange = false;
    return survive;
  }

  private surviveProjectionChange = false;

  /** Drop the whole selection. */
  clear(): void {
    if (this.selected.size === 0) return;
    this.selected.clear();
    this.bump();
  }

  private bump(): void {
    // Any real mutation drops the select-all latch by default; the one caller
    // that means to keep it ({@link selectAllInView}) re-arms it afterwards.
    this._allSelected.set(false);
    this._version.update((v) => v + 1);
  }
}
