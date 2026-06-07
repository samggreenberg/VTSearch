import { Injectable } from '@angular/core';
import { Observable, Subject } from 'rxjs';

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
  private _version = 0;
  private readonly changedSubject = new Subject<void>();

  /** Emits after every change to the selection set. */
  readonly changed$: Observable<void> = this.changedSubject.asObservable();

  /** How many items are currently selected. */
  get size(): number {
    return this.selected.size;
  }

  /** A monotonically increasing token that changes on every mutation. */
  get version(): number {
    return this._version;
  }

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

  /** Drop the whole selection. */
  clear(): void {
    if (this.selected.size === 0) return;
    this.selected.clear();
    this.bump();
  }

  private bump(): void {
    this._version++;
    this.changedSubject.next();
  }
}
