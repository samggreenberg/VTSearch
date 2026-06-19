import { Injectable, signal } from '@angular/core';

export type SortMode = 'text' | 'learned' | 'load';
export type SelectMode = 'top' | 'bottom' | 'hard' | 'new';

export interface SortedItem {
  id: number;
  score: number;
  /** Normalised [x0, y0, x1, y1] of the region the embedder/MLP matched best.
   *  Only set when the active dataset's embedder reports
   *  ``supports_patch_regions``. */
  bestRegion?: number[];
}

/**
 * Shared sort state. Backed by signals so a write from any context — an HTTP
 * subscribe, a poll timer, a learned-sort job continuation — notifies Angular's
 * scheduler and repaints the views that bind these getters, with no zone.js
 * involved (docs/plans/zoneless-migration.md, Phase 2.5 / Recipe B).
 *
 * Each value is exposed as a *value-returning getter* over a private signal, so
 * existing `sortState.sortBusy` template bindings stay byte-for-byte the same
 * yet become reactive under zoneless: a signal read through a getter during
 * template evaluation is tracked as a dependency of that view (proven by
 * `testing/getter-signal-zoneless.spec.ts`). Imperative callers read the same
 * getters; mutation goes through the `set*` methods.
 */
@Injectable({ providedIn: 'root' })
export class SortStateService {
  private readonly _sortMode = signal<SortMode>('text');
  private readonly _selectMode = signal<SelectMode>('top');
  private readonly _sortOrder = signal<SortedItem[] | null>(null);
  private readonly _threshold = signal<number | null>(null);
  private readonly _sortBusy = signal(false);
  private readonly _sortStatus = signal('');
  private readonly _sortProgress = signal(0);
  private readonly _sortProgressTotal = signal(0);
  private readonly _inclusion = signal(0);
  private readonly _loadSortLabel = signal('');
  private readonly _textQuery = signal('');

  get sortMode(): SortMode {
    return this._sortMode();
  }

  get selectMode(): SelectMode {
    return this._selectMode();
  }

  get sortOrder(): SortedItem[] | null {
    return this._sortOrder();
  }

  get threshold(): number | null {
    return this._threshold();
  }

  get sortBusy(): boolean {
    return this._sortBusy();
  }

  get sortStatus(): string {
    return this._sortStatus();
  }

  get sortProgress(): number {
    return this._sortProgress();
  }

  get sortProgressTotal(): number {
    return this._sortProgressTotal();
  }

  get inclusion(): number {
    return this._inclusion();
  }

  get loadSortLabel(): string {
    return this._loadSortLabel();
  }

  get textQuery(): string {
    return this._textQuery();
  }

  setSortMode(mode: SortMode): void {
    this._sortMode.set(mode);
  }

  setSelectMode(mode: SelectMode): void {
    this._selectMode.set(mode);
  }

  setSortResults(order: SortedItem[], threshold: number): void {
    this._sortOrder.set(order);
    this._threshold.set(threshold);
  }

  setSortBusy(busy: boolean): void {
    this._sortBusy.set(busy);
  }

  setSortStatus(status: string): void {
    this._sortStatus.set(status);
  }

  setSortProgress(current: number, total: number): void {
    this._sortProgress.set(current);
    this._sortProgressTotal.set(total);
  }

  setInclusion(value: number): void {
    this._inclusion.set(value);
  }

  setLoadSortLabel(label: string): void {
    this._loadSortLabel.set(label);
  }

  setTextQuery(query: string): void {
    this._textQuery.set(query);
  }

  clear(): void {
    this._sortMode.set('text');
    this._selectMode.set('top');
    this._sortOrder.set(null);
    this._threshold.set(null);
    this._sortBusy.set(false);
    this._sortStatus.set('');
    this._sortProgress.set(0);
    this._sortProgressTotal.set(0);
    this._inclusion.set(0);
    this._loadSortLabel.set('');
    this._textQuery.set('');
  }
}
