import { computed, effect, inject, Injectable, Signal, signal, WritableSignal } from '@angular/core';
import { ActiveContextService } from './active-context.service';
import { DatasetStateService } from './dataset-state.service';

export type SelectionKind = 'dataset' | 'detector';

/** Which detector-grid tab is showing: Drafts (editable, `!autofind`) or
 *  AutoRun (frozen, `autofind` — auto-run against each dataset on import).
 *  A detector lives in exactly one tab; the ⋯ menu moves it between them. */
export type DetectorTab = 'drafts' | 'autorun';

/** Header master-checkbox tri-state for one grid. */
export type SelectionState = 'none' | 'some' | 'all';

/**
 * Owns the Dashboard's table row-selection, so the top-bar
 * `vt-context-pulldown`s can show what the user has highlighted rather than
 * merely what is loaded into the backend.
 *
 * Two selection concepts coexist:
 *  - **active/loaded context** (`ActiveContextService`) — the pair the
 *    backend has loaded, which the pulldown shows on the label/find/browse
 *    views (there are no tables there to select from).
 *  - **table selection** (this service) — the highlighted rows that drive
 *    Train / Find / Combine / Delete.
 *
 * While the Dashboard is on screen the pulldown reads the *table selection*
 * from here; once the user navigates away the Dashboard is destroyed,
 * `dashboardVisible` flips to false, and the pulldown falls back to the
 * active/loaded context. That view-awareness rides on the Dashboard's
 * component lifecycle rather than the router, so it needs no URL parsing.
 *
 * The selection itself is *not* tied to that lifecycle: it lives here, in a
 * root singleton, so a round trip to the label view and back leaves the same
 * rows highlighted (and the same names in the top bar) rather than resetting
 * to whatever the registry auto-selects. `detectorTab` lives here for the
 * same reason — "selection is per-tab" has to survive the round trip too,
 * or a returning user gets a hidden selection feeding the section actions.
 *
 * Everything a template or pulldown reads is a signal, so there is no mirror
 * to push and no push to forget: `datasetIds` / `detectorIds` are `computed`
 * over the selection and the registry, and the one write-out that isn't a
 * read (mirroring an unambiguous single pick into the active-context intent)
 * is a single `effect` rather than a call at every mutation site.
 */
@Injectable({ providedIn: 'root' })
export class DashboardSelectionService {
  private datasetState = inject(DatasetStateService);
  private activeContext = inject(ActiveContextService);

  private readonly visible = signal(false);
  /** True while the Dashboard component is mounted. Set from its
   *  `ngOnInit` / `ngOnDestroy`. */
  readonly dashboardVisible: Signal<boolean> = this.visible.asReadonly();

  /** The selected ids per grid. Held as immutable `Set`s behind signals:
   *  every mutation publishes a fresh Set, so a template read (or a
   *  `computed`) is notified — a mutated-in-place Set is not. */
  private readonly sets: Record<SelectionKind, WritableSignal<ReadonlySet<string>>> = {
    dataset: signal<ReadonlySet<string>>(new Set<string>()),
    detector: signal<ReadonlySet<string>>(new Set<string>()),
  };

  private readonly tab = signal<DetectorTab>('drafts');
  readonly detectorTab: Signal<DetectorTab> = this.tab.asReadonly();

  /** Selected ids that still exist in the registry, in registry order.
   *  Filtered so a row deleted out from under the selection can't inflate
   *  the pulldown's "Multiple" count before the prune lands. */
  readonly datasetIds = computed(() =>
    this.datasetState.datasets.filter((d) => this.sets.dataset().has(d.id)).map((d) => d.id),
  );
  readonly detectorIds = computed(() =>
    this.datasetState.detectors.filter((d) => this.sets.detector().has(d.id)).map((d) => d.id),
  );

  constructor() {
    // While the Dashboard is on screen the pulldowns read the selection
    // above; the moment it unmounts they fall back to the active-context
    // *intent*. Without also mirroring a lone pick into that intent, a
    // freshly imported/created (implicitly selected) item is forgotten by
    // the pulldowns as soon as you leave the Dashboard by any route that
    // doesn't load a context — the picker snaps back to "Select a …".
    // Mirror only an unambiguous single selection; a 0- or multi-selection
    // leaves intent untouched so we never blank out the intent of an
    // already-loaded pair. Intent-only (never `setActive`), so the HTTP
    // interceptor keeps tagging the still-loaded pair, per the H25
    // intent/active split.
    effect(() => {
      const datasetIds = this.datasetIds();
      const detectorIds = this.detectorIds();
      if (!this.dashboardVisible()) return;
      const soleDataset = datasetIds.length === 1 ? datasetIds[0] : this.activeContext.intentDatasetId;
      const soleDetector =
        detectorIds.length === 1 ? detectorIds[0] : this.activeContext.intentModelId;
      this.activeContext.setIntent(soleDataset, soleDetector);
    });
  }

  setDashboardVisible(visible: boolean): void {
    this.visible.set(visible);
  }

  /** Switch detector-grid tabs. Selection is per-tab: a hidden selection
   *  would silently feed the section actions and the Train/Find buttons, so
   *  it's cleared on every switch. */
  setDetectorTab(tab: DetectorTab): void {
    if (this.tab() === tab) return;
    this.tab.set(tab);
    this.clear('detector');
  }

  // --- Selection reads ---

  /** The raw selected-id set for one grid, unfiltered by the registry. */
  ids(kind: SelectionKind): ReadonlySet<string> {
    return this.sets[kind]();
  }

  has(kind: SelectionKind, id: string): boolean {
    return this.sets[kind]().has(id);
  }

  count(kind: SelectionKind): number {
    return this.sets[kind]().size;
  }

  /** Tri-state for the grid's header master-checkbox. `total` is the number
   *  of rows the grid currently *shows* (the detector grid's visible tab, not
   *  the whole registry). */
  selectionState(kind: SelectionKind, total: number): SelectionState {
    const selected = this.count(kind);
    if (selected === 0) return 'none';
    return selected >= total ? 'all' : 'some';
  }

  // --- Selection writes ---

  /**
   * The row-click ladder, shared by both grids and by the top-bar pulldown.
   * `additive` (Ctrl/Cmd, or a checkbox click) toggles a single id in/out of
   * a multi-selection; otherwise it's a plain single-select that toggles off
   * when it's already the sole pick.
   */
  toggle(kind: SelectionKind, id: string, additive: boolean): void {
    const current = this.sets[kind]();
    if (additive) {
      const next = new Set(current);
      if (!next.delete(id)) next.add(id);
      this.write(kind, next);
    } else if (current.has(id) && current.size === 1) {
      this.write(kind, new Set());
    } else {
      this.write(kind, new Set([id]));
    }
  }

  /** Select-all ladder for the header master-checkbox: clear when everything
   *  visible is already picked, otherwise pick exactly the visible rows. */
  toggleAll(kind: SelectionKind, visibleIds: string[]): void {
    if (this.selectionState(kind, visibleIds.length) === 'all') this.write(kind, new Set());
    else this.write(kind, new Set(visibleIds));
  }

  /** Replace the selection outright (used by the registry auto-select). */
  selectOnly(kind: SelectionKind, ids: Iterable<string>): void {
    this.write(kind, new Set(ids));
  }

  clear(kind: SelectionKind): void {
    this.write(kind, new Set());
  }

  /** Drop one id (a row the user just deleted). */
  deselect(kind: SelectionKind, id: string): void {
    const current = this.sets[kind]();
    if (!current.has(id)) return;
    const next = new Set(current);
    next.delete(id);
    this.write(kind, next);
  }

  /** Prune the selection to ids that still exist in the registry. */
  retain(kind: SelectionKind, existing: ReadonlySet<string>): void {
    const current = this.sets[kind]();
    const next = new Set([...current].filter((id) => existing.has(id)));
    this.write(kind, next);
  }

  /** Publish a new set, skipping the write when nothing changed so an
   *  idempotent prune/reselect doesn't re-run every downstream `computed`. */
  private write(kind: SelectionKind, next: ReadonlySet<string>): void {
    const current = this.sets[kind]();
    if (current.size === next.size && [...next].every((id) => current.has(id))) return;
    this.sets[kind].set(next);
  }
}
