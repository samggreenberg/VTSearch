import { ChangeDetectionStrategy, ChangeDetectorRef, Component, effect, ElementRef, HostListener, inject, input, OnDestroy, OnInit, viewChild } from '@angular/core';
import { TitleCasePipe } from '@angular/common';
import { NavigationEnd, Router } from '@angular/router';

import { Observable, Subject } from 'rxjs';
import { filter, takeUntil } from 'rxjs/operators';
import { DatasetStateService } from '../../services/dataset-state.service';
import { ActiveContextService } from '../../services/active-context.service';
import { ContextSwitchService } from '../../services/context-switch.service';
import { NewThingFlowsService } from '../../services/new-thing-flows.service';
import { PulldownControlService } from '../../services/pulldown-control.service';
import { DashboardSortService } from '../../services/dashboard-sort.service';
import { DashboardSelectionService } from '../../services/dashboard-selection.service';
import { RunningJobsService, pairKey } from '../../services/running-jobs.service';
import { SortState, sortRowsByColumn } from '../../utils/sort-rows';
import { DatasetRegistryEntry } from '../../models/api.models';
import { IconComponent } from '../icon/icon.component';
import { DetectorRegistryEntry } from '../../generated/api-client/models/detector-registry-entry';
import { isPairCompatible } from '../../utils/context-compat';

type PulldownKind = 'dataset' | 'detector';

/** Wrap a possibly-empty id as a 0-or-1 element list, so intent-driven
 *  (single) and selection-driven (multi) sources share one code path. */
function idList(id: string): string[] {
  return id ? [id] : [];
}

/** The lone id in a list, or '' when there are zero or many — used where an
 *  unambiguous single partner is required (compatibility dimming). */
function singleId(ids: string[]): string {
  return ids.length === 1 ? ids[0] : '';
}

interface PulldownRow {
  id: string;
  name: string;
  mediaType: string;
  loaded: boolean;
  active: boolean;
  compatibleWithOther: boolean;
  incompatReason: string;
  /** True if this row's id paired with the other half's *active* id has a
   *  running or pending JobManager job. Drives the spinner glyph. */
  busy: boolean;
  /** Logical job-type names contributing to ``busy`` (``"learned-sort"``,
   *  ``"eval"``, …). Used for the spinner's tooltip. */
  busyJobTypes: string[];
}

/**
 * Top-bar pulldown that lets the user switch the active dataset or
 * detector without going back to the Dashboard. One instance per half
 * of the pair (parameterised by `kind`).
 *
 * Renders a closed-state button (glyph + label + caret) and, when
 * opened, a dropdown with one row per registry entry plus a
 * "+ Add New" footer. Click outside to close. Keyboard: ArrowUp /
 * ArrowDown to navigate, Enter to pick, Escape to close.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-context-pulldown',
  standalone: true,
  imports: [TitleCasePipe, IconComponent],
  templateUrl: './context-pulldown.component.html',
  styleUrl: './context-pulldown.component.scss',
})
export class ContextPulldownComponent implements OnInit, OnDestroy {
  private host = inject<ElementRef<HTMLElement>>(ElementRef);
  private router = inject(Router);
  private datasetState = inject(DatasetStateService);
  private activeContext = inject(ActiveContextService);
  private contextSwitch = inject(ContextSwitchService);
  private newThingFlows = inject(NewThingFlowsService);
  private pulldownControl = inject(PulldownControlService);
  private dashboardSort = inject(DashboardSortService);
  private dashSelection = inject(DashboardSelectionService);
  private runningJobs = inject(RunningJobsService);
  private cdr = inject(ChangeDetectorRef);

  readonly kind = input<PulldownKind>('dataset');

  readonly menuRef = viewChild<ElementRef<HTMLDivElement>>('menuRef');

  open = false;
  /** True on the browse (VTSBrowser) and find-results views, where the
   *  dataset/detector pair is fixed for the duration of the view. The
   *  pulldown then renders display-only: it still shows the active label
   *  but the trigger is disabled, so the pair can only be switched from the
   *  Dashboard. The label/train view is intentionally *not* locked — mid-
   *  training pair switching is the pulldown's original purpose. */
  locked = false;
  focusedIndex = -1;
  rows: PulldownRow[] = [];
  activeName = '';
  activeRowExists = false;
  registryError: string | null = null;

  /** Ids of this pulldown's rows that count as "active" — the selected
   *  table rows while the Dashboard is on screen, otherwise the single
   *  active/loaded id from `ActiveContextService`. Drives the active-row
   *  check icon, the closed-state label, and keyboard focus. */
  private selectedIds: string[] = [];

  /** True between an "Add New" click in this pulldown and the resulting
   *  registry-update that auto-selects the new item. Reset on success
   *  or on dismissal of the underlying modal. */
  private awaitingNew = false;
  /** Set when the modal we opened has emitted its success signal
   *  (`created$` for the detector flow, `importStarted$` for the dataset
   *  flow). Distinguishes a user-dismissal from a real submit so the
   *  modal-close handler knows whether to keep waiting. */
  private sawSuccessSignal = false;
  /** Dataset-flow only: snapshot of registry ids taken when the add
   *  started, so a fresh id can be detected once the load completes. */
  private knownIdsAtAddStart = new Set<string>();
  private prevImporterOpen = false;
  private prevDetectorOpen = false;

  private destroy$ = new Subject<void>();

  /** Live mirror of the Dashboard's sort for this pulldown's table. Kept
   *  in sync via a subscription in `ngOnInit`. */
  private sortState: SortState = { column: 'name', asc: true };

  /** Latest snapshot from `RunningJobsService.busyPairs$`. Re-read on every
   *  rebuild so a busy-pair tick re-renders the spinner glyph without
   *  needing a separate per-row subscription. */
  private busyPairs: Map<string, string[]> = new Map();

  constructor() {
    // On the Dashboard the bar shows the table selection instead of the
    // loaded context. Rebuild when visibility flips or either half's
    // selection changes (both halves matter here because a row's
    // compatibility/busy state depends on the *other* half's pick). An
    // `effect` rather than a subscription, so the bar stays live off its own
    // reads: nothing has to remember to push into it, and it keeps
    // repainting whether or not the Dashboard is mounted.
    effect(() => {
      this.dashSelection.dashboardVisible();
      this.dashSelection.datasetIds();
      this.dashSelection.detectorIds();
      this.rebuildRows();
    });
  }

  ngOnInit(): void {
    this.datasetState.datasets$.pipe(takeUntil(this.destroy$)).subscribe(() => {
      this.rebuildRows();
      this.maybeAutoSelectNewDataset();
    });
    this.datasetState.detectors$.pipe(takeUntil(this.destroy$)).subscribe(() => this.rebuildRows());
    // Read intent (not active) so the pulldown highlight updates the
    // moment the user picks a row, rather than waiting for any
    // dataset/detector load to finish.
    this.activeContext.intentPair$
      .pipe(takeUntil(this.destroy$))
      .subscribe(() => this.rebuildRows());
    this.datasetState.error$.pipe(takeUntil(this.destroy$)).subscribe((err) => {
      this.registryError = err;
      // Written from an async subscribe; notify the scheduler so the error row
      // repaints under zoneless.
      this.cdr.markForCheck();
    });

    this.newThingFlows.created$.pipe(takeUntil(this.destroy$)).subscribe(({ kind, id }) => {
      if (kind !== this.kind() || !id || !this.awaitingNew) return;
      this.sawSuccessSignal = true;
      this.awaitingNew = false;
      this.switchToNewItem(id);
    });
    if (this.isDataset) {
      this.newThingFlows.importStarted$.pipe(takeUntil(this.destroy$)).subscribe(() => {
        if (!this.awaitingNew) return;
        this.sawSuccessSignal = true;
        this.knownIdsAtAddStart = new Set(this.datasetState.datasets.map((d) => d.id));
      });
    }
    // Cancel the await if the user dismisses the underlying modal
    // without submitting. Without this, a later registry refresh from
    // an unrelated source would auto-select an unrelated dataset.
    this.newThingFlows.importer$.pipe(takeUntil(this.destroy$)).subscribe((state) => {
      const closing = this.prevImporterOpen && !state.open;
      this.prevImporterOpen = state.open;
      if (!closing || !this.isDataset) return;
      if (this.awaitingNew && !this.sawSuccessSignal) this.awaitingNew = false;
    });
    this.newThingFlows.newDetector$.pipe(takeUntil(this.destroy$)).subscribe((state) => {
      const closing = this.prevDetectorOpen && !state.open;
      this.prevDetectorOpen = state.open;
      if (!closing || this.isDataset) return;
      if (this.awaitingNew && !this.sawSuccessSignal) this.awaitingNew = false;
    });

    this.pulldownControl
      .openSignal$(this.kind())
      .pipe(takeUntil(this.destroy$))
      .subscribe(() => this.openMenu());

    // Mirror of the Dashboard table's sort. Read through
    // `DashboardSortService` rather than the owning `DashboardColumnsService`
    // so this eager component doesn't pull the column-management code onto
    // the initial bundle.
    const sortState$: Observable<SortState> = this.dashboardSort.sort$(this.kind());
    sortState$.pipe(takeUntil(this.destroy$)).subscribe((state) => {
      this.sortState = state;
      this.rebuildRows();
    });

    // Subscribe to the running-jobs poller so a job starting on another
    // pair lights up the spinner glyph here. The service polls lazily;
    // it only fires HTTP traffic while at least one component is
    // subscribed.
    this.runningJobs.busyPairs$.pipe(takeUntil(this.destroy$)).subscribe((pairs) => {
      this.busyPairs = pairs;
      this.rebuildRows();
    });

    // Lock the pulldown (display-only) on the browse / find views. Seed from
    // the current URL, then track navigations. Mirrors the router-URL view
    // detection in `AppComponent`.
    this.updateLocked(this.router.url);
    this.router.events
      .pipe(
        filter((e): e is NavigationEnd => e instanceof NavigationEnd),
        takeUntil(this.destroy$),
      )
      .subscribe((e) => this.updateLocked(e.urlAfterRedirects));

    this.rebuildRows();
  }

  /** Recompute {@link locked} from a URL. The browse and find views pin the
   *  active pair; the dashboard and label views leave the pulldown live. */
  private updateLocked(url: string): void {
    const locked = url.startsWith('/browse') || url.startsWith('/find');
    if (locked === this.locked) return;
    this.locked = locked;
    // A menu open when we navigate into a locked view must close.
    if (locked) this.close();
    // Written from the router-events subscribe (async callback), so notify the
    // scheduler to repaint the trigger's disabled state under zoneless.
    this.cdr.markForCheck();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  get isDataset(): boolean {
    return this.kind() === 'dataset';
  }

  get placeholderLabel(): string {
    return this.isDataset ? 'Select a dataset' : 'Select a detector';
  }

  get addNewLabel(): string {
    return this.isDataset ? '+ Add New Dataset' : '+ Add New Detector';
  }

  get fieldLabel(): string {
    return this.isDataset ? 'Data' : 'Detector';
  }

  get fieldTitle(): string {
    if (this.locked) {
      return this.isDataset
        ? 'Dataset is fixed here — return to the Dashboard to switch'
        : 'Detector is fixed here — return to the Dashboard to switch';
    }
    return this.isDataset
      ? 'Active dataset (click to switch)'
      : 'Active detector (click to switch)';
  }

  toggle(): void {
    if (this.open) this.close();
    else this.openMenu();
  }

  /**
   * Open the dropdown and pre-focus the first compatible row (or the
   * active row if all rows are incompatible / there's no other-half
   * context yet). Scrolls the focused row into view on the next tick.
   * Idempotent; calling on an already-open menu just re-focuses.
   */
  openMenu(): void {
    // Display-only on the browse / find views: no dropdown, no add-new. The
    // trigger button is also `disabled`, but this guard covers the programmatic
    // open path (`pulldownControl.openSignal$`) which bypasses the button.
    if (this.locked) return;
    this.open = true;
    const firstCompat = this.rows.findIndex((r) => r.compatibleWithOther);
    if (firstCompat >= 0) this.focusedIndex = firstCompat;
    else {
      const i = this.findActiveIndex();
      this.focusedIndex = i >= 0 ? i : -1;
    }
    setTimeout(() => this.scrollFocusedIntoView(), 0);
    // `openMenu` is also driven by the `pulldownControl.openSignal$` subscribe
    // (an unpatched callback), so notify the scheduler to open the menu under
    // zoneless. (Calls from bound click/keydown handlers already schedule CD.)
    this.cdr.markForCheck();
  }

  close(): void {
    this.open = false;
    this.focusedIndex = -1;
  }

  pickRow(row: PulldownRow): void {
    // On the Dashboard the pulldown shows the tables' selection: a pick is a
    // plain single-select of that row (which toggles off if it was the sole
    // pick), never a context load. Off the Dashboard it switches the
    // active/loaded pair as before.
    if (this.dashSelection.dashboardVisible()) {
      this.dashSelection.toggle(this.kind(), row.id, false);
      this.close();
      return;
    }
    if (row.active) {
      this.close();
      return;
    }
    if (this.isDataset) {
      this.contextSwitch.switchTo(row.id, this.activeContext.intentModelId);
    } else {
      this.contextSwitch.switchTo(this.activeContext.intentDatasetId, row.id);
    }
    this.close();
  }

  addNew(): void {
    this.close();
    this.awaitingNew = true;
    this.sawSuccessSignal = false;
    if (this.isDataset) {
      this.knownIdsAtAddStart = new Set(this.datasetState.datasets.map((d) => d.id));
      this.newThingFlows.openImporter();
    } else {
      // Seed the new-detector form's media type from the partner dataset:
      // the single selected dataset on the Dashboard, else the active one.
      const otherDsId = this.dashSelection.dashboardVisible()
        ? singleId(this.dashSelection.datasetIds())
        : this.activeContext.intentDatasetId;
      const other = otherDsId
        ? this.datasetState.datasets.find((d) => d.id === otherDsId)
        : null;
      this.newThingFlows.openNewDetector({
        defaultMediaType: other?.media_type || '',
        datasetEmbedder: other?.embedder || '',
      });
    }
  }

  retryRegistry(): void {
    this.datasetState.refresh();
  }

  /** Row click in the Add-New footer should swallow the click so the
   *  document-level outside-click listener doesn't immediately close
   *  the modal we just opened (modals listen on document too). */
  onMenuMouseDown(event: MouseEvent): void {
    event.stopPropagation();
  }

  onKeydown(event: KeyboardEvent): void {
    if (!this.open) {
      if (event.key === 'Enter' || event.key === ' ' || event.key === 'ArrowDown') {
        event.preventDefault();
        this.toggle();
      }
      return;
    }
    const rowsLen = this.rows.length;
    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        if (rowsLen === 0) {
          this.focusedIndex = -1; // footer
        } else if (this.focusedIndex === -1 || this.focusedIndex >= rowsLen - 1) {
          this.focusedIndex = 0;
        } else {
          this.focusedIndex += 1;
        }
        this.scrollFocusedIntoView();
        break;
      case 'ArrowUp':
        event.preventDefault();
        if (rowsLen === 0) {
          this.focusedIndex = -1;
        } else if (this.focusedIndex <= 0) {
          this.focusedIndex = rowsLen - 1;
        } else {
          this.focusedIndex -= 1;
        }
        this.scrollFocusedIntoView();
        break;
      case 'Home':
        event.preventDefault();
        if (rowsLen > 0) this.focusedIndex = 0;
        this.scrollFocusedIntoView();
        break;
      case 'End':
        event.preventDefault();
        if (rowsLen > 0) this.focusedIndex = rowsLen - 1;
        this.scrollFocusedIntoView();
        break;
      case 'Enter':
      case ' ':
        event.preventDefault();
        if (this.focusedIndex >= 0 && this.focusedIndex < rowsLen) {
          this.pickRow(this.rows[this.focusedIndex]);
        } else {
          this.addNew();
        }
        break;
      case 'Escape':
        event.preventDefault();
        this.close();
        break;
    }
  }

  @HostListener('document:click', ['$event'])
  onDocumentClick(event: Event): void {
    if (!this.open) return;
    const target = event.target as Node | null;
    if (target && this.host.nativeElement.contains(target)) return;
    this.close();
  }

  glyphFor(row: PulldownRow): string {
    // The active state renders a `vt-icon` check in the template; this getter
    // only covers the loaded (●) / unloaded (○) dot for inactive rows.
    return row.loaded ? '●' : '○';
  }

  glyphTitle(row: PulldownRow): string {
    if (row.active) return 'Currently active';
    if (row.loaded) return 'Loaded in memory';
    return 'Not loaded (click to load)';
  }

  trackRow(_: number, row: PulldownRow): string {
    return row.id;
  }

  /** Sort registry entries the same way the Dashboard's tables sort them
   *  (column + direction read from `DashboardSortService`). Mirrors
   *  the comparator in `DashboardComponent.sortedDatasets` /
   *  `sortedDetectors`. */
  private applySort<T>(arr: T[]): T[] {
    const { column, asc } = this.sortState;
    return sortRowsByColumn(arr, column, asc);
  }

  private findActiveIndex(): number {
    // First active row (there can be several under Dashboard multi-select);
    // used only to seed keyboard focus, so the first is fine.
    return this.rows.findIndex((r) => r.active);
  }

  private scrollFocusedIntoView(): void {
    const menu = this.menuRef()?.nativeElement;
    if (!menu || this.focusedIndex < 0) return;
    const rowEl = menu.querySelectorAll('.pulldown-row')[this.focusedIndex] as
      | HTMLElement
      | undefined;
    rowEl?.scrollIntoView({ block: 'nearest' });
  }

  private maybeAutoSelectNewDataset(): void {
    if (!this.isDataset || !this.awaitingNew || !this.sawSuccessSignal) return;
    const newOne = this.datasetState.datasets.find(
      (d) => !this.knownIdsAtAddStart.has(d.id),
    );
    if (!newOne) return;
    this.awaitingNew = false;
    this.sawSuccessSignal = false;
    this.switchToNewItem(newOne.id);
  }

  private switchToNewItem(id: string): void {
    // On the Dashboard a freshly-added item becomes the selected row (the
    // Dashboard also auto-selects new ids, so this just keeps them aligned)
    // rather than being loaded as the active pair. `selectOnly`, not the pick
    // ladder: this is "make the new thing the selection", and the ladder would
    // toggle it back *off* if the Dashboard's own auto-select got there first.
    if (this.dashSelection.dashboardVisible()) {
      this.dashSelection.selectOnly(this.kind(), [id]);
      return;
    }
    if (this.isDataset) {
      this.contextSwitch.switchTo(id, this.activeContext.intentModelId);
    } else {
      this.contextSwitch.switchTo(this.activeContext.intentDatasetId, id);
    }
  }

  private rebuildRows(): void {
    const datasets = this.datasetState.datasets;
    const detectors = this.datasetState.detectors;
    // Two sources for what counts as "active":
    //  - On the Dashboard, the highlighted table rows (which can be more
    //    than one — the closed label collapses that to "Multiple").
    //  - Elsewhere, the single intent id (what the user picked), not what's
    //    loaded, so picking a row feels instant while a load runs behind it.
    const dashMode = this.dashSelection.dashboardVisible();
    const dsIds = dashMode
      ? this.dashSelection.datasetIds()
      : idList(this.activeContext.intentDatasetId);
    const detIds = dashMode
      ? this.dashSelection.detectorIds()
      : idList(this.activeContext.intentModelId);

    // The row's own selected set (highlight + label) versus the *other*
    // half's single active id (compat/busy). Compatibility dimming needs an
    // unambiguous partner, so a multi-selected other half reads as "none".
    this.selectedIds = this.isDataset ? dsIds : detIds;
    const selected = new Set(this.selectedIds);
    if (this.isDataset) {
      const activeDetId = singleId(detIds);
      const activeDetector = activeDetId ? detectors.find((d) => d.id === activeDetId) : null;
      const sorted = this.applySort(datasets);
      this.rows = sorted.map((d) => this.datasetRow(d, selected, activeDetector));
    } else {
      const activeDsId = singleId(dsIds);
      const activeDataset = activeDsId ? datasets.find((d) => d.id === activeDsId) : null;
      const sorted = this.applySort(detectors);
      this.rows = sorted.map((d) => this.detectorRow(d, selected, activeDataset));
    }

    // Closed-state label: nothing → placeholder; one → its name; many →
    // "Multiple" (only reachable in Dashboard multi-select).
    const activeRows = this.rows.filter((r) => r.active);
    this.activeRowExists = activeRows.length > 0;
    if (activeRows.length === 1) this.activeName = activeRows[0].name;
    else if (activeRows.length > 1) this.activeName = 'Multiple';
    else this.activeName = '';
    if (this.open) {
      const i = this.findActiveIndex();
      if (i !== -1) this.focusedIndex = i;
      else if (this.focusedIndex >= this.rows.length) this.focusedIndex = -1;
    }
    // Driven by the registry `datasets$`/`detectors$`/`intentPair$`/`busyPairs$`
    // subscribes (unpatched callbacks), so notify the scheduler to repaint the
    // row list / active-name chip under zoneless.
    this.cdr.markForCheck();
  }

  private datasetRow(
    dataset: DatasetRegistryEntry,
    activeIds: Set<string>,
    activeDetector: DetectorRegistryEntry | null | undefined,
  ): PulldownRow {
    const compatible = activeDetector
      ? isPairCompatible(dataset, activeDetector)
      : true;
    let reason = '';
    if (!compatible && activeDetector) {
      reason = `Active detector "${activeDetector.name}" works on ${activeDetector.media_type}; this dataset is ${dataset.media_type}.`;
    }
    // Per-pair busy check: a dataset row shows the spinner when paired
    // with the currently-active detector half has a running job. We
    // never show a spinner on a row whose "other half" is empty; there
    // is no real pair to attach work to.
    const otherId = activeDetector?.id || '';
    const busyJobTypes = otherId ? this.busyPairs.get(pairKey(dataset.id, otherId)) || [] : [];
    return {
      id: dataset.id,
      name: dataset.name,
      mediaType: dataset.media_type,
      loaded: !!dataset.loaded,
      active: activeIds.has(dataset.id),
      compatibleWithOther: compatible,
      incompatReason: reason,
      busy: busyJobTypes.length > 0,
      busyJobTypes,
    };
  }

  private detectorRow(
    detector: DetectorRegistryEntry,
    activeIds: Set<string>,
    activeDataset: DatasetRegistryEntry | null | undefined,
  ): PulldownRow {
    const compatible = activeDataset
      ? isPairCompatible(activeDataset, detector)
      : true;
    let reason = '';
    if (!compatible && activeDataset) {
      reason = `This detector embeds ${detector.media_type}; active dataset "${activeDataset.name}" is ${activeDataset.media_type}.`;
    }
    const otherId = activeDataset?.id || '';
    const busyJobTypes = otherId ? this.busyPairs.get(pairKey(otherId, detector.id)) || [] : [];
    return {
      id: detector.id,
      name: detector.name,
      mediaType: detector.media_type,
      loaded: !!detector.detector_loaded,
      active: activeIds.has(detector.id),
      compatibleWithOther: compatible,
      incompatReason: reason,
      busy: busyJobTypes.length > 0,
      busyJobTypes,
    };
  }

  busyTitle(row: PulldownRow): string {
    if (!row.busy) return '';
    const labels: Record<string, string> = {
      'learned-sort': 'Learned sort',
      eval: 'Eval indicator',
    };
    const names = row.busyJobTypes.map((t) => labels[t] || t).join(', ');
    return `${names} running on this pair`;
  }
}
