import { Component, ElementRef, HostListener, Input, OnDestroy, OnInit, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Observable, Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { DatasetStateService } from '../../services/dataset-state.service';
import { ActiveContextService } from '../../services/active-context.service';
import { ContextSwitchService } from '../../services/context-switch.service';
import { NewThingFlowsService } from '../../services/new-thing-flows.service';
import { PulldownControlService } from '../../services/pulldown-control.service';
import { DashboardColumnsService } from '../../services/dashboard-columns.service';
import { RunningJobsService, pairKey } from '../../services/running-jobs.service';
import { SortState } from '../../utils/managed-columns';
import { DatasetRegistryEntry, DetectorRegistryEntry } from '../../models/api.models';
import { isPairCompatible } from '../../utils/context-compat';

type PulldownKind = 'dataset' | 'detector';

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
  selector: 'vt-context-pulldown',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './context-pulldown.component.html',
  styleUrl: './context-pulldown.component.scss',
})
export class ContextPulldownComponent implements OnInit, OnDestroy {
  @Input() kind: PulldownKind = 'dataset';

  @ViewChild('menuRef') menuRef?: ElementRef<HTMLDivElement>;

  open = false;
  focusedIndex = -1;
  rows: PulldownRow[] = [];
  activeName = '';
  activeRowExists = false;
  registryError: string | null = null;

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

  constructor(
    private host: ElementRef<HTMLElement>,
    private datasetState: DatasetStateService,
    private activeContext: ActiveContextService,
    private contextSwitch: ContextSwitchService,
    private newThingFlows: NewThingFlowsService,
    private pulldownControl: PulldownControlService,
    private dashboardColumns: DashboardColumnsService,
    private runningJobs: RunningJobsService,
  ) {}

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
    });

    this.newThingFlows.created$.pipe(takeUntil(this.destroy$)).subscribe(({ kind, id }) => {
      if (kind !== this.kind || !id || !this.awaitingNew) return;
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
      .openSignal$(this.kind)
      .pipe(takeUntil(this.destroy$))
      .subscribe(() => this.openMenu());

    // Erase the column-type union to a plain `SortState`; the pulldown
    // doesn't care which specific column-type union the source carries.
    const sortState$: Observable<SortState> = this.isDataset
      ? this.dashboardColumns.datasetCols.sortState$
      : this.dashboardColumns.detectorCols.sortState$;
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

    this.rebuildRows();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  get isDataset(): boolean {
    return this.kind === 'dataset';
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
    this.open = true;
    const firstCompat = this.rows.findIndex((r) => r.compatibleWithOther);
    if (firstCompat >= 0) this.focusedIndex = firstCompat;
    else {
      const i = this.findActiveIndex();
      this.focusedIndex = i >= 0 ? i : -1;
    }
    setTimeout(() => this.scrollFocusedIntoView(), 0);
  }

  close(): void {
    this.open = false;
    this.focusedIndex = -1;
  }

  pickRow(row: PulldownRow): void {
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
      const other = this.activeContext.intentDatasetId
        ? this.datasetState.datasets.find((d) => d.id === this.activeContext.intentDatasetId)
        : null;
      this.newThingFlows.openNewDetector({ defaultMediaType: other?.media_type || '' });
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
    if (row.active) return '✓';
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
   *  (column + direction read from `DashboardColumnsService`). Mirrors
   *  the comparator in `DashboardComponent.sortedDatasets` /
   *  `sortedDetectors`. */
  private applySort<T extends { name: string; [k: string]: unknown }>(arr: T[]): T[] {
    const { column, asc } = this.sortState;
    const dir = asc ? 1 : -1;
    return [...arr].sort((a, b) => {
      const va = a[column] ?? '';
      const vb = b[column] ?? '';
      if (typeof va === 'number' && typeof vb === 'number') return (va - vb) * dir;
      return String(va).localeCompare(String(vb)) * dir;
    });
  }

  private findActiveIndex(): number {
    const id = this.isDataset
      ? this.activeContext.intentDatasetId
      : this.activeContext.intentModelId;
    return this.rows.findIndex((r) => r.id === id);
  }

  private scrollFocusedIntoView(): void {
    const menu = this.menuRef?.nativeElement;
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
    if (this.isDataset) {
      this.contextSwitch.switchTo(id, this.activeContext.intentModelId);
    } else {
      this.contextSwitch.switchTo(this.activeContext.intentDatasetId, id);
    }
  }

  private rebuildRows(): void {
    const datasets = this.datasetState.datasets;
    const detectors = this.datasetState.detectors;
    // Highlight rows by intent (what the user picked), not by what's
    // currently loaded; picking a row should feel instant even when
    // a dataset/detector load is still running behind the scenes.
    const activeDsId = this.activeContext.intentDatasetId;
    const activeDetId = this.activeContext.intentModelId;
    const activeDataset = activeDsId ? datasets.find((d) => d.id === activeDsId) : null;
    const activeDetector = activeDetId ? detectors.find((d) => d.id === activeDetId) : null;

    if (this.isDataset) {
      const sorted = this.applySort(datasets);
      this.rows = sorted.map((d) => this.datasetRow(d, activeDsId, activeDetector));
    } else {
      const sorted = this.applySort(detectors);
      this.rows = sorted.map((d) => this.detectorRow(d, activeDetId, activeDataset));
    }

    const active = this.rows.find((r) => r.active);
    this.activeRowExists = !!active;
    this.activeName = active?.name || '';
    if (this.open) {
      const i = this.findActiveIndex();
      if (i !== -1) this.focusedIndex = i;
      else if (this.focusedIndex >= this.rows.length) this.focusedIndex = -1;
    }
  }

  private datasetRow(
    dataset: DatasetRegistryEntry,
    activeId: string,
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
      active: dataset.id === activeId,
      compatibleWithOther: compatible,
      incompatReason: reason,
      busy: busyJobTypes.length > 0,
      busyJobTypes,
    };
  }

  private detectorRow(
    detector: DetectorRegistryEntry,
    activeId: string,
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
      active: detector.id === activeId,
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
