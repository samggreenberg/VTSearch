import { Component, HostListener, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { NavigationCancel, NavigationEnd, NavigationError, Router } from '@angular/router';
import { EMPTY, Subject, timer } from 'rxjs';
import { catchError, filter, switchMap, take, takeUntil } from 'rxjs/operators';
import { DatasetsApiService } from '../../services/datasets-api.service';
import { DetectorsApiService } from '../../services/detectors-api.service';
import { ProgressEventsService } from '../../services/progress-events.service';
import { VtDialogService } from '../../services/dialog.service';
import { LabelSessionService } from '../../services/label-session.service';
import { DatasetStateService } from '../../services/dataset-state.service';
import { ActiveContextService } from '../../services/active-context.service';
import { ContextSwitchService } from '../../services/context-switch.service';
import { AuthService } from '../../services/auth.service';
import { TopBarStateService } from '../../services/top-bar-state.service';
import { NewThingFlowsService } from '../../services/new-thing-flows.service';
import { AchievementsService } from '../../services/achievements.service';
import { AutoDetectResultsData, DatasetRegistryEntry, DemoDataset, LoadingTask, DetectorRegistryEntry, ImporterInfo } from '../../models/api.models';
import { formatProgressFraction } from '../../utils/format-progress';
import {
  DashboardColumnsService,
  DatasetColumn,
  DetectorColumn,
} from '../../services/dashboard-columns.service';
import { ProgressBarComponent } from '../progress-bar/progress-bar.component';
import { AutoDetectResultsModalComponent } from '../modals/autodetect-results-modal/autodetect-results-modal.component';
import { DatasetCardComponent } from './dataset-card/dataset-card.component';
import { DetectorCardComponent } from './detector-card/detector-card.component';
import { CombineDatasetsModalComponent } from './combine-datasets-modal/combine-datasets-modal.component';
import { CombineDetectorsModalComponent } from './combine-detectors-modal/combine-detectors-modal.component';
import { LabelExporterModalComponent } from '../modals/label-exporter-modal/label-exporter-modal.component';
import { LabelImporterModalComponent } from '../modals/label-importer-modal/label-importer-modal.component';
import { DatasetStatsModalComponent } from '../modals/dataset-stats-modal/dataset-stats-modal.component';
import { IconComponent } from '../icon/icon.component';
import { DiskUsageComponent, DiskUsageBytes } from './disk-usage/disk-usage.component';

@Component({
  selector: 'vt-dashboard',
  standalone: true,
  imports: [
    CommonModule,
    ProgressBarComponent,
    AutoDetectResultsModalComponent,
    DatasetCardComponent,
    DetectorCardComponent,
    CombineDatasetsModalComponent,
    CombineDetectorsModalComponent,
    LabelExporterModalComponent,
    LabelImporterModalComponent,
    DatasetStatsModalComponent,
    IconComponent,
    DiskUsageComponent,
  ],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.scss',
})
export class DashboardComponent implements OnInit, OnDestroy {
  selectedDatasetIds: Set<string> = new Set();
  selectedDetectorIds: Set<string> = new Set();

  // Confirm flags hold the trash icon at 90° while the confirm dialog is up,
  // and play a reverse animation back to 0° once the dialog resolves.
  deletingSelectedDatasetsConfirm = false;
  wasDeletingSelectedDatasetsConfirm = false;
  deletingSelectedDetectorsConfirm = false;
  wasDeletingSelectedDetectorsConfirm = false;

  progressValue = 0;
  progressTotal = 0;
  progressIndeterminate = false;

  loadingTasks: LoadingTask[] = [];
  detectorLoadingTasks: LoadingTask[] = [];

  importerClosing = false;
  /** Visible importers (i.e. ``hidden_from_picker !== true``), fetched
   *  once on init so the welcome banner can detect Services-tab
   *  importers without waiting for the modal to open. */
  visibleImporters: ImporterInfo[] = [];
  combineModalOpen = false;
  /** Datasets passed into the Combine modal when it opens. */
  combineModalDatasets: DatasetRegistryEntry[] = [];
  newDetectorClosing = false;
  combineDetectorsModalOpen = false;
  exportModalOpen = false;
  exportDetectorName = '';
  addLabelsModalOpen = false;
  addLabelsDetectorName = '';
  findResultsOpen = false;
  findResultsData: AutoDetectResultsData = { results: {} };
  statsModalOpen = false;
  statsDatasetId = '';
  statsDatasetName = '';
  deletingDatasetId = '';
  deletingDetectorId = '';
  addLabelsDetectorId = '';
  trainAfterModelCreation = false;
  /** Mirrors the user's click intent so the Train/Find button icons
   *  can waggle while the `activeContextGuard` waits between click and
   *  route activation. Reset when `contextSwitch.switching$` settles
   *  back to false (either route activated or canActivate denied). */
  trainLoading = false;
  findLoading = false;

  /** Lifted out of this component into `DashboardColumnsService` so the
   *  top-bar context pulldowns can mirror the user's sort. Assigned in
   *  the constructor; the template binds against these names unchanged. */
  datasetCols!: DashboardColumnsService['datasetCols'];
  detectorCols!: DashboardColumnsService['detectorCols'];

  get visibleDatasetColumns(): DatasetColumn[] {
    if (this.isDefaultLogin) {
      return this.datasetCols.columnOrder.filter((c) => c !== 'created_by' && c !== 'readers');
    }
    return this.datasetCols.columnOrder;
  }

  get visibleDetectorColumns(): DetectorColumn[] {
    return this.detectorCols.columnOrder;
  }

  onDatasetHeaderClick(col: string): void {
    if (this.datasetCols.meta(col).sortable) {
      this.datasetCols.sortBy(col as DatasetColumn);
    }
  }

  onDetectorHeaderClick(col: string): void {
    if (this.detectorCols.meta(col).sortable) {
      this.detectorCols.sortBy(col as DetectorColumn);
    }
  }

  private destroy$ = new Subject<void>();
  private polling$ = new Subject<void>();
  private detectorPolling$ = new Subject<void>();
  private findPolling$ = new Subject<void>();
  private knownDatasetIds = new Set<string>();
  private knownDetectorIds = new Set<string>();
  private completedTaskIds = new Set<string>();
  private completedModelTaskIds = new Set<string>();
  private datasetPollingActive = false;
  private detectorPollingActive = false;

  currentUser = '';
  isDefaultLogin = true;

  diskUsage: DiskUsageBytes | null = null;

  constructor(
    private router: Router,
    private datasetsApi: DatasetsApiService,
    private detectorsApi: DetectorsApiService,
    private dialog: VtDialogService,
    private labelSession: LabelSessionService,
    public datasetState: DatasetStateService,
    private activeContext: ActiveContextService,
    private contextSwitch: ContextSwitchService,
    private authService: AuthService,
    private topBarState: TopBarStateService,
    private newThingFlows: NewThingFlowsService,
    private achievements: AchievementsService,
    private progressEvents: ProgressEventsService,
    columnsService: DashboardColumnsService,
  ) {
    this.datasetCols = columnsService.datasetCols;
    this.detectorCols = columnsService.detectorCols;
  }

  ngOnInit(): void {
    this.authService.status$
      .pipe(takeUntil(this.destroy$))
      .subscribe((status) => {
        this.currentUser = status?.user || '';
        this.isDefaultLogin = status?.provider === 'default';
      });
    // Auto-select newly added items whenever the dataset/model lists change
    this.datasetState.datasets$
      .pipe(takeUntil(this.destroy$))
      .subscribe((datasets) => {
        const currentIds = new Set(datasets.map((d) => d.id));
        // Prune selections that no longer exist in the registry
        for (const id of this.selectedDatasetIds) {
          if (!currentIds.has(id)) this.selectedDatasetIds.delete(id);
        }
        const newIds = [...currentIds].filter((id) => !this.knownDatasetIds.has(id));
        if (newIds.length > 0 && this.knownDatasetIds.size > 0) {
          // Items were added after initial load — select only the new ones
          this.selectedDatasetIds.clear();
          for (const id of newIds) {
            this.selectedDatasetIds.add(id);
          }
        } else if (datasets.length === 1 && this.selectedDatasetIds.size === 0) {
          // First load with exactly one item — auto-select it
          this.selectedDatasetIds.add(datasets[0].id);
        }
        this.knownDatasetIds = currentIds;
        this.pushTopBarLabels();
      });
    this.datasetState.detectors$
      .pipe(takeUntil(this.destroy$))
      .subscribe((models) => {
        const currentIds = new Set(models.map((m) => m.id));
        // Prune selections that no longer exist in the registry
        for (const id of this.selectedDetectorIds) {
          if (!currentIds.has(id)) this.selectedDetectorIds.delete(id);
        }
        const newIds = [...currentIds].filter((id) => !this.knownDetectorIds.has(id));
        if (newIds.length > 0 && this.knownDetectorIds.size > 0) {
          // Items were added after initial load — select only the new ones
          this.selectedDetectorIds.clear();
          for (const id of newIds) {
            this.selectedDetectorIds.add(id);
          }
        } else if (models.length === 1 && this.selectedDetectorIds.size === 0) {
          // First load with exactly one item — auto-select it
          this.selectedDetectorIds.add(models[0].id);
        }
        this.knownDetectorIds = currentIds;
        this.pushTopBarLabels();
      });
    this.refresh();
    this.resumeActivePolling();
    this.startDiskUsagePolling();
    this.datasetsApi.getAllImporters().subscribe({
      next: (res) => {
        this.visibleImporters = (res.importers || []).filter((imp) => !imp['hidden_from_picker']);
      },
    });
    this.subscribeNewThingFlows();
    // Reset the per-button waggle flags whenever a navigation
    // initiated by onLabel/onFind settles (succeeded, cancelled by the
    // guard, or errored). Without this, a guard redirect to /dashboard
    // would leave the icon waggling forever.
    this.router.events
      .pipe(takeUntil(this.destroy$))
      .subscribe((e) => {
        if (
          e instanceof NavigationEnd ||
          e instanceof NavigationCancel ||
          e instanceof NavigationError
        ) {
          this.trainLoading = false;
          this.findLoading = false;
        }
      });
  }

  /** Translate `NewThingFlowsService` events into the legacy Dashboard
   *  behaviors: kick off background polls after an import, auto-select
   *  newly created detectors, drive the `+` icon's close animation,
   *  etc. The importer / new-detector modals themselves are hosted on
   *  `AppComponent`; this component just reacts to their outcomes. */
  private subscribeNewThingFlows(): void {
    this.newThingFlows.importStarted$
      .pipe(takeUntil(this.destroy$))
      .subscribe(() => this.handleImportStarted());
    this.newThingFlows.demoSelected$
      .pipe(takeUntil(this.destroy$))
      .subscribe(({ demo }) => this.handleDemoSelected(demo));
    this.newThingFlows.created$
      .pipe(takeUntil(this.destroy$))
      .subscribe(({ kind, id }) => {
        if (kind === 'detector') this.handleDetectorCreated(id);
        else this.datasetState.refresh();
      });
    // Trigger the close-animation when the modal flips from open → closed.
    let importerWasOpen = false;
    this.newThingFlows.importer$
      .pipe(takeUntil(this.destroy$))
      .subscribe((state) => {
        if (importerWasOpen && !state.open) this.importerClosing = true;
        importerWasOpen = state.open;
      });
    let detectorWasOpen = false;
    this.newThingFlows.newDetector$
      .pipe(takeUntil(this.destroy$))
      .subscribe((state) => {
        if (detectorWasOpen && !state.open) {
          this.newDetectorClosing = true;
          // Reset the "navigate to Train after create" intent if the
          // user dismissed without creating. The handler in
          // handleDetectorCreated() also clears it on success.
          this.trainAfterModelCreation = false;
        }
        detectorWasOpen = state.open;
      });
  }

  private handleDetectorCreated(modelId: string): void {
    this.datasetState.refresh();
    if (this.trainAfterModelCreation && modelId) {
      this.trainAfterModelCreation = false;
      this.selectedDetectorIds.clear();
      this.selectedDetectorIds.add(modelId);
      this.knownDetectorIds.add(modelId);
      this.datasetState.detectors$
        .pipe(
          filter((models) => models.some((m) => m.id === modelId)),
          take(1),
          takeUntil(this.destroy$),
        )
        .subscribe(() => this.onLabel());
    }
  }

  private startDiskUsagePolling(): void {
    timer(0, 10000)
      .pipe(
        takeUntil(this.destroy$),
        switchMap(() => this.datasetsApi.getDiskUsage().pipe(catchError(() => EMPTY))),
      )
      .subscribe((usage) => {
        this.diskUsage = { total: usage.total, used: usage.used, free: usage.free };
      });
  }

  /** Check for in-progress loading tasks (e.g. after a page reload) and start watching. */
  private resumeActivePolling(): void {
    // SSE pushes the initial snapshot the moment we connect, so the first
    // event on each channel tells us whether there's anything in flight.
    this.progressEvents.loadingTasks$
      .pipe(filter((tasks) => tasks.some((t) => t.status !== 'idle')), take(1), takeUntil(this.destroy$))
      .subscribe(() => this.startProgressPolling());
    this.progressEvents.detectorLoadingTasks$
      .pipe(filter((tasks) => tasks.some((t) => t.status !== 'idle')), take(1), takeUntil(this.destroy$))
      .subscribe(() => this.startDetectorProgressPolling());
  }

  // --- Column resize / drag-reorder ---
  //
  // The actual logic lives in `ManagedColumns`. We just forward document-level
  // mouse events to both managers so resize tracking works regardless of which
  // table the user grabbed. Drag-reorder uses native HTML5 drag events and is
  // dispatched directly from the template.

  @HostListener('document:mousemove', ['$event'])
  onColResizeMove(event: MouseEvent): void {
    this.datasetCols.onResizeMove(event);
    this.detectorCols.onResizeMove(event);
  }

  @HostListener('document:mouseup')
  onColResizeEnd(): void {
    this.datasetCols.onResizeEnd();
    this.detectorCols.onResizeEnd();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
    this.polling$.next();
    this.polling$.complete();
    this.detectorPolling$.next();
    this.detectorPolling$.complete();
    this.findPolling$.next();
    this.findPolling$.complete();
  }

  get datasets(): DatasetRegistryEntry[] {
    return this.datasetState.datasets;
  }

  get detectors(): DetectorRegistryEntry[] {
    return this.datasetState.detectors;
  }

  get loading(): boolean {
    return this.datasetState.loading;
  }

  get progressMessage(): string {
    return this.datasetState.progressMessage;
  }

  /** Map dataset_id → LoadingTask for tasks that match an existing dataset row. */
  get inlineTaskMap(): Map<string, LoadingTask> {
    const map = new Map<string, LoadingTask>();
    const datasetIds = new Set(this.datasets.map((d) => d.id));
    for (const task of this.loadingTasks) {
      if (task.dataset_id && datasetIds.has(task.dataset_id)) {
        map.set(task.dataset_id, task);
      }
    }
    return map;
  }

  /** Loading tasks that have no matching dataset row (new imports, etc.). */
  get orphanLoadingTasks(): LoadingTask[] {
    const datasetIds = new Set(this.datasets.map((d) => d.id));
    return this.loadingTasks.filter((t) => !t.dataset_id || !datasetIds.has(t.dataset_id));
  }

  getInlineTask(datasetId: string): LoadingTask | undefined {
    return this.inlineTaskMap.get(datasetId);
  }

  refresh(): void {
    this.datasetState.refresh();
  }

  // --- Dataset selection ---

  private pushTopBarLabels(): void {
    const selDatasets = this.datasets.filter((d) => this.selectedDatasetIds.has(d.id));
    if (selDatasets.length === 0) this.topBarState.setDatasetLabel('None');
    else if (selDatasets.length === 1) this.topBarState.setDatasetLabel(selDatasets[0].name);
    else this.topBarState.setDatasetLabel('Multiple');

    const selModels = this.detectors.filter((d) => this.selectedDetectorIds.has(d.id));
    if (selModels.length === 0) this.topBarState.setModelLabel('None');
    else if (selModels.length === 1) this.topBarState.setModelLabel(selModels[0].name);
    else this.topBarState.setModelLabel('Multiple');
  }

  toggleDatasetSelection(id: string, event: MouseEvent): void {
    if (event.ctrlKey || event.metaKey) {
      if (this.selectedDatasetIds.has(id)) {
        this.selectedDatasetIds.delete(id);
      } else {
        this.selectedDatasetIds.add(id);
      }
    } else {
      if (this.selectedDatasetIds.has(id) && this.selectedDatasetIds.size === 1) {
        this.selectedDatasetIds.clear();
      } else {
        this.selectedDatasetIds.clear();
        this.selectedDatasetIds.add(id);
      }
    }
    this.pushTopBarLabels();
  }

  isDatasetSelected(id: string): boolean {
    return this.selectedDatasetIds.has(id);
  }

  toggleDatasetCheckbox(id: string): void {
    if (this.selectedDatasetIds.has(id)) {
      this.selectedDatasetIds.delete(id);
    } else {
      this.selectedDatasetIds.add(id);
    }
    this.pushTopBarLabels();
  }

  get datasetSelectionState(): 'none' | 'some' | 'all' {
    const sel = this.selectedDatasetIds.size;
    if (sel === 0) return 'none';
    if (sel >= this.datasets.length) return 'all';
    return 'some';
  }

  toggleAllDatasets(): void {
    if (this.datasetSelectionState === 'all') {
      this.selectedDatasetIds.clear();
    } else {
      for (const d of this.datasets) {
        this.selectedDatasetIds.add(d.id);
      }
    }
    this.pushTopBarLabels();
  }

  /**
   * Open the dedicated Combine Datasets modal pre-loaded with the
   * currently-selected datasets. The modal handles its own validation,
   * row removal, and submission.
   */
  combineSelectedDatasets(): void {
    const targets = this.datasets.filter((d) => this.selectedDatasetIds.has(d.id));
    if (targets.length < 2) return;
    this.combineModalDatasets = targets;
    this.combineModalOpen = true;
  }

  closeCombineModal(): void {
    this.combineModalOpen = false;
    this.combineModalDatasets = [];
  }

  onCombineStarted(): void {
    this.closeCombineModal();
    this.startProgressPolling();
  }

  /**
   * True when the "Combine Selected Datasets" button should be enabled:
   * at least two datasets selected AND all of them share a media type.
   */
  get combineSelectedDatasetsEnabled(): boolean {
    if (this.selectedDatasetIds.size < 2) return false;
    const targets = this.datasets.filter((d) => this.selectedDatasetIds.has(d.id));
    if (targets.length < 2) return false;
    const types = new Set(targets.map((d) => d.media_type));
    return types.size === 1;
  }

  /**
   * Hint shown in the Combine button's tooltip explaining why it's
   * disabled (or describing the action when enabled).
   */
  get combineSelectedDatasetsHint(): string {
    if (this.selectedDatasetIds.size < 2) {
      return 'Select two or more datasets to combine';
    }
    if (!this.combineSelectedDatasetsEnabled) {
      return 'All selected datasets must be of the same media type';
    }
    return 'Combine selected datasets into a new one';
  }

  async deleteSelectedDatasets(): Promise<void> {
    const ids = [...this.selectedDatasetIds];
    if (ids.length === 0) return;
    const targets = this.datasets.filter((d) => this.selectedDatasetIds.has(d.id));
    if (targets.length === 0) return;
    const names = targets.map((d) => `"${d.name}"`).join(', ');
    this.deletingSelectedDatasetsConfirm = true;
    let ok = false;
    try {
      const question =
        targets.length === 1
          ? `Delete dataset ${names}?`
          : `Delete ${targets.length} datasets: ${names}?`;
      const detail =
        targets.length === 1
          ? 'This removes the dataset from the registry and deletes its cached pickle file. Detectors trained against it are unaffected.'
          : 'This removes the datasets from the registry and deletes their cached pickle files. Detectors trained against them are unaffected.';
      ok = await this.dialog.confirmDestructive(question, detail);
    } finally {
      this.deletingSelectedDatasetsConfirm = false;
      this.wasDeletingSelectedDatasetsConfirm = true;
    }
    if (!ok) return;
    for (const dataset of targets) {
      this.datasetsApi.deleteRegistered(dataset.id).subscribe({
        next: () => {
          this.selectedDatasetIds.delete(dataset.id);
          this.datasetState.refresh();
        },
      });
    }
  }

  onDeleteSelectedDatasetsAnimationEnd(): void {
    if (!this.deletingSelectedDatasetsConfirm) {
      this.wasDeletingSelectedDatasetsConfirm = false;
    }
  }

  // --- Model selection ---

  toggleDetectorSelection(id: string, event: MouseEvent): void {
    if (event.ctrlKey || event.metaKey) {
      if (this.selectedDetectorIds.has(id)) {
        this.selectedDetectorIds.delete(id);
      } else {
        this.selectedDetectorIds.add(id);
      }
    } else {
      if (this.selectedDetectorIds.has(id) && this.selectedDetectorIds.size === 1) {
        this.selectedDetectorIds.clear();
      } else {
        this.selectedDetectorIds.clear();
        this.selectedDetectorIds.add(id);
      }
    }
    this.pushTopBarLabels();
  }

  isDetectorSelected(id: string): boolean {
    return this.selectedDetectorIds.has(id);
  }

  toggleDetectorCheckbox(id: string): void {
    if (this.selectedDetectorIds.has(id)) {
      this.selectedDetectorIds.delete(id);
    } else {
      this.selectedDetectorIds.add(id);
    }
    this.pushTopBarLabels();
  }

  get detectorSelectionState(): 'none' | 'some' | 'all' {
    const sel = this.selectedDetectorIds.size;
    if (sel === 0) return 'none';
    if (sel >= this.detectors.length) return 'all';
    return 'some';
  }

  toggleAllDetectors(): void {
    if (this.detectorSelectionState === 'all') {
      this.selectedDetectorIds.clear();
    } else {
      for (const m of this.detectors) {
        this.selectedDetectorIds.add(m.id);
      }
    }
    this.pushTopBarLabels();
  }

  async deleteSelectedDetectors(): Promise<void> {
    const ids = [...this.selectedDetectorIds];
    if (ids.length === 0) return;
    const targets = this.detectors.filter((d) => this.selectedDetectorIds.has(d.id));
    if (targets.length === 0) return;
    const names = targets.map((m) => `"${m.name}"`).join(', ');
    this.deletingSelectedDetectorsConfirm = true;
    let ok = false;
    try {
      const question =
        targets.length === 1
          ? `Delete detector ${names}?`
          : `Delete ${targets.length} detectors: ${names}?`;
      const detail =
        targets.length === 1
          ? 'This removes its labelset and training metadata. The dataset is unaffected.'
          : 'This removes their labelsets and training metadata. The datasets are unaffected.';
      ok = await this.dialog.confirmDestructive(question, detail);
    } finally {
      this.deletingSelectedDetectorsConfirm = false;
      this.wasDeletingSelectedDetectorsConfirm = true;
    }
    if (!ok) return;
    for (const model of targets) {
      this.detectorsApi.deleteFromRegistry(model.id).subscribe({
        next: () => {
          this.selectedDetectorIds.delete(model.id);
          this.datasetState.refresh();
        },
        error: () => {
          // Global error interceptor surfaces the failure in the banner.
        },
      });
    }
  }

  /**
   * True when the "Combine Selected Models" button should be enabled:
   * at least two trainable models selected AND all of them share a media type.
   */
  get combineSelectedDetectorsEnabled(): boolean {
    if (this.selectedDetectorIds.size < 2) return false;
    const targets = this.detectors.filter((d) => this.selectedDetectorIds.has(d.id));
    if (targets.length < 2) return false;
    const types = new Set(targets.map((m) => m.media_type));
    return types.size === 1;
  }

  get combineSelectedDetectorsHint(): string {
    if (this.selectedDetectorIds.size < 2) {
      return 'Select two or more trainable detectors to combine';
    }
    const targets = this.detectors.filter((d) => this.selectedDetectorIds.has(d.id));
    const types = new Set(targets.map((m) => m.media_type));
    if (types.size !== 1) {
      return 'All selected detectors must be of the same media type';
    }
    return 'Combine selected detectors into a new one';
  }

  get combineSelectedDetectorSources(): DetectorRegistryEntry[] {
    return this.detectors.filter((d) => this.selectedDetectorIds.has(d.id));
  }

  get allDetectorNames(): string[] {
    return this.detectors.map((m) => m.name);
  }

  openCombineDetectorsModal(): void {
    if (!this.combineSelectedDetectorsEnabled) return;
    this.combineDetectorsModalOpen = true;
  }

  closeCombineDetectorsModal(): void {
    this.combineDetectorsModalOpen = false;
  }

  onDetectorsCombined(newName: string): void {
    this.combineDetectorsModalOpen = false;
    // The new model will appear via the datasets$/detectors$ subscription's
    // new-id auto-select logic; nothing more to do besides refreshing.
    this.datasetState.refresh();
    void newName;
  }

  onDeleteSelectedDetectorsAnimationEnd(): void {
    if (!this.deletingSelectedDetectorsConfirm) {
      this.wasDeletingSelectedDetectorsConfirm = false;
    }
  }

  // --- Dataset actions ---

  renameDataset(dataset: DatasetRegistryEntry, newName: string): void {
    this.datasetsApi.renameRegistered(dataset.id, newName).subscribe({
      next: () => this.datasetState.refresh(),
    });
  }

  async deleteDataset(dataset: DatasetRegistryEntry): Promise<void> {
    this.deletingDatasetId = dataset.id;
    const ok = await this.dialog.confirmDestructive(
      `Delete dataset "${dataset.name}"?`,
      'This removes the dataset from the registry and deletes its cached pickle file. Detectors trained against it are unaffected.',
    );
    this.deletingDatasetId = '';
    if (!ok) return;
    this.datasetsApi.deleteRegistered(dataset.id).subscribe({
      next: () => {
        this.selectedDatasetIds.delete(dataset.id);
        this.datasetState.refresh();
      },
    });
  }

  async editDatasetSecurity(dataset: DatasetRegistryEntry): Promise<void> {
    const current = (dataset.readers || []).join(', ');
    const result = await this.dialog.prompt(
      `Edit access list for "${dataset.name}".\nEnter usernames separated by commas, or * for public:`,
      current,
    );
    if (result === null) return;
    const readers = result
      .split(',')
      .map((s: string) => s.trim())
      .filter((s: string) => s.length > 0);
    this.datasetsApi.updateReaders(dataset.id, readers).subscribe({
      next: () => this.datasetState.refresh(),
    });
  }

  showDatasetStats(dataset: DatasetRegistryEntry): void {
    this.statsDatasetId = dataset.id;
    this.statsDatasetName = dataset.name;
    this.statsModalOpen = true;
  }

  closeStatsModal(): void {
    this.statsModalOpen = false;
    this.statsDatasetId = '';
    this.statsDatasetName = '';
  }

  // --- Model actions ---

  renameDetector(model: DetectorRegistryEntry, newName: string): void {
    this.detectorsApi.renameInRegistry(model.id, newName).subscribe({
      next: () => this.datasetState.refresh(),
    });
  }

  async deleteDetector(model: DetectorRegistryEntry): Promise<void> {
    this.deletingDetectorId = model.id;
    const ok = await this.dialog.confirmDestructive(
      `Delete detector "${model.name}"?`,
      'This removes its labelset and training metadata. The dataset is unaffected.',
    );
    this.deletingDetectorId = '';
    if (!ok) return;
    this.detectorsApi.deleteFromRegistry(model.id).subscribe({
      next: () => {
        this.selectedDetectorIds.delete(model.id);
        this.datasetState.refresh();
      },
      error: () => {
        // Global error interceptor surfaces the failure in the banner.
      },
    });
  }

  loadDataset(dataset: DatasetRegistryEntry): void {
    this.datasetsApi.loadRegistered(dataset.id).subscribe({
      next: () => this.startProgressPolling(),
    });
  }

  loadDetector(model: DetectorRegistryEntry): void {
    this.detectorsApi.loadDetector(model.id).subscribe({
      next: () => this.startDetectorProgressPolling(),
    });
  }

  unloadDetector(model: DetectorRegistryEntry): void {
    this.detectorsApi.unloadDetector(model.id).subscribe({
      next: () => this.datasetState.refresh(),
    });
  }

  getInlineDetectorTask(modelId: string): LoadingTask | undefined {
    return this.detectorLoadingTasks.find((t) => t.detector_id === modelId);
  }

  toggleAutorun(model: DetectorRegistryEntry, autorun: boolean): void {
    this.detectorsApi.setAutorun(model.id, autorun).subscribe({
      next: () => this.datasetState.refresh(),
    });
  }

  // --- Export modal ---

  openExportModal(model: DetectorRegistryEntry): void {
    this.exportDetectorName = model.name;
    this.exportModalOpen = true;
  }

  closeExportModal(): void {
    this.exportModalOpen = false;
    this.exportDetectorName = '';
  }

  // --- Add Labels modal ---

  openAddLabelsModal(model: DetectorRegistryEntry): void {
    this.addLabelsDetectorId = model.id;
    this.addLabelsDetectorName = model.name;
    this.addLabelsModalOpen = true;
  }

  closeAddLabelsModal(): void {
    this.addLabelsModalOpen = false;
    this.addLabelsDetectorId = '';
    this.addLabelsDetectorName = '';
  }

  onAddLabelsImported(): void {
    this.datasetState.refresh();
  }

  // --- Importer modal ---

  /** Guess the media type the user likely wants based on existing datasets, models, and in-progress loads. */
  get guessedMediaType(): string {
    const types = new Set<string>();
    for (const d of this.datasets) {
      if (d.media_type) types.add(d.media_type);
    }
    for (const m of this.detectors) {
      if (m.media_type) types.add(m.media_type);
    }
    for (const t of this.loadingTasks) {
      if (t.media_type && !t.error) types.add(t.media_type);
    }
    return types.size === 1 ? [...types][0] : '';
  }

  /** Guess the media embedder the user likely wants based on existing datasets and in-progress loads. */
  get guessedMediaEmbedder(): string {
    const embedders = new Set<string>();
    for (const d of this.datasets) {
      const emb = d['embedder'] as string;
      if (emb) embedders.add(emb);
    }
    for (const t of this.loadingTasks) {
      if (t.embedder && !t.error) embedders.add(t.embedder);
    }
    return embedders.size === 1 ? [...embedders][0] : '';
  }

  openImporterModal(): void {
    this.importerClosing = false;
    this.newThingFlows.openImporter({
      guessedMediaType: this.guessedMediaType,
      guessedMediaEmbedder: this.guessedMediaEmbedder,
    });
  }

  /** Welcome-banner CTA: open the importer modal with one of the picker
   *  tabs pre-selected so a fresh user lands on the relevant choices. */
  openImporterModalOnTab(tabId: string): void {
    this.importerClosing = false;
    this.newThingFlows.openImporter({
      initialTab: tabId,
      guessedMediaType: this.guessedMediaType,
      guessedMediaEmbedder: this.guessedMediaEmbedder,
    });
  }

  onImporterAnimationEnd(): void {
    this.importerClosing = false;
  }

  // --- First-run welcome banner ---

  /** Visible importers in the Services tab, in registry order. */
  get servicesImporters(): ImporterInfo[] {
    return this.visibleImporters.filter((imp) => (imp.category || '') === 'services');
  }

  /** Pretty name for a Services importer (display_name falls back to name). */
  private importerLabel(imp: ImporterInfo): string {
    return imp.display_name || imp.name;
  }

  /** Body text for the welcome banner. Three branches:
   *   - no Services importers → push Server Folder
   *   - exactly one Services importer → name it
   *   - multiple Services importers → generic Services prompt */
  get welcomeBannerMessage(): string {
    const services = this.servicesImporters;
    if (services.length === 1) {
      const label = this.importerLabel(services[0]);
      return `Welcome — load a dataset to get started. You have ${label} registered in the Services tab, which is probably what you want.`;
    }
    if (services.length > 1) {
      return 'Welcome — load a dataset to get started. You have importers registered in the Services tab, which is probably where to start.';
    }
    return 'Welcome — load a dataset to get started. Server Folder is the most common starting point: point it at a folder of files already on this machine.';
  }

  /** CTA label paired with the welcome-banner message. */
  get welcomeBannerCtaLabel(): string {
    return this.servicesImporters.length > 0 ? 'Open Services' : 'Open Server';
  }

  /** Picker tab the welcome-banner CTA pre-selects in the importer modal. */
  get welcomeBannerCtaTab(): string {
    return this.servicesImporters.length > 0 ? 'services' : 'server';
  }

  /** Open-state of the dataset importer modal (hosted on AppComponent;
   *  this getter just proxies the singleton state). Drives the `+` icon
   *  animation. */
  get importerModalOpen(): boolean {
    return this.newThingFlows.importer.open;
  }

  /** Open-state of the new-detector modal (hosted on AppComponent). */
  get newDetectorModalOpen(): boolean {
    return this.newThingFlows.newDetector.open;
  }

  private handleImportStarted(): void {
    this.importerClosing = true;
    this.startProgressPolling();
  }

  private handleDemoSelected(demo: DemoDataset): void {
    this.importerClosing = true;
    // The importer modal augments the DemoDataset payload with the
    // user-chosen embedder / clipper / display name before emitting;
    // those extras aren't part of the typed schema so we read them
    // through a permissive cast.
    const extras = demo as DemoDataset & {
      embedder?: string;
      clipper?: string;
      dataset_name?: string;
    };
    const params: Record<string, string> = {};
    if (extras.embedder) params['embedder'] = extras.embedder;
    if (extras.clipper) params['clipper'] = extras.clipper;
    const userName = (extras.dataset_name || '').trim();
    if (userName) params['dataset_name'] = userName;
    this.datasetsApi.loadDemo(demo.name, params).subscribe({
      next: () => {
        this.startProgressPolling();
      },
    });
  }

  // --- New model modal ---

  /** Media type used as default for new models: single-selected dataset wins, then first loaded dataset, then in-progress loads. */
  get activeDatasetMediaType(): string {
    if (this.selectedDatasetIds.size === 1) {
      const selId = [...this.selectedDatasetIds][0];
      const sel = this.datasets.find((d) => d.id === selId);
      if (sel?.media_type) return sel.media_type;
    }
    const loaded = this.datasets.find((d) => d.loaded);
    if (loaded?.media_type) return loaded.media_type;
    // Fall back to in-progress loading tasks when no dataset is fully loaded yet.
    const loadingTypes = new Set(
      this.loadingTasks.filter((t) => t.media_type && !t.error).map((t) => t.media_type!),
    );
    return loadingTypes.size === 1 ? [...loadingTypes][0] : '';
  }

  openNewDetectorModal(): void {
    this.newDetectorClosing = false;
    this.newThingFlows.openNewDetector({ defaultMediaType: this.activeDatasetMediaType });
  }

  onNewDetectorAnimationEnd(): void {
    this.newDetectorClosing = false;
  }

  // --- Cancel ---

  onCancelIngest(): void {
    this.datasetsApi.cancelIngest().subscribe();
  }

  cancelLoadingTask(taskId: string): void {
    this.datasetsApi.cancelTask(taskId).subscribe();
  }

  dismissLoadingTask(taskId: string): void {
    this.loadingTasks = this.loadingTasks.filter((t) => t.task_id !== taskId);
  }

  cancelDetectorLoadingTask(taskId: string): void {
    this.detectorsApi.cancelDetectorLoadingTask(taskId).subscribe();
  }

  dismissDetectorLoadingTask(taskId: string): void {
    this.detectorLoadingTasks = this.detectorLoadingTasks.filter((t) => t.task_id !== taskId);
  }

  // --- Progress polling ---

  startProgressPolling(onComplete?: () => void): void {
    // If polling is already active, don't restart — the existing loop
    // already covers all tasks.  This avoids clearing completedTaskIds
    // and losing track of tasks that just finished.
    if (this.datasetPollingActive) {
      return;
    }
    this.datasetPollingActive = true;
    this.completedTaskIds.clear();

    this.progressEvents.loadingTasks$
      .pipe(takeUntil(this.polling$), takeUntil(this.destroy$))
      .subscribe({
        next: (tasks: LoadingTask[]) => {
          // Separate active from finished. Failed tasks are surfaced
          // globally by SseErrorRouterService → ToastService; we just
          // keep them in the inline list so the row still shows the
          // dashed loading bar with the error text.
          const active = tasks.filter((t) => t.status !== 'idle');
          const errored = tasks.filter((t) => t.status === 'idle' && !!t.error);
          const failed = errored.filter((t) => t.error !== 'Cancelled');

          this.loadingTasks = [...active, ...failed];
          this.datasetState.setLoadingTasks(active);

          // Detect tasks that just completed successfully so we can
          // refresh the registry immediately (not only when ALL finish).
          const justFinished = tasks.filter(
            (t) => t.status === 'idle' && !t.error && !this.completedTaskIds.has(t.task_id),
          );
          for (const t of justFinished) {
            this.completedTaskIds.add(t.task_id);
          }
          if (justFinished.length > 0) {
            this.datasetState.refresh();
            this.achievements.refresh();
          }

          this.datasetState.setLoading(active.length > 0);

          if (active.length === 0) {
            // No more active tasks — stop polling
            this.polling$.next();
            this.datasetPollingActive = false;
            // Refresh unless we just did (justFinished already triggered it)
            if (justFinished.length === 0) {
              this.datasetState.refresh();
            }
            if (onComplete && failed.length === 0) {
              onComplete();
            }
          }
        },
      });
  }

  startDetectorProgressPolling(onComplete?: () => void): void {
    if (this.detectorPollingActive) {
      return;
    }
    this.detectorPollingActive = true;
    this.completedModelTaskIds.clear();

    this.progressEvents.detectorLoadingTasks$
      .pipe(takeUntil(this.detectorPolling$), takeUntil(this.destroy$))
      .subscribe({
        next: (tasks: LoadingTask[]) => {
          const active = tasks.filter((t) => t.status !== 'idle');
          const errored = tasks.filter((t) => t.status === 'idle' && !!t.error);
          const failed = errored.filter((t) => t.error !== 'Cancelled');

          this.detectorLoadingTasks = [...active, ...failed];

          // Detect tasks that just completed successfully
          const justFinished = tasks.filter(
            (t) => t.status === 'idle' && !t.error && !this.completedModelTaskIds.has(t.task_id),
          );
          for (const t of justFinished) {
            this.completedModelTaskIds.add(t.task_id);
          }
          if (justFinished.length > 0) {
            this.datasetState.refresh();
            this.achievements.refresh();
          }

          if (active.length === 0) {
            this.detectorPolling$.next();
            this.detectorPollingActive = false;
            if (justFinished.length === 0) {
              this.datasetState.refresh();
            }
            if (onComplete && failed.length === 0) {
              onComplete();
            }
          }
        },
      });
  }

  // --- Sorting ---

  get sortedDatasets(): DatasetRegistryEntry[] {
    const col = this.datasetCols.sortColumn;
    const asc = this.datasetCols.sortAsc ? 1 : -1;
    return [...this.datasets].sort((a, b) => {
      const va = a[col] ?? '';
      const vb = b[col] ?? '';
      if (typeof va === 'number' && typeof vb === 'number') return (va - vb) * asc;
      return String(va).localeCompare(String(vb)) * asc;
    });
  }

  get sortedDetectors(): DetectorRegistryEntry[] {
    const col = this.detectorCols.sortColumn;
    const asc = this.detectorCols.sortAsc ? 1 : -1;
    return [...this.detectors].sort((a, b) => {
      const va = a[col] ?? '';
      const vb = b[col] ?? '';
      if (typeof va === 'number' && typeof vb === 'number') return (va - vb) * asc;
      return String(va).localeCompare(String(vb)) * asc;
    });
  }

  // --- Button state ---

  get isLoading(): boolean {
    // Phase 2: the `activeContextGuard` owns the dataset/detector load
    // path the Train/Find buttons used to drive locally. We mirror its
    // signals (plus the per-button click-intent flags) here so
    // dashboard controls stay disabled while the guard waits between
    // click and route activation.
    return (
      this.trainLoading ||
      this.findLoading ||
      this.contextSwitch.switching ||
      this.datasetState.loading
    );
  }

  get labelEnabled(): boolean {
    const selectedDatasets = this.resolvedSelectedDatasets;
    const selectedModels = this.resolvedSelectedModels;
    if (selectedDatasets.length !== 1) return false;
    if (selectedModels.length === 0) return true; // will prompt to create a model
    if (selectedModels.length !== 1) return false;
    const model = selectedModels[0];
    if (model.media_type !== selectedDatasets[0].media_type) return false;
    return true;
  }

  private findMediaTypesMatch(): boolean {
    const types = new Set([
      ...this.resolvedSelectedDatasets.map((d) => d.media_type),
      ...this.resolvedSelectedModels.map((m) => m.media_type),
    ]);
    return types.size === 1;
  }

  private hasUntrainedModel(): boolean {
    return this.resolvedSelectedModels.some((m) => (m.num_training ?? 0) === 0);
  }

  private get resolvedSelectedDatasets(): DatasetRegistryEntry[] {
    return this.datasets.filter((d) => this.selectedDatasetIds.has(d.id));
  }

  private get resolvedSelectedModels(): DetectorRegistryEntry[] {
    return this.detectors.filter((d) => this.selectedDetectorIds.has(d.id));
  }

  get findEnabled(): boolean {
    if (this.resolvedSelectedDatasets.length < 1 || this.resolvedSelectedModels.length < 1) return false;
    if (!this.findMediaTypesMatch()) return false;
    if (this.hasUntrainedModel()) return false;
    return true;
  }

  get findHint(): string {
    const nDatasets = this.resolvedSelectedDatasets.length;
    const nModels = this.resolvedSelectedModels.length;
    if (nDatasets === 0 && nModels === 0) return 'Select a dataset and a detector';
    if (nDatasets === 0) return 'Select a dataset';
    if (nModels === 0) return 'Select a detector';
    if (!this.findMediaTypesMatch()) return 'Media type mismatch';
    if (this.hasUntrainedModel()) return 'Selected detector has no training labels';
    return 'Score selected datasets with selected detectors';
  }

  get labelHint(): string {
    const nDatasets = this.resolvedSelectedDatasets.length;
    const nModels = this.resolvedSelectedModels.length;
    if (nDatasets === 0) return 'Select a dataset';
    if (nDatasets > 1) return 'Select exactly 1 dataset';
    if (nModels === 0) return 'Create a new detector and start training';
    if (nModels > 1) return 'Select exactly 1 detector';
    const model = this.resolvedSelectedModels[0];
    const dataset = this.resolvedSelectedDatasets[0];
    if (model && dataset && model.media_type !== dataset.media_type) {
      return 'Media type mismatch';
    }
    return 'Open Train Mode with the selected dataset and detector';
  }

  private storeSelectedModelTextQuery(): void {
    const model = this.detectors.find((m) => this.selectedDetectorIds.has(m.id));
    this.labelSession.textQuery = model?.text_query || '';
    this.labelSession.mediaExample = model?.media_example || '';
    this.labelSession.examples = (model?.['examples'] as { type: string; value: string }[]) || [];
    this.labelSession.modelName = model?.name || '';
  }

  onLabel(): void {
    const dataset = this.datasets.find((d) => this.selectedDatasetIds.has(d.id));
    if (!dataset) return;

    const modelId = [...this.selectedDetectorIds][0] || null;
    if (!modelId) {
      // No model selected — open the New Model modal; on creation we'll proceed to training
      this.trainAfterModelCreation = true;
      this.openNewDetectorModal();
      return;
    }

    this.storeSelectedModelTextQuery();
    // The `activeContextGuard` on /label/:datasetId/:detectorId flips
    // ActiveContextService and runs any needed loads via
    // ContextSwitchService before activating the route.
    this.trainLoading = true;
    this.router.navigate(['/label', dataset.id, modelId]);
  }

  onFind(): void {
    const dataset = this.datasets.find((d) => this.selectedDatasetIds.has(d.id));
    if (!dataset) return;

    const model = this.detectors.find((m) => this.selectedDetectorIds.has(m.id));
    if (!model) return;

    // See `onLabel` — the route guard owns context + loading.
    this.findLoading = true;
    this.router.navigate(['/find', dataset.id, model.id]);
  }

  /** Old Find window — runs multi-dataset multi-detector find and shows results modal. */
  onOldFind(): void {
    const datasetIds = [...this.selectedDatasetIds];
    const detectorIds = [...this.selectedDetectorIds];
    const findParams = { dataset_ids: datasetIds, detector_ids: detectorIds };

    this.datasetState.setLoading(true);
    this.datasetState.setProgressMessage('Checking labels...');
    this.progressIndeterminate = true;

    // Pre-flight: check if any labels fail to resolve
    this.detectorsApi.findCheckLabels(findParams).subscribe({
      next: async (checkResult) => {
        const warnings = checkResult.warnings || [];
        if (warnings.length > 0) {
          // Build warning message
          const lines = warnings.map(
            (w) => `${w.failed_labels} of ${w.total_labels} labels failed to resolve for "${w.detector_name}".`,
          );
          const message = lines.join('\n') + '\n\nDo you want to continue?';
          const ok = await this.dialog.confirm(message, 'warning');
          if (!ok) {
            this.datasetState.setLoading(false);
            this.progressIndeterminate = false;
            return;
          }
        }
        this.runFind(findParams);
      },
      error: () => {
        // If check-labels fails, proceed with Find anyway
        this.runFind(findParams);
      },
    });
  }

  private startFindProgressPolling(): void {
    this.findPolling$.next(); // cancel previous
    this.progressEvents.find$
      .pipe(takeUntil(this.findPolling$), takeUntil(this.destroy$))
      .subscribe({
        next: (progress: any) => {
          if (!progress || progress.status === 'idle') return;

          if (progress.current != null && progress.total != null && progress.total > 0) {
            this.progressIndeterminate = false;
            this.progressValue = progress.current;
            this.progressTotal = progress.total;
          } else {
            this.progressIndeterminate = true;
          }

          let msg = progress.message || 'Running Find...';
          if (progress.step != null && progress.total_steps != null && progress.total_steps > 1) {
            msg = `[Step ${progress.step}/${progress.total_steps}] ${msg}`;
          }
          if (progress.current != null && progress.total != null && progress.total > 0) {
            const fraction = `(${formatProgressFraction(progress.current, progress.total)})`;
            const stepEnd = msg.indexOf('] ');
            if (stepEnd !== -1) {
              msg = msg.slice(0, stepEnd + 2) + fraction + ' ' + msg.slice(stepEnd + 2);
            } else {
              msg = fraction + ' ' + msg;
            }
          }
          this.datasetState.setProgressMessage(msg);
        },
      });
  }

  private stopFindProgressPolling(): void {
    this.findPolling$.next();
  }

  private runFind(findParams: Record<string, unknown>): void {
    this.datasetState.setProgressMessage('Running Find...');
    this.startFindProgressPolling();

    this.detectorsApi.find(findParams).subscribe({
      next: (response: any) => {
        this.stopFindProgressPolling();
        this.datasetState.setLoading(false);
        this.progressIndeterminate = false;

        // Convert /api/find response to AutoDetectResultsData format
        const mapHit = (r: any) => ({
          md5: r.md5 || '',
          filename: r.filename || '',
          origin_name: r.origin_name || '',
          origin: r.origin,
          dataset_name: r.dataset_name || '',
          detector_verdicts: r.detector_verdicts || {},
        });

        const hits = (response.results || []).map(mapHit);
        const negativeHits = (response.negative_results || []).map(mapHit);

        const detectorNames: string[] = response.detectors || [];
        const detectorResults: Record<string, any> = {};

        if (detectorNames.length <= 1) {
          // Single detector: one result group
          const label = detectorNames[0] || 'Find';
          detectorResults[label] = {
            detector_name: label,
            total_hits: hits.length,
            hits,
            negative_hits: negativeHits,
          };
        } else {
          // Multiple detectors: group hits by detector
          for (const name of detectorNames) {
            const detectorHits = hits.filter(
              (h: any) => h.detector_verdicts?.[name]?.verdict === 'Good',
            );
            const detectorNegHits = negativeHits.filter(
              (h: any) => h.detector_verdicts?.[name]?.verdict !== 'Good',
            );
            detectorResults[name] = {
              detector_name: name,
              total_hits: detectorHits.length,
              hits: detectorHits,
              negative_hits: detectorNegHits,
            };
          }
        }

        this.findResultsData = {
          media_type: response.media_type || 'unknown',
          detectors_run: detectorNames.length,
          results: detectorResults,
          detectors: detectorNames,
          datasets: response.datasets || [],
          multiple_datasets: response.multiple_datasets || false,
          multiple_detectors: response.multiple_detectors || false,
        } as AutoDetectResultsData;

        this.findResultsOpen = true;
      },
      error: () => {
        this.stopFindProgressPolling();
        this.datasetState.setLoading(false);
        this.progressIndeterminate = false;
        // Global error interceptor surfaces the failure in the banner.
      },
    });
  }

  closeFindResults(): void {
    this.findResultsOpen = false;
  }

  // --- Loading task helpers ---

  taskProgressMessage(task: LoadingTask): string {
    let msg = task.message || 'Loading...';
    if (task.step != null && task.total_steps != null && task.total_steps > 1) {
      msg = `[Step ${task.step}/${task.total_steps}] ${msg}`;
    }
    if (task.current != null && task.total != null && task.total > 0) {
      const fraction = `(${formatProgressFraction(task.current, task.total)})`;
      const stepEnd = msg.indexOf('] ');
      if (stepEnd !== -1) {
        msg = msg.slice(0, stepEnd + 2) + fraction + ' ' + msg.slice(stepEnd + 2);
      } else {
        msg = fraction + ' ' + msg;
      }
    }
    return msg;
  }

  taskIsIndeterminate(task: LoadingTask): boolean {
    return !(task.current != null && task.total != null && task.total > 0);
  }
}
