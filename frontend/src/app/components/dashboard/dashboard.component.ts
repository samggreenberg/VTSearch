import { Component, HostListener, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { EMPTY, Subject, timer } from 'rxjs';
import { catchError, filter, take, takeUntil, switchMap } from 'rxjs/operators';
import { DatasetsApiService } from '../../services/datasets-api.service';
import { DetectorsApiService } from '../../services/detectors-api.service';
import { TrainableModelsApiService } from '../../services/trainable-models-api.service';
import { VtDialogService } from '../../services/dialog.service';
import { LabelSessionService } from '../../services/label-session.service';
import { FindSessionService } from '../../services/find-session.service';
import { DatasetStateService } from '../../services/dataset-state.service';
import { ActiveContextService } from '../../services/active-context.service';
import { AuthService } from '../../services/auth.service';
import { TopBarStateService } from '../../services/top-bar-state.service';
import { AutoDetectResultsData, DatasetRegistryEntry, LoadingTask, LoadingTasksResponse, ModelRegistryEntry } from '../../models/api.models';
import { formatProgressFraction } from '../../utils/format-progress';
import { ProgressBarComponent } from '../progress-bar/progress-bar.component';
import { AutoDetectResultsModalComponent } from '../modals/autodetect-results-modal/autodetect-results-modal.component';
import { DatasetCardComponent } from './dataset-card/dataset-card.component';
import { ModelCardComponent } from './model-card/model-card.component';
import { DatasetImporterModalComponent } from './dataset-importer-modal/dataset-importer-modal.component';
import { NewModelModalComponent } from './new-model-modal/new-model-modal.component';
import { LabelExporterModalComponent } from '../modals/label-exporter-modal/label-exporter-modal.component';
import { LabelImporterModalComponent } from '../modals/label-importer-modal/label-importer-modal.component';
import { DatasetStatsModalComponent } from '../modals/dataset-stats-modal/dataset-stats-modal.component';
import { IconComponent } from '../icon/icon.component';

@Component({
  selector: 'vt-dashboard',
  standalone: true,
  imports: [
    CommonModule,
    ProgressBarComponent,
    AutoDetectResultsModalComponent,
    DatasetCardComponent,
    ModelCardComponent,
    DatasetImporterModalComponent,
    NewModelModalComponent,
    LabelExporterModalComponent,
    LabelImporterModalComponent,
    DatasetStatsModalComponent,
    IconComponent,
  ],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.scss',
})
export class DashboardComponent implements OnInit, OnDestroy {
  selectedDatasetIds: Set<string> = new Set();
  selectedModelIds: Set<string> = new Set();

  progressValue = 0;
  progressTotal = 0;
  progressIndeterminate = false;

  loadingTasks: LoadingTask[] = [];
  modelLoadingTasks: LoadingTask[] = [];

  importerModalOpen = false;
  importerClosing = false;
  newModelModalOpen = false;
  newModelClosing = false;
  exportModalOpen = false;
  exportModelName = '';
  addLabelsModalOpen = false;
  addLabelsModelName = '';
  findResultsOpen = false;
  findResultsData: AutoDetectResultsData = { results: {} };
  statsModalOpen = false;
  statsDatasetId = '';
  statsDatasetName = '';
  deletingDatasetId = '';
  deletingModelId = '';
  addLabelsModelId = '';
  trainLoading = false;
  findLoading = false;
  trainAfterModelCreation = false;

  datasetSortColumn = 'name';
  datasetSortAsc = true;
  modelSortColumn = 'name';
  modelSortAsc = true;

  // Column resize state
  datasetColWidths: Record<string, number> = {};
  modelColWidths: Record<string, number> = {};
  datasetsTableFixed = false;
  modelsTableFixed = false;
  datasetsTableWidth = 0;
  modelsTableWidth = 0;
  private datasetsResizeInit = false;
  private modelsResizeInit = false;
  private resizeState: {
    startX: number;
    startWidth: number;
    table: 'datasets' | 'models';
    col: string;
    dragged: boolean;
    tableEl: HTMLTableElement;
  } | null = null;

  // Column order. The "name" column is fixed in position 0 (so the loading-task
  // colspan that replaces "all columns after name" keeps working); these arrays
  // hold the columns that follow name and are user-reorderable.
  static readonly DATASET_COLUMNS_DEFAULT = [
    'media_type', 'num_items', 'created_at', 'created_by', 'readers', 'loaded', 'actions',
  ];
  static readonly MODEL_COLUMNS_DEFAULT = [
    'media_type', 'num_training', 'trainable', 'autodetect', 'last_trained_at',
    'created_at', 'detector_loaded', 'actions',
  ];
  private static readonly DATASET_COL_ORDER_KEY = 'vtsearch.dashboard.datasetColumnOrder';
  private static readonly MODEL_COL_ORDER_KEY = 'vtsearch.dashboard.modelColumnOrder';
  datasetColumnOrder: string[] = [...DashboardComponent.DATASET_COLUMNS_DEFAULT];
  modelColumnOrder: string[] = [...DashboardComponent.MODEL_COLUMNS_DEFAULT];

  // Drag-reorder state
  dragCol: { table: 'datasets' | 'models'; col: string } | null = null;
  dropTargetCol: { table: 'datasets' | 'models'; col: string } | null = null;

  // Per-column display metadata. Keyed by `data-col` value; used both by the
  // header template and by card components when rendering body cells in order.
  static readonly DATASET_COL_META: Record<string, { label: string; title: string; sortable: boolean }> = {
    name: { label: 'Name', title: 'Dataset display name (click to sort)', sortable: true },
    media_type: { label: 'Type', title: 'Media type: audio, image, text, video, or document (click to sort)', sortable: true },
    num_items: { label: '# Items', title: 'Number of media items in the dataset (click to sort)', sortable: true },
    created_at: { label: 'Created', title: 'When the dataset was first imported (click to sort)', sortable: true },
    created_by: { label: 'Creator', title: 'User who created this dataset (click to sort)', sortable: true },
    readers: { label: 'Readers', title: 'Users with access to this dataset (click to sort)', sortable: true },
    loaded: { label: 'Loaded?', title: 'Whether the dataset is currently loaded in memory', sortable: false },
    actions: { label: 'Actions', title: 'Available operations for this dataset', sortable: false },
  };
  static readonly MODEL_COL_META: Record<string, { label: string; title: string; sortable: boolean }> = {
    name: { label: 'Name', title: 'Model display name (click to sort)', sortable: true },
    media_type: { label: 'Type', title: 'Media type this model operates on (click to sort)', sortable: true },
    num_training: { label: '# Training', title: 'Number of labeled training examples (click to sort)', sortable: true },
    trainable: { label: 'Trainable?', title: 'Is this Model one we can load into Train Mode and improve?', sortable: false },
    autodetect: { label: 'Autorun?', title: 'Include this model in CLI autorun (click to sort)', sortable: true },
    last_trained_at: { label: 'Last Trained', title: 'When the model was last trained (click to sort)', sortable: true },
    created_at: { label: 'Created', title: 'When the model was created (click to sort)', sortable: true },
    detector_loaded: { label: 'Loaded?', title: "Whether the model's inference data is cached in memory", sortable: false },
    actions: { label: 'Actions', title: 'Available operations for this model', sortable: false },
  };

  get visibleDatasetColumns(): string[] {
    if (this.isDefaultLogin) {
      return this.datasetColumnOrder.filter((c) => c !== 'created_by' && c !== 'readers');
    }
    return this.datasetColumnOrder;
  }

  get visibleModelColumns(): string[] {
    return this.modelColumnOrder;
  }

  datasetColMeta(col: string): { label: string; title: string; sortable: boolean } {
    return DashboardComponent.DATASET_COL_META[col] ?? { label: col, title: '', sortable: false };
  }

  modelColMeta(col: string): { label: string; title: string; sortable: boolean } {
    return DashboardComponent.MODEL_COL_META[col] ?? { label: col, title: '', sortable: false };
  }

  onDatasetHeaderClick(col: string): void {
    if (this.datasetColMeta(col).sortable) this.sortDatasets(col);
  }

  onModelHeaderClick(col: string): void {
    if (this.modelColMeta(col).sortable) this.sortModels(col);
  }

  private destroy$ = new Subject<void>();
  private polling$ = new Subject<void>();
  private modelPolling$ = new Subject<void>();
  private findPolling$ = new Subject<void>();
  private knownDatasetIds = new Set<string>();
  private knownModelIds = new Set<string>();
  private completedTaskIds = new Set<string>();
  private completedModelTaskIds = new Set<string>();
  private datasetPollingActive = false;
  private modelPollingActive = false;

  currentUser = '';
  isDefaultLogin = true;

  constructor(
    private router: Router,
    private datasetsApi: DatasetsApiService,
    private detectorsApi: DetectorsApiService,
    private modelsApi: TrainableModelsApiService,
    private dialog: VtDialogService,
    private labelSession: LabelSessionService,
    private findSession: FindSessionService,
    public datasetState: DatasetStateService,
    private activeContext: ActiveContextService,
    private authService: AuthService,
    private topBarState: TopBarStateService,
  ) {}

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
    this.datasetState.models$
      .pipe(takeUntil(this.destroy$))
      .subscribe((models) => {
        const currentIds = new Set(models.map((m) => m.id));
        // Prune selections that no longer exist in the registry
        for (const id of this.selectedModelIds) {
          if (!currentIds.has(id)) this.selectedModelIds.delete(id);
        }
        const newIds = [...currentIds].filter((id) => !this.knownModelIds.has(id));
        if (newIds.length > 0 && this.knownModelIds.size > 0) {
          // Items were added after initial load — select only the new ones
          this.selectedModelIds.clear();
          for (const id of newIds) {
            this.selectedModelIds.add(id);
          }
        } else if (models.length === 1 && this.selectedModelIds.size === 0) {
          // First load with exactly one item — auto-select it
          this.selectedModelIds.add(models[0].id);
        }
        this.knownModelIds = currentIds;
        this.pushTopBarLabels();
      });
    this.loadColumnOrders();
    this.refresh();
    this.resumeActivePolling();
  }

  /** Check for in-progress loading tasks (e.g. after a page reload) and resume polling. */
  private resumeActivePolling(): void {
    this.datasetsApi.getLoadingTasks().subscribe((tasks) => {
      if (tasks.some((t) => t.status !== 'idle')) {
        this.startProgressPolling();
      }
    });
    this.modelsApi.getModelLoadingTasks().subscribe((resp) => {
      if ((resp.tasks ?? []).some((t: LoadingTask) => t.status !== 'idle')) {
        this.startModelProgressPolling();
      }
    });
  }

  // --- Column resize ---

  startResize(event: MouseEvent, table: 'datasets' | 'models'): void {
    event.stopPropagation();
    event.preventDefault();

    // The handle sits on the LEFT edge of its host <th> and resizes the
    // column to its left (see dashboard.component.scss for why).
    const th = (event.target as HTMLElement).closest('th') as HTMLElement;
    const prevTh = th.previousElementSibling as HTMLElement | null;
    const col = prevTh?.getAttribute('data-col');
    if (!col) return;
    const tableEl = th.closest('table') as HTMLTableElement;
    const colWidths = table === 'datasets' ? this.datasetColWidths : this.modelColWidths;
    const initialized = table === 'datasets' ? this.datasetsResizeInit : this.modelsResizeInit;

    if (!initialized) {
      const ths = tableEl.querySelectorAll('thead tr th') as NodeListOf<HTMLElement>;
      let totalWidth = 0;
      ths.forEach((t) => {
        const colKey = t.getAttribute('data-col');
        const w = t.offsetWidth;
        if (colKey) colWidths[colKey] = w;
        totalWidth += w;
      });
      if (table === 'datasets') {
        this.datasetsTableWidth = totalWidth;
        this.datasetsResizeInit = true;
        this.datasetsTableFixed = true;
      } else {
        this.modelsTableWidth = totalWidth;
        this.modelsResizeInit = true;
        this.modelsTableFixed = true;
      }
    }

    this.resizeState = {
      startX: event.clientX,
      startWidth: colWidths[col] ?? 100,
      table,
      col,
      dragged: false,
      tableEl,
    };
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }

  @HostListener('document:mousemove', ['$event'])
  onColResizeMove(event: MouseEvent): void {
    if (!this.resizeState) return;
    const delta = event.clientX - this.resizeState.startX;
    if (Math.abs(delta) > 3) this.resizeState.dragged = true;
    if (!this.resizeState.dragged) return;
    const colWidths = this.resizeState.table === 'datasets' ? this.datasetColWidths : this.modelColWidths;
    const newWidth = Math.max(30, this.resizeState.startWidth + delta);
    const prevWidth = colWidths[this.resizeState.col] ?? this.resizeState.startWidth;
    colWidths[this.resizeState.col] = newWidth;
    const widthChange = newWidth - prevWidth;
    if (this.resizeState.table === 'datasets') {
      this.datasetsTableWidth = Math.max(100, this.datasetsTableWidth + widthChange);
    } else {
      this.modelsTableWidth = Math.max(100, this.modelsTableWidth + widthChange);
    }
  }

  @HostListener('document:mouseup')
  onColResizeEnd(): void {
    if (!this.resizeState) return;
    const state = this.resizeState;
    this.resizeState = null;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';

    if (!state.dragged) {
      this.autoSizeColumn(state.tableEl, state.table, state.col);
    }
  }

  private autoSizeColumn(tableEl: HTMLTableElement, table: 'datasets' | 'models', col: string): void {
    const colWidths = table === 'datasets' ? this.datasetColWidths : this.modelColWidths;

    // Find the column index from the header
    const ths = tableEl.querySelectorAll('thead tr th') as NodeListOf<HTMLElement>;
    let colIndex = -1;
    for (let i = 0; i < ths.length; i++) {
      if (ths[i].getAttribute('data-col') === col) {
        colIndex = i;
        break;
      }
    }
    if (colIndex < 0) return;

    // Temporarily switch to auto layout so content determines width
    const wasFixed = tableEl.classList.contains('table-fixed');
    const prevTableWidth = tableEl.style.width;
    const prevColWidth = ths[colIndex].style.width;

    tableEl.classList.remove('table-fixed');
    // Force width:auto so the table shrinks to content instead of stretching
    // to 100% (the CSS class default).  Without this, the browser distributes
    // extra container space across columns, inflating every measurement.
    tableEl.style.width = 'auto';
    ths[colIndex].style.width = '';

    // Measure the natural width of the column from body cells.
    let maxWidth = 0;
    const rows = tableEl.querySelectorAll('tbody tr');
    rows.forEach((row) => {
      const cell = row.children[colIndex] as HTMLElement | undefined;
      if (cell) {
        // For cells with colspan, skip (loading task rows)
        if (cell.hasAttribute('colspan')) return;
        maxWidth = Math.max(maxWidth, cell.scrollWidth);
      }
    });
    // Fall back to header width only when the table has no body rows
    if (maxWidth === 0) {
      maxWidth = ths[colIndex].offsetWidth;
    }

    // Add a small buffer for padding
    maxWidth = Math.max(30, maxWidth + 2);

    // Apply the measured width and restore fixed layout
    const prevWidth = colWidths[col] ?? 0;
    colWidths[col] = maxWidth;

    if (table === 'datasets') {
      this.datasetsTableFixed = true;
      this.datasetsResizeInit = true;
      // Recalculate total table width
      let total = 0;
      ths.forEach((t) => {
        const key = t.getAttribute('data-col');
        if (key && key !== col) {
          if (!colWidths[key]) colWidths[key] = t.offsetWidth;
          total += colWidths[key];
        } else if (key === col) {
          total += maxWidth;
        }
      });
      this.datasetsTableWidth = total;
    } else {
      this.modelsTableFixed = true;
      this.modelsResizeInit = true;
      let total = 0;
      ths.forEach((t) => {
        const key = t.getAttribute('data-col');
        if (key && key !== col) {
          if (!colWidths[key]) colWidths[key] = t.offsetWidth;
          total += colWidths[key];
        } else if (key === col) {
          total += maxWidth;
        }
      });
      this.modelsTableWidth = total;
    }

    // Restore layout (Angular binding will apply on next change detection).
    // Clear the temporary 'auto' override so Angular bindings take over.
    tableEl.style.width = prevTableWidth;
    if (wasFixed) {
      tableEl.classList.add('table-fixed');
    }
  }

  // --- Column reorder (drag and drop) ---

  private loadColumnOrders(): void {
    this.datasetColumnOrder = this.normalizeColumnOrder(
      DashboardComponent.DATASET_COL_ORDER_KEY,
      DashboardComponent.DATASET_COLUMNS_DEFAULT,
    );
    this.modelColumnOrder = this.normalizeColumnOrder(
      DashboardComponent.MODEL_COL_ORDER_KEY,
      DashboardComponent.MODEL_COLUMNS_DEFAULT,
    );
  }

  private normalizeColumnOrder(storageKey: string, defaults: readonly string[]): string[] {
    let stored: unknown = null;
    try {
      const raw = localStorage.getItem(storageKey);
      stored = raw ? JSON.parse(raw) : null;
    } catch {
      stored = null;
    }
    const known = new Set(defaults);
    const result: string[] = [];
    if (Array.isArray(stored)) {
      for (const c of stored) {
        if (typeof c === 'string' && known.has(c) && !result.includes(c)) {
          result.push(c);
        }
      }
    }
    // Append any defaults missing from the stored order (e.g. new columns
    // added after the user's order was saved).
    for (const c of defaults) {
      if (!result.includes(c)) result.push(c);
    }
    return result;
  }

  private persistColumnOrder(table: 'datasets' | 'models'): void {
    const key = table === 'datasets'
      ? DashboardComponent.DATASET_COL_ORDER_KEY
      : DashboardComponent.MODEL_COL_ORDER_KEY;
    const order = table === 'datasets' ? this.datasetColumnOrder : this.modelColumnOrder;
    try {
      localStorage.setItem(key, JSON.stringify(order));
    } catch {
      // Ignore quota / private-mode failures.
    }
  }

  onColDragStart(event: DragEvent, table: 'datasets' | 'models', col: string): void {
    // Don't start a drag if a column resize is in progress.
    if (this.resizeState) {
      event.preventDefault();
      return;
    }
    this.dragCol = { table, col };
    this.dropTargetCol = null;
    if (event.dataTransfer) {
      event.dataTransfer.effectAllowed = 'move';
      // Some browsers refuse to start the drag without setData.
      event.dataTransfer.setData('text/plain', col);
    }
  }

  onColDragOver(event: DragEvent, table: 'datasets' | 'models', col: string): void {
    if (!this.dragCol || this.dragCol.table !== table || this.dragCol.col === col) return;
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = 'move';
    if (
      !this.dropTargetCol
      || this.dropTargetCol.col !== col
      || this.dropTargetCol.table !== table
    ) {
      this.dropTargetCol = { table, col };
    }
  }

  onColDragLeave(_event: DragEvent, table: 'datasets' | 'models', col: string): void {
    if (this.dropTargetCol && this.dropTargetCol.table === table && this.dropTargetCol.col === col) {
      this.dropTargetCol = null;
    }
  }

  onColDrop(event: DragEvent, table: 'datasets' | 'models', targetCol: string): void {
    if (!this.dragCol || this.dragCol.table !== table) {
      this.dragCol = null;
      this.dropTargetCol = null;
      return;
    }
    event.preventDefault();
    const sourceCol = this.dragCol.col;
    this.dragCol = null;
    this.dropTargetCol = null;
    if (sourceCol === targetCol) return;

    const order = table === 'datasets' ? this.datasetColumnOrder : this.modelColumnOrder;
    const fromIdx = order.indexOf(sourceCol);
    const toIdx = order.indexOf(targetCol);
    if (fromIdx < 0 || toIdx < 0) return;

    // Shift semantics: source lands at target's slot; intermediate columns shift
    // by one toward source's old slot. Splicing in at the target's *original*
    // index achieves this for both directions: when fromIdx < toIdx, removing
    // source first shifts target down by one, so inserting at the original
    // toIdx places source just past target; when fromIdx > toIdx, target's
    // index is unchanged, so inserting at toIdx places source just before it.
    const next = [...order];
    next.splice(fromIdx, 1);
    next.splice(toIdx, 0, sourceCol);

    if (table === 'datasets') {
      this.datasetColumnOrder = next;
    } else {
      this.modelColumnOrder = next;
    }
    this.persistColumnOrder(table);
  }

  onColDragEnd(_event: DragEvent): void {
    this.dragCol = null;
    this.dropTargetCol = null;
  }

  isDropTarget(table: 'datasets' | 'models', col: string): boolean {
    return !!this.dropTargetCol
      && this.dropTargetCol.table === table
      && this.dropTargetCol.col === col;
  }

  isDragging(table: 'datasets' | 'models', col: string): boolean {
    return !!this.dragCol && this.dragCol.table === table && this.dragCol.col === col;
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
    this.polling$.next();
    this.polling$.complete();
    this.modelPolling$.next();
    this.modelPolling$.complete();
    this.findPolling$.next();
    this.findPolling$.complete();
  }

  get datasets(): DatasetRegistryEntry[] {
    return this.datasetState.datasets;
  }

  get models(): ModelRegistryEntry[] {
    return this.datasetState.models;
  }

  get loading(): boolean {
    return this.datasetState.loading;
  }

  get progressMessage(): string {
    return this.datasetState.progressMessage;
  }

  get errorMessage(): string {
    return this.datasetState.errorMessage;
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

  dismissError(): void {
    this.datasetState.setErrorMessage('');
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

    const selModels = this.models.filter((m) => this.selectedModelIds.has(m.id));
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

  // --- Model selection ---

  toggleModelSelection(id: string, event: MouseEvent): void {
    if (event.ctrlKey || event.metaKey) {
      if (this.selectedModelIds.has(id)) {
        this.selectedModelIds.delete(id);
      } else {
        this.selectedModelIds.add(id);
      }
    } else {
      if (this.selectedModelIds.has(id) && this.selectedModelIds.size === 1) {
        this.selectedModelIds.clear();
      } else {
        this.selectedModelIds.clear();
        this.selectedModelIds.add(id);
      }
    }
    this.pushTopBarLabels();
  }

  isModelSelected(id: string): boolean {
    return this.selectedModelIds.has(id);
  }

  // --- Dataset actions ---

  renameDataset(dataset: DatasetRegistryEntry, newName: string): void {
    this.datasetsApi.renameRegistered(dataset.id, newName).subscribe({
      next: () => this.datasetState.refresh(),
    });
  }

  async deleteDataset(dataset: DatasetRegistryEntry): Promise<void> {
    this.deletingDatasetId = dataset.id;
    const ok = await this.dialog.confirm(`Delete dataset "${dataset.name}"?`);
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

  renameModel(model: ModelRegistryEntry, newName: string): void {
    this.modelsApi.renameInRegistry(model.id, newName).subscribe({
      next: () => this.datasetState.refresh(),
    });
  }

  async deleteModel(model: ModelRegistryEntry): Promise<void> {
    this.deletingModelId = model.id;
    const ok = await this.dialog.confirm(`Delete model "${model.name}"?`);
    this.deletingModelId = '';
    if (!ok) return;
    this.modelsApi.deleteFromRegistry(model.id).subscribe({
      next: () => {
        this.selectedModelIds.delete(model.id);
        this.datasetState.refresh();
      },
      error: () => {
        this.dialog.alert('Failed to delete model. Please try again.', 'error');
      },
    });
  }

  loadDataset(dataset: DatasetRegistryEntry): void {
    this.datasetsApi.loadRegistered(dataset.id).subscribe({
      next: () => this.startProgressPolling(),
    });
  }

  loadModel(model: ModelRegistryEntry): void {
    this.modelsApi.loadModel(model.id).subscribe({
      next: () => this.startModelProgressPolling(),
    });
  }

  unloadModel(model: ModelRegistryEntry): void {
    this.modelsApi.unloadModel(model.id).subscribe({
      next: () => this.datasetState.refresh(),
    });
  }

  getInlineModelTask(modelId: string): LoadingTask | undefined {
    return this.modelLoadingTasks.find((t) => t.model_id === modelId);
  }

  toggleAutorun(model: ModelRegistryEntry, autorun: boolean): void {
    const detectorName = model.detector_name || model.name;
    this.detectorsApi.setAutodetect(detectorName, autorun).subscribe({
      next: () => this.datasetState.refresh(),
    });
  }

  // --- Export modal ---

  openExportModal(model: ModelRegistryEntry): void {
    this.exportModelName = model.detector_name || model.name;
    this.exportModalOpen = true;
  }

  closeExportModal(): void {
    this.exportModalOpen = false;
    this.exportModelName = '';
  }

  // --- Add Labels modal ---

  openAddLabelsModal(model: ModelRegistryEntry): void {
    this.addLabelsModelId = model.id;
    this.addLabelsModelName = (model['trainable_model_name'] as string) || model.name;
    this.addLabelsModalOpen = true;
  }

  closeAddLabelsModal(): void {
    this.addLabelsModalOpen = false;
    this.addLabelsModelId = '';
    this.addLabelsModelName = '';
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
    for (const m of this.models) {
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
    this.importerModalOpen = true;
  }

  closeImporterModal(): void {
    this.importerModalOpen = false;
    this.importerClosing = true;
  }

  onImporterAnimationEnd(): void {
    this.importerClosing = false;
  }

  onImportComplete(): void {
    this.importerModalOpen = false;
    this.importerClosing = true;
    this.startProgressPolling();
  }

  onDemoSelected(demo: { label: string; name: string; embedder?: string; clipper?: string }): void {
    this.importerModalOpen = false;
    this.importerClosing = true;
    const params: Record<string, string> = {};
    if (demo.embedder) {
      params['embedder'] = demo.embedder;
    }
    if (demo.clipper) {
      params['clipper'] = demo.clipper;
    }
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

  openNewModelModal(): void {
    this.newModelClosing = false;
    this.newModelModalOpen = true;
  }

  closeNewModelModal(): void {
    this.newModelModalOpen = false;
    this.newModelClosing = true;
    this.trainAfterModelCreation = false;
  }

  onNewModelAnimationEnd(): void {
    this.newModelClosing = false;
  }

  onModelCreated(modelId?: string): void {
    this.newModelModalOpen = false;
    this.newModelClosing = true;
    this.datasetState.refresh();

    if (this.trainAfterModelCreation && modelId) {
      this.trainAfterModelCreation = false;
      // Select the newly created model and proceed to training once models list is refreshed
      this.selectedModelIds.clear();
      this.selectedModelIds.add(modelId);
      this.knownModelIds.add(modelId);
      this.datasetState.models$
        .pipe(
          filter((models) => models.some((m) => m.id === modelId)),
          take(1),
          takeUntil(this.destroy$),
        )
        .subscribe(() => this.onLabel());
    }
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

  cancelModelLoadingTask(taskId: string): void {
    this.modelsApi.cancelModelLoadingTask(taskId).subscribe();
  }

  dismissModelLoadingTask(taskId: string): void {
    this.modelLoadingTasks = this.modelLoadingTasks.filter((t) => t.task_id !== taskId);
  }

  // --- Progress polling ---

  startProgressPolling(onComplete?: () => void): void {
    this.datasetState.setErrorMessage('');

    // If polling is already active, don't restart — the existing loop
    // already covers all tasks.  This avoids clearing completedTaskIds
    // and losing track of tasks that just finished.
    if (this.datasetPollingActive) {
      return;
    }
    this.datasetPollingActive = true;
    this.completedTaskIds.clear();

    timer(0, 1000)
      .pipe(
        takeUntil(this.polling$),
        takeUntil(this.destroy$),
        switchMap(() => this.datasetsApi.getLoadingTasks().pipe(
          catchError(() => EMPTY),
        )),
      )
      .subscribe({
        next: (tasks: LoadingTask[]) => {
          // Separate active from finished
          const active = tasks.filter((t) => t.status !== 'idle');
          const errored = tasks.filter((t) => t.status === 'idle' && !!t.error);
          const cancelled = errored.filter((t) => t.error === 'Cancelled');
          const failed = errored.filter((t) => t.error !== 'Cancelled');

          // Show both active tasks and failed tasks (so users see the error)
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
          }

          // Also set the top-level error banner for failed tasks
          for (const t of failed) {
            this.datasetState.setErrorMessage(t.error!);
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

  startModelProgressPolling(onComplete?: () => void): void {
    if (this.modelPollingActive) {
      return;
    }
    this.modelPollingActive = true;
    this.completedModelTaskIds.clear();

    timer(0, 1000)
      .pipe(
        takeUntil(this.modelPolling$),
        takeUntil(this.destroy$),
        switchMap(() => this.modelsApi.getModelLoadingTasks().pipe(
          catchError(() => EMPTY),
        )),
      )
      .subscribe({
        next: (resp: LoadingTasksResponse) => {
          const tasks = resp.tasks ?? [];
          const active = tasks.filter((t) => t.status !== 'idle');
          const errored = tasks.filter((t) => t.status === 'idle' && !!t.error);
          const failed = errored.filter((t) => t.error !== 'Cancelled');

          this.modelLoadingTasks = [...active, ...failed];

          // Detect tasks that just completed successfully
          const justFinished = tasks.filter(
            (t) => t.status === 'idle' && !t.error && !this.completedModelTaskIds.has(t.task_id),
          );
          for (const t of justFinished) {
            this.completedModelTaskIds.add(t.task_id);
          }
          if (justFinished.length > 0) {
            this.datasetState.refresh();
          }

          for (const t of failed) {
            this.datasetState.setErrorMessage(t.error!);
          }

          if (active.length === 0) {
            this.modelPolling$.next();
            this.modelPollingActive = false;
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

  sortDatasets(column: string): void {
    if (this.datasetSortColumn === column) {
      this.datasetSortAsc = !this.datasetSortAsc;
    } else {
      this.datasetSortColumn = column;
      this.datasetSortAsc = true;
    }
  }

  get sortedDatasets(): DatasetRegistryEntry[] {
    const col = this.datasetSortColumn;
    const asc = this.datasetSortAsc ? 1 : -1;
    return [...this.datasets].sort((a, b) => {
      const va = a[col] ?? '';
      const vb = b[col] ?? '';
      if (typeof va === 'number' && typeof vb === 'number') return (va - vb) * asc;
      return String(va).localeCompare(String(vb)) * asc;
    });
  }

  sortModels(column: string): void {
    if (this.modelSortColumn === column) {
      this.modelSortAsc = !this.modelSortAsc;
    } else {
      this.modelSortColumn = column;
      this.modelSortAsc = true;
    }
  }

  get sortedModels(): ModelRegistryEntry[] {
    const col = this.modelSortColumn;
    const asc = this.modelSortAsc ? 1 : -1;
    return [...this.models].sort((a, b) => {
      const va = a[col] ?? '';
      const vb = b[col] ?? '';
      if (typeof va === 'number' && typeof vb === 'number') return (va - vb) * asc;
      return String(va).localeCompare(String(vb)) * asc;
    });
  }

  datasetSortIndicator(column: string): string {
    if (this.datasetSortColumn !== column) return '\u25B2';
    return this.datasetSortAsc ? '\u25B2' : '\u25BC';
  }

  isDatasetSortActive(column: string): boolean {
    return this.datasetSortColumn === column;
  }

  modelSortIndicator(column: string): string {
    if (this.modelSortColumn !== column) return '\u25B2';
    return this.modelSortAsc ? '\u25B2' : '\u25BC';
  }

  isModelSortActive(column: string): boolean {
    return this.modelSortColumn === column;
  }

  // --- Button state ---

  get isLoading(): boolean {
    return this.trainLoading || this.findLoading;
  }

  get labelEnabled(): boolean {
    const selectedDatasets = this.resolvedSelectedDatasets;
    const selectedModels = this.resolvedSelectedModels;
    if (selectedDatasets.length !== 1) return false;
    if (selectedModels.length === 0) return true; // will prompt to create a model
    if (selectedModels.length !== 1) return false;
    const model = selectedModels[0];
    if (!model.trainable) return false;
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
    return this.resolvedSelectedModels.some((m) => m.trainable && (m.num_training ?? 0) === 0);
  }

  private get resolvedSelectedDatasets(): DatasetRegistryEntry[] {
    return this.datasets.filter((d) => this.selectedDatasetIds.has(d.id));
  }

  private get resolvedSelectedModels(): ModelRegistryEntry[] {
    return this.models.filter((m) => this.selectedModelIds.has(m.id));
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
    if (nDatasets === 0 && nModels === 0) return 'Select a dataset and a model';
    if (nDatasets === 0) return 'Select a dataset';
    if (nModels === 0) return 'Select a model';
    if (!this.findMediaTypesMatch()) return 'Media type mismatch';
    if (this.hasUntrainedModel()) return 'Selected model has no training labels';
    return 'Score selected datasets with selected models';
  }

  get labelHint(): string {
    const nDatasets = this.resolvedSelectedDatasets.length;
    const nModels = this.resolvedSelectedModels.length;
    if (nDatasets === 0) return 'Select a dataset';
    if (nDatasets > 1) return 'Select exactly 1 dataset';
    if (nModels === 0) return 'Create a new model and start training';
    if (nModels > 1) return 'Select exactly 1 model';
    const model = this.resolvedSelectedModels[0];
    if (model && !model.trainable) return 'Model is not trainable';
    const dataset = this.resolvedSelectedDatasets[0];
    if (model && dataset && model.media_type !== dataset.media_type) {
      return 'Media type mismatch';
    }
    return 'Open Train Mode with the selected dataset and model';
  }

  private storeSelectedModelTextQuery(): void {
    const model = this.models.find((m) => this.selectedModelIds.has(m.id));
    this.labelSession.textQuery = model?.text_query || '';
    this.labelSession.mediaExample = model?.media_example || '';
    this.labelSession.examples = (model?.['examples'] as { type: string; value: string }[]) || [];
    this.labelSession.modelName = model?.name || '';
  }

  onLabel(): void {
    const dataset = this.datasets.find((d) => this.selectedDatasetIds.has(d.id));
    if (!dataset) return;

    const modelId = [...this.selectedModelIds][0] || null;
    if (!modelId) {
      // No model selected — open the New Model modal; on creation we'll proceed to training
      this.trainAfterModelCreation = true;
      this.openNewModelModal();
      return;
    }
    const model = modelId ? this.models.find((m) => m.id === modelId) : null;

    this.storeSelectedModelTextQuery();
    this.trainLoading = true;

    // Set active context so the HTTP interceptor attaches headers
    this.activeContext.setDatasetId(dataset.id);
    this.activeContext.setModelId(modelId || '');

    // Gate: navigate only once both dataset and model are ready.
    let pending = 2;
    const gate = (): void => {
      if (--pending === 0) {
        this.trainLoading = false;
        this.datasetState.refresh();
        this.router.navigate(['/label']);
      }
    };

    // --- Model loading (parallel) ---
    if (model && !model.detector_loaded) {
      this.modelsApi.loadModel(modelId).subscribe({
        next: () => this.startModelProgressPolling(() => gate()),
        error: () => gate(),
      });
    } else {
      gate();
    }

    // --- Dataset loading (parallel) ---
    if (dataset.loaded) {
      gate();
    } else {
      this.datasetsApi.loadRegistered(dataset.id).subscribe({
        next: () => {
          this.startProgressPolling(() => gate());
        },
      });
    }
  }

  onFind(): void {
    const dataset = this.datasets.find((d) => this.selectedDatasetIds.has(d.id));
    if (!dataset) return;

    const model = this.models.find((m) => this.selectedModelIds.has(m.id));
    if (!model) return;

    // Store model and dataset info in the find session service
    this.findSession.modelId = model.id;
    this.findSession.modelName = model.name;
    this.findSession.datasetId = dataset.id;
    this.findLoading = true;

    // Set active context so the HTTP interceptor attaches headers
    this.activeContext.setDatasetId(dataset.id);
    this.activeContext.setModelId(model.id);

    // Gate: navigate only once both dataset and model are ready.
    let pending = 2;
    const gate = (): void => {
      if (--pending === 0) {
        this.findLoading = false;
        this.datasetState.refresh();
        this.router.navigate(['/find']);
      }
    };

    // --- Model loading (parallel) ---
    if (!model.detector_loaded) {
      this.modelsApi.loadModel(model.id).subscribe({
        next: () => this.startModelProgressPolling(() => gate()),
        error: () => gate(),
      });
    } else {
      gate();
    }

    // --- Dataset loading (parallel) ---
    if (dataset.loaded) {
      gate();
    } else {
      this.datasetsApi.loadRegistered(dataset.id).subscribe({
        next: () => {
          this.startProgressPolling(() => gate());
        },
      });
    }
  }

  /** Old Find window — runs multi-dataset multi-model find and shows results modal. */
  onOldFind(): void {
    const datasetIds = [...this.selectedDatasetIds];
    const modelIds = [...this.selectedModelIds];
    const findParams = { dataset_ids: datasetIds, model_ids: modelIds };

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
            (w) => `${w.failed_labels} of ${w.total_labels} labels failed to resolve for "${w.model_name}".`,
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
    timer(0, 500)
      .pipe(
        takeUntil(this.findPolling$),
        takeUntil(this.destroy$),
        switchMap(() => this.detectorsApi.getFindProgress().pipe(
          catchError(() => EMPTY),
        )),
      )
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
          model_verdicts: r.model_verdicts || {},
        });

        const hits = (response.results || []).map(mapHit);
        const negativeHits = (response.negative_results || []).map(mapHit);

        const modelNames: string[] = response.models || [];
        const detectorResults: Record<string, any> = {};

        if (modelNames.length <= 1) {
          // Single model: one result group
          const label = modelNames[0] || 'Find';
          detectorResults[label] = {
            detector_name: label,
            total_hits: hits.length,
            hits,
            negative_hits: negativeHits,
          };
        } else {
          // Multiple models: group hits by model
          for (const name of modelNames) {
            const modelHits = hits.filter(
              (h: any) => h.model_verdicts?.[name]?.verdict === 'Good',
            );
            const modelNegHits = negativeHits.filter(
              (h: any) => h.model_verdicts?.[name]?.verdict !== 'Good',
            );
            detectorResults[name] = {
              detector_name: name,
              total_hits: modelHits.length,
              hits: modelHits,
              negative_hits: modelNegHits,
            };
          }
        }

        this.findResultsData = {
          media_type: response.media_type || 'unknown',
          detectors_run: modelNames.length,
          results: detectorResults,
          models: modelNames,
          datasets: response.datasets || [],
          multiple_datasets: response.multiple_datasets || false,
          multiple_models: response.multiple_models || false,
        } as AutoDetectResultsData;

        this.findResultsOpen = true;
      },
      error: (err) => {
        this.stopFindProgressPolling();
        this.datasetState.setLoading(false);
        this.progressIndeterminate = false;
        this.dialog.alert(err.error?.error || 'Find failed.', 'error');
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
