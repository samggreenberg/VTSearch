import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { EMPTY, Subject, timer } from 'rxjs';
import { catchError, takeUntil, switchMap } from 'rxjs/operators';
import { DatasetsApiService } from '../../services/datasets-api.service';
import { DetectorsApiService } from '../../services/detectors-api.service';
import { TrainableModelsApiService } from '../../services/trainable-models-api.service';
import { VtDialogService } from '../../services/dialog.service';
import { LabelSessionService } from '../../services/label-session.service';
import { DatasetStateService } from '../../services/dataset-state.service';
import { AuthService } from '../../services/auth.service';
import { AutoDetectResultsData, DatasetRegistryEntry, ModelRegistryEntry } from '../../models/api.models';
import { ProgressBarComponent } from '../progress-bar/progress-bar.component';
import { AutoDetectResultsModalComponent } from '../modals/autodetect-results-modal/autodetect-results-modal.component';
import { DatasetCardComponent } from './dataset-card/dataset-card.component';
import { ModelCardComponent } from './model-card/model-card.component';
import { DatasetImporterModalComponent } from './dataset-importer-modal/dataset-importer-modal.component';
import { NewModelModalComponent } from './new-model-modal/new-model-modal.component';

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

  importerModalOpen = false;
  newModelModalOpen = false;
  findResultsOpen = false;
  findResultsData: AutoDetectResultsData = { results: {} };

  datasetSortColumn = 'name';
  datasetSortAsc = true;
  modelSortColumn = 'name';
  modelSortAsc = true;

  private destroy$ = new Subject<void>();
  private polling$ = new Subject<void>();
  private knownDatasetIds = new Set<string>();
  private knownModelIds = new Set<string>();

  currentUser = '';
  isDefaultLogin = true;

  constructor(
    private router: Router,
    private datasetsApi: DatasetsApiService,
    private detectorsApi: DetectorsApiService,
    private modelsApi: TrainableModelsApiService,
    private dialog: VtDialogService,
    private labelSession: LabelSessionService,
    public datasetState: DatasetStateService,
    private authService: AuthService,
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
          // Items were added after initial load — select the new ones
          for (const id of newIds) {
            this.selectedDatasetIds.add(id);
          }
        } else if (datasets.length === 1 && this.selectedDatasetIds.size === 0) {
          // First load with exactly one item — auto-select it
          this.selectedDatasetIds.add(datasets[0].id);
        }
        this.knownDatasetIds = currentIds;
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
          // Items were added after initial load — select the new ones
          for (const id of newIds) {
            this.selectedModelIds.add(id);
          }
        } else if (models.length === 1 && this.selectedModelIds.size === 0) {
          // First load with exactly one item — auto-select it
          this.selectedModelIds.add(models[0].id);
        }
        this.knownModelIds = currentIds;
      });
    this.refresh();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
    this.polling$.next();
    this.polling$.complete();
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

  refresh(): void {
    this.datasetState.refresh();
  }

  // --- Dataset selection ---

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
    const ok = await this.dialog.confirm(`Delete dataset "${dataset.name}"?`);
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

  // --- Model actions ---

  renameModel(model: ModelRegistryEntry, newName: string): void {
    this.modelsApi.renameInRegistry(model.id, newName).subscribe({
      next: () => this.datasetState.refresh(),
    });
  }

  async deleteModel(model: ModelRegistryEntry): Promise<void> {
    const ok = await this.dialog.confirm(`Delete model "${model.name}"?`);
    if (!ok) return;
    this.modelsApi.deleteFromRegistry(model.id).subscribe({
      next: () => {
        this.selectedModelIds.delete(model.id);
        this.datasetState.refresh();
      },
    });
  }

  // --- Importer modal ---

  openImporterModal(): void {
    this.importerModalOpen = true;
  }

  closeImporterModal(): void {
    this.importerModalOpen = false;
  }

  onImportComplete(): void {
    this.importerModalOpen = false;
    this.startProgressPolling();
  }

  onDemoSelected(demo: { label: string; name: string; embedder?: string; clipper?: string }): void {
    this.importerModalOpen = false;
    this.datasetState.setLoading(true);
    this.datasetState.setProgressMessage(`Loading demo: ${demo.label}...`);
    this.progressIndeterminate = true;
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
      error: () => {
        this.datasetState.setLoading(false);
        this.progressIndeterminate = false;
      },
    });
  }

  // --- New model modal ---

  openNewModelModal(): void {
    this.newModelModalOpen = true;
  }

  closeNewModelModal(): void {
    this.newModelModalOpen = false;
  }

  onModelCreated(): void {
    this.newModelModalOpen = false;
    this.datasetState.refresh();
  }

  // --- Cancel ---

  onCancelIngest(): void {
    this.datasetsApi.cancelIngest().subscribe();
  }

  // --- Progress polling ---

  startProgressPolling(onComplete?: () => void): void {
    this.datasetState.setLoading(true);
    this.progressIndeterminate = true;
    this.polling$.next(); // cancel previous polling

    timer(0, 1000)
      .pipe(
        takeUntil(this.polling$),
        takeUntil(this.destroy$),
        switchMap(() => this.datasetsApi.getProgress().pipe(
          catchError(() => EMPTY),
        )),
      )
      .subscribe({
        next: (progress: any) => {
          if (progress.current != null && progress.total != null && progress.total > 0) {
            this.progressIndeterminate = false;
            this.progressValue = progress.current;
            this.progressTotal = progress.total;
          } else {
            this.progressIndeterminate = true;
          }

          // Build message with step info and percentage when available
          let msg = progress.message || 'Loading...';
          if (progress.step != null && progress.total_steps != null && progress.total_steps > 1) {
            msg = `[Step ${progress.step}/${progress.total_steps}] ${msg}`;
          }
          if (progress.current != null && progress.total != null && progress.total > 0) {
            const pct = Math.min(100, Math.round((progress.current / progress.total) * 100));
            msg += ` (${pct}%)`;
          }
          this.datasetState.setProgressMessage(msg);

          if (progress.status === 'idle' || progress.status === 'error') {
            this.datasetState.setLoading(false);
            this.progressIndeterminate = false;
            this.polling$.next();
            this.datasetState.refresh();
            if (progress.status === 'idle' && onComplete) {
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

  get labelEnabled(): boolean {
    const selectedDatasets = this.resolvedSelectedDatasets;
    const selectedModels = this.resolvedSelectedModels;
    if (selectedDatasets.length !== 1 || selectedModels.length !== 1) return false;
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
    if (nModels === 0) return 'Select a model';
    if (nModels > 1) return 'Select exactly 1 model';
    const model = this.resolvedSelectedModels[0];
    if (model && !model.trainable) return 'Model is not trainable';
    const dataset = this.resolvedSelectedDatasets[0];
    if (model && dataset && model.media_type !== dataset.media_type) {
      return 'Media type mismatch';
    }
    return 'Label the selected dataset to train the selected model';
  }

  private storeSelectedModelTextQuery(): void {
    const model = this.models.find((m) => this.selectedModelIds.has(m.id));
    this.labelSession.textQuery = model?.text_query || '';
    this.labelSession.mediaExample = model?.media_example || '';
    this.labelSession.examples = (model?.['examples'] as { type: string; value: string }[]) || [];
  }

  onLabel(): void {
    const dataset = this.datasets.find((d) => this.selectedDatasetIds.has(d.id));
    if (!dataset) return;

    const modelId = [...this.selectedModelIds][0] || null;

    const navigateToLabel = (): void => {
      this.storeSelectedModelTextQuery();
      // Tell the backend which model is loaded so votes auto-sync
      this.modelsApi.loadModel(modelId).subscribe({
        next: () => this.router.navigate(['/label']),
        error: () => this.router.navigate(['/label']),
      });
    };

    if (dataset.loaded) {
      navigateToLabel();
      return;
    }

    // Dataset not loaded — load it first, then navigate
    this.datasetState.setLoading(true);
    this.datasetState.setProgressMessage(`Loading ${dataset.name}...`);
    this.progressIndeterminate = true;
    this.datasetsApi.loadRegistered(dataset.id).subscribe({
      next: () => {
        this.startProgressPolling(() => {
          navigateToLabel();
        });
      },
      error: () => {
        this.datasetState.setLoading(false);
        this.progressIndeterminate = false;
      },
    });
  }

  onFind(): void {
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

  private runFind(findParams: Record<string, unknown>): void {
    this.datasetState.setProgressMessage('Running Find...');

    this.detectorsApi.find(findParams).subscribe({
      next: (response: any) => {
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
        this.datasetState.setLoading(false);
        this.progressIndeterminate = false;
        this.dialog.alert(err.error?.error || 'Find failed.', 'error');
      },
    });
  }

  closeFindResults(): void {
    this.findResultsOpen = false;
  }
}
