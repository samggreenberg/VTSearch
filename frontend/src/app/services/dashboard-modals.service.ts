import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { AutoDetectResultsData, DatasetRegistryEntry } from '../models/api.models';

export interface CombineDatasetsState {
  open: boolean;
  datasets: DatasetRegistryEntry[];
}

export interface CombineDetectorsState {
  open: boolean;
}

export interface ExportState {
  open: boolean;
  detectorName: string;
}

export interface AddLabelsState {
  open: boolean;
  detectorId: string;
  detectorName: string;
}

export interface FindResultsState {
  open: boolean;
  data: AutoDetectResultsData;
}

export interface StatsState {
  open: boolean;
  datasetId: string;
  datasetName: string;
}

/**
 * Singleton owner of the Dashboard's row-action and selection-action
 * modal states. Lifting these out of `DashboardComponent` keeps the
 * dashboard a thinner layout/wiring shell - opening any of these modals
 * is `modals.openX(...)`, closing is `modals.closeX()`, and the modal
 * `@if` blocks in the template read directly off `modals.x.open`.
 *
 * The dataset-importer and new-detector flows stay on
 * `NewThingFlowsService` (they're opened from places other than the
 * dashboard too - e.g. the top-bar context pulldowns).
 */
@Injectable({ providedIn: 'root' })
export class DashboardModalsService {
  private readonly combineDatasetsSubject = new BehaviorSubject<CombineDatasetsState>({
    open: false,
    datasets: [],
  });
  private readonly combineDetectorsSubject = new BehaviorSubject<CombineDetectorsState>({
    open: false,
  });
  private readonly exportSubject = new BehaviorSubject<ExportState>({
    open: false,
    detectorName: '',
  });
  private readonly addLabelsSubject = new BehaviorSubject<AddLabelsState>({
    open: false,
    detectorId: '',
    detectorName: '',
  });
  private readonly findResultsSubject = new BehaviorSubject<FindResultsState>({
    open: false,
    data: { results: {} },
  });
  private readonly statsSubject = new BehaviorSubject<StatsState>({
    open: false,
    datasetId: '',
    datasetName: '',
  });

  readonly combineDatasets$ = this.combineDatasetsSubject.asObservable();
  readonly combineDetectors$ = this.combineDetectorsSubject.asObservable();
  readonly export$ = this.exportSubject.asObservable();
  readonly addLabels$ = this.addLabelsSubject.asObservable();
  readonly findResults$ = this.findResultsSubject.asObservable();
  readonly stats$ = this.statsSubject.asObservable();

  get combineDatasets(): CombineDatasetsState {
    return this.combineDatasetsSubject.value;
  }

  get combineDetectors(): CombineDetectorsState {
    return this.combineDetectorsSubject.value;
  }

  get export(): ExportState {
    return this.exportSubject.value;
  }

  get addLabels(): AddLabelsState {
    return this.addLabelsSubject.value;
  }

  get findResults(): FindResultsState {
    return this.findResultsSubject.value;
  }

  get stats(): StatsState {
    return this.statsSubject.value;
  }

  openCombineDatasets(datasets: DatasetRegistryEntry[]): void {
    this.combineDatasetsSubject.next({ open: true, datasets });
  }

  closeCombineDatasets(): void {
    this.combineDatasetsSubject.next({ open: false, datasets: [] });
  }

  openCombineDetectors(): void {
    this.combineDetectorsSubject.next({ open: true });
  }

  closeCombineDetectors(): void {
    this.combineDetectorsSubject.next({ open: false });
  }

  openExport(detectorName: string): void {
    this.exportSubject.next({ open: true, detectorName });
  }

  closeExport(): void {
    this.exportSubject.next({ open: false, detectorName: '' });
  }

  openAddLabels(detectorId: string, detectorName: string): void {
    this.addLabelsSubject.next({ open: true, detectorId, detectorName });
  }

  closeAddLabels(): void {
    this.addLabelsSubject.next({ open: false, detectorId: '', detectorName: '' });
  }

  openFindResults(data: AutoDetectResultsData): void {
    this.findResultsSubject.next({ open: true, data });
  }

  closeFindResults(): void {
    this.findResultsSubject.next({ open: false, data: { results: {} } });
  }

  openStats(datasetId: string, datasetName: string): void {
    this.statsSubject.next({ open: true, datasetId, datasetName });
  }

  closeStats(): void {
    this.statsSubject.next({ open: false, datasetId: '', datasetName: '' });
  }
}
