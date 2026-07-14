import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpTestingController } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { DashboardComponent } from './dashboard.component';
import { LoadingTask } from '../../models/api.models';
import { LabelSessionService } from '../../services/label-session.service';
import { NewThingFlowsService } from '../../services/new-thing-flows.service';
import { ActiveContextService } from '../../services/active-context.service';
import { provideZoneless } from '../../testing/zoneless-testbed';
import { provideHttpTesting } from '../../testing/test-providers';

describe('DashboardComponent', () => {
  let component: DashboardComponent;
  let fixture: ComponentFixture<DashboardComponent>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DashboardComponent],
      providers: [...provideZoneless(), ...provideHttpTesting(), provideRouter([])],
    }).compileComponents();

    fixture = TestBed.createComponent(DashboardComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    // Best-effort background pollers (disk/RAM usage on a timer, embedder
    // preloads triggered by auto-selection) may have fired during the test
    // without being asserted. Drain them so verify() only fails on
    // genuinely-unexpected requests.
    drainBackgroundRequests();
    httpMock.verify();
  });

  /** Flush the fire-and-forget requests the dashboard issues on init and
   *  as a side effect of selection: the disk/RAM usage pollers
   *  (`timer(0, 10000)`) and the per-dataset embedder preload. None of
   *  these are asserted on by the selection/hint/button tests, so we just
   *  swallow whatever has accumulated. */
  function drainBackgroundRequests(): void {
    for (const req of httpMock.match('/api/dashboard/disk-usage')) {
      if (!req.cancelled) req.flush({ total: 0, used: 0, free: 0 });
    }
    for (const req of httpMock.match('/api/dashboard/ram-usage')) {
      if (!req.cancelled) req.flush({ total: 0, used: 0, free: 0 });
    }
    for (const req of httpMock.match((r) => /\/preload-embedder$/.test(r.url))) {
      if (!req.cancelled) req.flush({ ok: true, embedder: '' });
    }
  }

  function flushInitialRequests(
    datasets: any[] = [],
    detectors: any[] = [],
    importers: any[] = [],
  ): void {
    TestBed.tick();
    httpMock.expectOne('/api/datasets/registry').flush({ datasets });
    httpMock.expectOne('/api/detectors/registry').flush({ detectors });
    httpMock.expectOne('/api/dataset/all-importers').flush({ importers, tabs: [] });
    // DatasetStateService is signal-backed and exposes `datasets$`/`detectors$`
    // as `toObservable` bridges, which emit on the next change-detection pass
    // (not synchronously when the signal is set). Tick once more so the
    // dashboard's `datasets$`/`detectors$` subscriptions (registry auto-select)
    // run before the test asserts on the resulting selection state.
    TestBed.tick();
    // Auto-selecting a single dataset kicks off an embedder preload; drain it
    // (and any usage polls already in flight) here so tests that don't assert
    // on them stay clean.
    drainBackgroundRequests();
  }

  it('should create', () => {
    flushInitialRequests();
    expect(component).toBeTruthy();
  });

  it('should fetch datasets and detectors on init', () => {
    const datasets = [{ id: 'd1', name: 'Test Dataset', media_type: 'audio', num_items: 10 }];
    const detectors = [{ id: 'm1', name: 'Test Detector' }];
    flushInitialRequests(datasets, detectors);
    expect(component.datasets.length).toBe(1);
    expect(component.detectors.length).toBe(1);
  });

  it('should auto-select single dataset', () => {
    const datasets = [{ id: 'd1', name: 'Only One' }];
    flushInitialRequests(datasets);
    expect(component.selectedDatasetIds.has('d1')).toBe(true);
  });

  it('should auto-select single model', () => {
    const models = [{ id: 'm1', name: 'Only One' }];
    flushInitialRequests([], models);
    expect(component.selectedDetectorIds.has('m1')).toBe(true);
  });

  it('should not auto-select when multiple datasets on initial load', () => {
    const datasets = [
      { id: 'd1', name: 'First' },
      { id: 'd2', name: 'Second' },
    ];
    flushInitialRequests(datasets);
    expect(component.selectedDatasetIds.size).toBe(0);
  });

  it('should auto-select newly added dataset', () => {
    const datasets = [{ id: 'd1', name: 'First' }];
    flushInitialRequests(datasets);
    expect(component.selectedDatasetIds.has('d1')).toBe(true);

    // Simulate adding a second dataset via refresh
    component.refresh();
    httpMock.expectOne('/api/datasets/registry').flush({
      datasets: [
        { id: 'd1', name: 'First' },
        { id: 'd2', name: 'Second' },
      ],
    });
    httpMock.expectOne('/api/detectors/registry').flush({ detectors: [] });
    // Let the `datasets$` bridge deliver the new registry to the auto-select sub.
    TestBed.tick();

    expect(component.selectedDatasetIds.has('d1')).toBe(false);
    expect(component.selectedDatasetIds.has('d2')).toBe(true);
  });

  it('should auto-select newly added model', () => {
    const models = [{ id: 'm1', name: 'First' }];
    flushInitialRequests([], models);
    expect(component.selectedDetectorIds.has('m1')).toBe(true);

    // Simulate adding a second model via refresh
    component.refresh();
    httpMock.expectOne('/api/datasets/registry').flush({ datasets: [] });
    httpMock.expectOne('/api/detectors/registry').flush({
      detectors: [
        { id: 'm1', name: 'First' },
        { id: 'm2', name: 'Second' },
      ],
    });
    // Let the `detectors$` bridge deliver the new registry to the auto-select sub.
    TestBed.tick();

    // Adding items after the initial load clears the prior selection and
    // selects only the new ones (same behavior as datasets above).
    expect(component.selectedDetectorIds.has('m1')).toBe(false);
    expect(component.selectedDetectorIds.has('m2')).toBe(true);
  });

  it('mirrors an implicitly selected dataset into the active-context intent', () => {
    // Off the Dashboard the top-bar pulldowns read the active-context intent
    // (not the mirrored table selection), so an auto-selected import must land
    // there too or the picker forgets it the moment the Dashboard unmounts.
    const activeContext = TestBed.inject(ActiveContextService);
    const datasets = [{ id: 'd1', name: 'Only One' }];
    flushInitialRequests(datasets);
    expect(component.selectedDatasetIds.has('d1')).toBe(true);
    expect(activeContext.intentDatasetId).toBe('d1');
  });

  it('mirrors an implicitly selected model into the active-context intent', () => {
    const activeContext = TestBed.inject(ActiveContextService);
    const models = [{ id: 'm1', name: 'Only One' }];
    flushInitialRequests([], models);
    expect(component.selectedDetectorIds.has('m1')).toBe(true);
    expect(activeContext.intentModelId).toBe('m1');
  });

  it('does not blank out the intent when the selection is empty or multiple', () => {
    // A 0- or multi-selection is ambiguous, so it must leave a previously
    // loaded pair's intent alone rather than snapping the picker to a
    // placeholder.
    const activeContext = TestBed.inject(ActiveContextService);
    activeContext.setActivePair('d0', 'm0');
    const datasets = [
      { id: 'd1', name: 'First' },
      { id: 'd2', name: 'Second' },
    ];
    flushInitialRequests(datasets);
    // Multiple datasets on initial load → nothing auto-selected, so the
    // empty-selection mirror leaves the loaded pair's intent untouched.
    expect(component.selectedDatasetIds.size).toBe(0);
    expect(activeContext.intentDatasetId).toBe('d0');
    // Select both at once → an ambiguous multi-selection also leaves it alone.
    component.toggleAllDatasets();
    expect(component.selectedDatasetIds.size).toBe(2);
    expect(activeContext.intentDatasetId).toBe('d0');
    expect(activeContext.intentModelId).toBe('m0');
  });

  it('should toggle dataset selection on click', () => {
    flushInitialRequests();
    const event = new MouseEvent('click');
    component.toggleDatasetSelection('d1', event);
    expect(component.isDatasetSelected('d1')).toBe(true);
    component.toggleDatasetSelection('d1', event);
    expect(component.isDatasetSelected('d1')).toBe(false);
  });

  it('should support multi-select with ctrl key', () => {
    flushInitialRequests();
    const ctrlEvent = new MouseEvent('click', { ctrlKey: true });
    component.toggleDatasetSelection('d1', new MouseEvent('click'));
    component.toggleDatasetSelection('d2', ctrlEvent);
    expect(component.isDatasetSelected('d1')).toBe(true);
    expect(component.isDatasetSelected('d2')).toBe(true);
  });

  it('should replace selection without ctrl key', () => {
    flushInitialRequests();
    component.toggleDatasetSelection('d1', new MouseEvent('click'));
    component.toggleDatasetSelection('d2', new MouseEvent('click'));
    expect(component.isDatasetSelected('d1')).toBe(false);
    expect(component.isDatasetSelected('d2')).toBe(true);
  });

  it('should sort datasets by column', () => {
    const datasets = [
      { id: 'd1', name: 'Bravo', num_items: 5 },
      { id: 'd2', name: 'Alpha', num_items: 10 },
    ];
    flushInitialRequests(datasets);

    // 'name' is the initial sort column (ascending), so the list starts
    // sorted ascending; clicking it toggles to descending, then back.
    expect(component.sortedDatasets[0].name).toBe('Alpha');

    component.datasetCols.sortBy('name');
    expect(component.sortedDatasets[0].name).toBe('Bravo');

    component.datasetCols.sortBy('name');
    expect(component.sortedDatasets[0].name).toBe('Alpha');
  });

  it('should sort models by column', () => {
    const models = [
      { id: 'm1', name: 'Zeta', num_training: 5 },
      { id: 'm2', name: 'Alpha', num_training: 10 },
    ];
    flushInitialRequests([], models);

    // 'name' is the initial sort column (ascending) \u2192 Alpha first.
    expect(component.sortedDetectors[0].name).toBe('Alpha');
    // Toggling it flips to descending \u2192 Zeta first.
    component.detectorCols.sortBy('name');
    expect(component.sortedDetectors[0].name).toBe('Zeta');
  });

  it('should show sort indicators', () => {
    flushInitialRequests();
    // 'name' starts as the active ascending sort, so it already shows \u25B2.
    expect(component.datasetCols.sortIndicator('name')).toContain('\u25B2');
    expect(component.datasetCols.isSortActive('name')).toBe(true);
    // Clicking the active column toggles to descending \u2192 \u25BC.
    component.datasetCols.sortBy('name');
    expect(component.datasetCols.sortIndicator('name')).toContain('\u25BC');
    expect(component.datasetCols.isSortActive('other')).toBe(false);
  });

  describe('button state', () => {
    it('should disable Label when nothing selected', () => {
      flushInitialRequests();
      component.selectedDatasetIds.clear();
      component.selectedDetectorIds.clear();
      expect(component.labelEnabled).toBe(false);
    });

    it('should enable Label with 1 dataset + 1 model', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'audio' }];
      flushInitialRequests(datasets, models);
      // Auto-selected since only 1 each
      expect(component.labelEnabled).toBe(true);
    });

    it('should disable Label on media type mismatch', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'image' }];
      flushInitialRequests(datasets, models);
      expect(component.labelEnabled).toBe(false);
    });

    it('should disable Label when model media_type is "any" (not a valid type)', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'any' }];
      flushInitialRequests(datasets, models);
      expect(component.labelEnabled).toBe(false);
    });

    it('should disable Find with no selections', () => {
      flushInitialRequests();
      component.selectedDatasetIds.clear();
      component.selectedDetectorIds.clear();
      expect(component.findEnabled).toBe(false);
    });

    it('should enable Find with matching media types', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      // Find requires the detector to have training labels.
      const models = [{ id: 'm1', name: 'M', media_type: 'audio', num_training: 5 }];
      flushInitialRequests(datasets, models);
      expect(component.findEnabled).toBe(true);
    });

    it('should disable Find on media type mismatch', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'image' }];
      flushInitialRequests(datasets, models);
      expect(component.findEnabled).toBe(false);
    });

    it('should disable Find when multiple datasets have different media types', () => {
      const datasets = [
        { id: 'd1', name: 'DS1', media_type: 'audio' },
        { id: 'd2', name: 'DS2', media_type: 'image' },
      ];
      const models = [{ id: 'm1', name: 'M', media_type: 'audio' }];
      flushInitialRequests(datasets, models);
      component.selectedDatasetIds.add('d2');
      expect(component.findEnabled).toBe(false);
    });

    it('should enable Find when all selected items share media type', () => {
      const datasets = [
        { id: 'd1', name: 'DS1', media_type: 'image' },
        { id: 'd2', name: 'DS2', media_type: 'image' },
      ];
      const models = [
        { id: 'm1', name: 'M1', media_type: 'image', num_training: 5 },
        { id: 'm2', name: 'M2', media_type: 'image', num_training: 5 },
      ];
      flushInitialRequests(datasets, models);
      component.selectedDatasetIds.add('d2');
      component.selectedDetectorIds.add('m2');
      expect(component.findEnabled).toBe(true);
    });

    it('should disable Find when the selected model has 0 training', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'audio', num_training: 0 }];
      flushInitialRequests(datasets, models);
      expect(component.findEnabled).toBe(false);
    });

    it('should enable Find for a model with training', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'audio', num_training: 5 }];
      flushInitialRequests(datasets, models);
      expect(component.findEnabled).toBe(true);
    });
  });

  describe('parallel-task gating (isContextSwitching / isNavBusy)', () => {
    // Regression guard for the GRID parallelism fix. Background work
    // (dataset imports, card-initiated loads, browse-prep of another row)
    // sets `datasetState.loading`, but that must NOT freeze independent
    // dashboard actions — on a big machine you can saturate the import
    // slots and still start another import, create/delete a detector, or
    // change the selection. Only an in-flight *active-pair switch* gates
    // them, via `isContextSwitching`. Train/Find additionally wait out a
    // browse-prep (whose completion fires a competing /browse navigation),
    // via `isNavBusy`.

    it('keeps independent actions live while only a background load runs', () => {
      flushInitialRequests();
      vi.spyOn(component.datasetState, 'loading', 'get').mockReturnValue(true);
      // A background import/load is the ONLY thing in flight.
      expect(component.isContextSwitching).toBe(false);
      expect(component.isNavBusy).toBe(false);
    });

    it('gates the whole dashboard during an active-pair context switch', () => {
      flushInitialRequests();
      vi.spyOn(component['contextSwitch'], 'switching', 'get').mockReturnValue(true);
      expect(component.isContextSwitching).toBe(true);
      expect(component.isNavBusy).toBe(true);
    });

    it('gates on a Train or Find click intent', () => {
      flushInitialRequests();
      component.trainLoading.set(true);
      expect(component.isContextSwitching).toBe(true);
      component.trainLoading.set(false);
      component.findLoading.set(true);
      expect(component.isContextSwitching).toBe(true);
    });

    it('gates Train/Find during browse-prep but leaves independent actions live', () => {
      flushInitialRequests();
      vi.spyOn(component.browsePrep, 'preparing', 'get').mockReturnValue(true);
      expect(component.isContextSwitching).toBe(false);
      expect(component.isNavBusy).toBe(true);
    });
  });

  describe('label hints', () => {
    it('should hint about missing dataset', () => {
      flushInitialRequests();
      component.selectedDatasetIds.clear();
      expect(component.labelHint).toBe('Select a dataset row in the table above');
    });

    it('should hint about missing model', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      flushInitialRequests(datasets);
      // Single dataset is auto-selected; no detector selected.
      component.selectedDetectorIds.clear();
      expect(component.labelHint).toBe('Create a new detector and start training');
    });

    it('should hint about multiple datasets', () => {
      const datasets = [
        { id: 'd1', name: 'DS1', media_type: 'audio' },
        { id: 'd2', name: 'DS2', media_type: 'audio' },
      ];
      flushInitialRequests(datasets);
      component.selectedDatasetIds.add('d1');
      component.selectedDatasetIds.add('d2');
      expect(component.labelHint).toBe('Select exactly 1 dataset');
    });

    it('should hint about multiple models', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [
        { id: 'm1', name: 'M1', media_type: 'audio' },
        { id: 'm2', name: 'M2', media_type: 'audio' },
      ];
      flushInitialRequests(datasets, models);
      component.selectedDatasetIds.add('d1');
      component.selectedDetectorIds.add('m1');
      component.selectedDetectorIds.add('m2');
      expect(component.labelHint).toBe('Select exactly 1 detector');
    });

    it('should hint about media type mismatch', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'image' }];
      flushInitialRequests(datasets, models);
      expect(component.labelHint).toBe('Media type mismatch');
    });

    it('should return empty hint when label is enabled', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'audio' }];
      flushInitialRequests(datasets, models);
      expect(component.labelHint).toBe('Open Train Mode with the selected dataset and detector');
    });
  });

  describe('find hints', () => {
    it('should hint about missing dataset and model', () => {
      flushInitialRequests();
      component.selectedDatasetIds.clear();
      component.selectedDetectorIds.clear();
      expect(component.findHint).toBe('Select a dataset and a detector row above');
    });

    it('should hint about missing dataset', () => {
      const models = [{ id: 'm1', name: 'M', media_type: 'audio' }];
      flushInitialRequests([], models);
      // Single detector is auto-selected; no dataset selected.
      component.selectedDatasetIds.clear();
      expect(component.findHint).toBe('Select a dataset row in the table above');
    });

    it('should hint about missing model', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      flushInitialRequests(datasets);
      // Single dataset is auto-selected; no detector selected.
      component.selectedDetectorIds.clear();
      expect(component.findHint).toBe('Select a detector row in the table above');
    });

    it('should hint about media type mismatch', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'image' }];
      flushInitialRequests(datasets, models);
      expect(component.findHint).toBe('Media type mismatch');
    });

    it('should return score hint when find is enabled', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'audio', num_training: 5 }];
      flushInitialRequests(datasets, models);
      expect(component.findHint).toBe('Score selected datasets with selected detectors');
    });

    it('should hint about untrained model', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'audio', num_training: 0 }];
      flushInitialRequests(datasets, models);
      expect(component.findHint).toBe('Selected detector has no training labels');
    });
  });

  it('should rename a dataset', () => {
    const datasets = [{ id: 'd1', name: 'Old', media_type: 'audio' }];
    flushInitialRequests(datasets);

    component.renameDataset(datasets[0], 'New');
    const req = httpMock.expectOne('/api/datasets/registry/d1/rename');
    expect(req.request.method).toBe('PUT');
    expect(req.request.body).toEqual({ name: 'New' });
    req.flush({});

    // Refresh calls
    httpMock.expectOne('/api/datasets/registry').flush({ datasets: [] });
    httpMock.expectOne('/api/detectors/registry').flush({ detectors: [] });
  });

  it('should delete a dataset after confirmation', async () => {
    const datasets = [{ id: 'd1', name: 'ToDelete', media_type: 'audio' }];
    flushInitialRequests(datasets);
    component.selectedDatasetIds.add('d1');

    // Mock dialog confirmation
    vi.spyOn(component['dialog'], 'confirmDestructive').mockReturnValue(Promise.resolve(true));

    component.deleteDataset(datasets[0]);
    // Drain the confirm() promise continuation that issues the DELETE.
    await new Promise<void>((resolve) => setTimeout(resolve));

    const req = httpMock.expectOne('/api/datasets/registry/d1');
    expect(req.request.method).toBe('DELETE');
    req.flush({});

    expect(component.selectedDatasetIds.has('d1')).toBe(false);

    httpMock.expectOne('/api/datasets/registry').flush({ datasets: [] });
    httpMock.expectOne('/api/detectors/registry').flush({ detectors: [] });
  });

  it('should open and close importer modal via NewThingFlowsService', () => {
    flushInitialRequests();
    const flows = TestBed.inject(NewThingFlowsService);
    expect(component.importerModalOpen).toBe(false);
    component.openImporterModal();
    expect(component.importerModalOpen).toBe(true);
    flows.closeImporter();
    expect(component.importerModalOpen).toBe(false);
  });

  it('should render empty state when no datasets', () => {
    flushInitialRequests();
    // DatasetStateService is BehaviorSubject-backed, so the registry flush
    // updates the dashboard's view state through plain (non-signal) reads that
    // don't dirty the host under zoneless. markForCheck() marks it dirty so the
    // subsequent tick repaints over the now-settled state in one clean pass.
    fixture.changeDetectorRef.markForCheck();
    TestBed.tick();
    const el = fixture.nativeElement as HTMLElement;
    const empty = el.querySelector('.empty-state');
    expect(empty).toBeTruthy();
    expect(empty?.textContent || '').toContain('No datasets yet. Click + to add one.');
  });

  it('should render dataset table when datasets exist', () => {
    const datasets = [{ id: 'd1', name: 'Test', media_type: 'audio', num_items: 5 }];
    flushInitialRequests(datasets);
    fixture.changeDetectorRef.markForCheck();
    TestBed.tick();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.dash-table')).toBeTruthy();
  });

  describe('onLabel', () => {
    // Phase 2: onLabel just navigates to the URL-encoded pair; the
    // `activeContextGuard` owns the dataset/detector load and any
    // progress polling. These tests cover the navigation contract,
    // not the load orchestration (which moved to the guard +
    // ContextSwitchService).

    it('navigates to /label/:datasetId/:detectorId for the selected pair', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio', loaded: true }];
      const models = [{ id: 'm1', name: 'M', media_type: 'audio' }];
      flushInitialRequests(datasets, models);

      const routerSpy = vi.spyOn(component['router'], 'navigate').mockResolvedValue(true);
      component.onLabel();

      expect(routerSpy).toHaveBeenCalledWith(['/label', 'd1', 'm1']);
    });

    it('stores the selected model text_query in LabelSessionService before navigating', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio', loaded: true }];
      const models = [{ id: 'm1', name: 'M', media_type: 'audio', text_query: 'dog barking' }];
      flushInitialRequests(datasets, models);

      const session = TestBed.inject(LabelSessionService);
      vi.spyOn(component['router'], 'navigate').mockResolvedValue(true);
      component.onLabel();

      expect(session.textQuery).toBe('dog barking');
    });

    it('does nothing when no dataset is selected', () => {
      flushInitialRequests();
      component.selectedDatasetIds.clear();
      const routerSpy = vi.spyOn(component['router'], 'navigate').mockResolvedValue(true);
      component.onLabel();
      expect(routerSpy).not.toHaveBeenCalled();
    });

    it('opens the new-detector modal (no navigation) when no model is selected', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio', loaded: true }];
      flushInitialRequests(datasets, []);
      component.selectedDetectorIds.clear();
      const routerSpy = vi.spyOn(component['router'], 'navigate').mockResolvedValue(true);

      component.onLabel();

      expect(routerSpy).not.toHaveBeenCalled();
      expect(component.newDetectorModalOpen).toBe(true);
      expect(component.trainAfterModelCreation).toBe(true);
    });
  });

  it('should keep refreshing after an HTTP error on the registry fetch', () => {
    flushInitialRequests();

    // Phase 2: the HTTP /api/dataset/progress poller was replaced by the
    // SSE stream, so the resilient "keep going after a transient error"
    // guarantee now lives in the registry refresh pipeline
    // (DatasetStateService catchError + retry). A failed registry fetch
    // must not wedge the dashboard: a subsequent refresh still resolves.
    component.refresh();
    // forkJoin subscribes to both registry fetches at once; erroring one
    // cancels its sibling, and DatasetStateService's catchError swallows
    // the failure so the pipeline stays alive for the next refresh.
    httpMock
      .expectOne('/api/datasets/registry')
      .error(new ProgressEvent('error'), { status: 500, statusText: 'Internal Server Error' });
    for (const req of httpMock.match('/api/detectors/registry')) {
      if (!req.cancelled) {
        req.error(new ProgressEvent('error'), { status: 500, statusText: 'Internal Server Error' });
      }
    }

    // The dashboard survives and a later refresh succeeds.
    component.refresh();
    httpMock
      .expectOne('/api/datasets/registry')
      .flush({ datasets: [{ id: 'd1', name: 'Recovered', media_type: 'audio' }] });
    httpMock.expectOne('/api/detectors/registry').flush({ detectors: [] });

    expect(component.datasets.length).toBe(1);
    expect(component.datasets[0].name).toBe('Recovered');
  });

  it('should load demo dataset on demoSelected', () => {
    flushInitialRequests();
    const flows = TestBed.inject(NewThingFlowsService);
    flows.openImporter();
    expect(component.importerModalOpen).toBe(true);
    const demo = { name: 'gtzan', label: 'GTZAN' } as any;
    flows.emitDemoSelected(demo);
    flows.closeImporter();

    expect(component.importerModalOpen).toBe(false);
    // Phase 2: the demo flow POSTs /api/dataset/load-demo and hands the
    // returned task_id to the SSE-driven loading-tasks poller. Loading
    // state and progress now live on the SSE stream
    // (DashboardLoadingTasksService), not on synchronous component fields,
    // so we only assert the request contract here.
    const req = httpMock.expectOne('/api/dataset/load-demo');
    expect(req.request.method).toBe('POST');
    expect(req.request.body.name).toBe('gtzan');
    req.flush({ task_id: 't1' });
    // startProgressPolling subscribes to the SSE channel; no further HTTP
    // is issued until a loading-task event arrives, so nothing else to flush.
  });

  describe('loadDataset / loadDetector → active context', () => {
    // Loading from a dashboard card should make the item the active context
    // so the top-bar selector reflects it — but only *after* the load
    // settles (the SSE task reaches idle without error), never before, so we
    // don't reintroduce the H25 race where the interceptor tags requests
    // with an id the backend hasn't finished loading.

    it('promotes the loaded dataset to active only after the load settles', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'image' }];
      flushInitialRequests(datasets);

      const activeCtx = component['activeContext'];
      const setActive = vi.spyOn(activeCtx, 'setActivePair');
      // Capture the completion callback instead of running the real SSE poll.
      let onComplete: ((completed: LoadingTask[]) => void) | undefined;
      vi.spyOn(component['loadingTasksSvc'], 'startProgressPolling').mockImplementation(
        (_taskId?: string, cb?: (completed: LoadingTask[]) => void) => {
          onComplete = cb;
        },
      );

      component.loadDataset(datasets[0] as any);
      httpMock.expectOne('/api/datasets/registry/d1/load').flush({ task_id: 't1' });

      // Still not active mid-load.
      expect(setActive).not.toHaveBeenCalled();
      expect(onComplete).toBeTypeOf('function');

      onComplete!([]);
      expect(setActive).toHaveBeenCalledWith('d1', '');
    });

    it('promotes the loaded detector to active (preserving the dataset half) after settle', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'image', loaded: true }];
      const models = [{ id: 'm1', name: 'M', media_type: 'image' }];
      flushInitialRequests(datasets, models);

      const activeCtx = component['activeContext'];
      // A dataset is already active; loading a detector must keep it.
      activeCtx.setActivePair('d1', '');
      const setActive = vi.spyOn(activeCtx, 'setActivePair');
      let onComplete: (() => void) | undefined;
      vi.spyOn(component['loadingTasksSvc'], 'startDetectorProgressPolling').mockImplementation(
        (cb?: () => void) => {
          onComplete = cb;
        },
      );

      component.loadDetector(models[0] as any);
      httpMock.expectOne('/api/detectors/registry/load').flush({ task_id: 't2' });

      expect(setActive).not.toHaveBeenCalled();
      onComplete!();
      expect(setActive).toHaveBeenCalledWith('d1', 'm1');
    });
  });

  describe('onCombineStarted → summary toast', () => {
    it('reports unique kept vs. duplicates dropped once the combine settles', () => {
      flushInitialRequests();

      const success = vi.spyOn(component['toast'], 'success');
      let onComplete: ((completed: any[]) => void) | undefined;
      vi.spyOn(component['loadingTasksSvc'], 'startProgressPolling').mockImplementation(
        (_taskId?: string, cb?: (completed: any[]) => void) => {
          onComplete = cb;
        },
      );

      component.onCombineStarted({ taskId: 'tc', numSources: 2, totalItems: 80 });
      expect(onComplete).toBeTypeOf('function');
      // No toast until the task settles.
      expect(success).not.toHaveBeenCalled();

      // Task done: it registered dataset "dc" with 50 unique items, so 30
      // of the 80 source items were duplicates.
      onComplete!([{ task_id: 'tc', status: 'idle', dataset_id: 'dc' } as any]);
      httpMock
        .expectOne('/api/datasets/registry')
        .flush({ datasets: [{ id: 'dc', name: 'Combined', media_type: 'audio', num_items: 50 }] });

      expect(success).toHaveBeenCalledWith({
        message: 'Combined 2 datasets into 1 — 50 unique kept, 30 duplicates dropped',
      });
    });

    it('skips the toast when the completed task has no dataset id', () => {
      flushInitialRequests();

      const success = vi.spyOn(component['toast'], 'success');
      let onComplete: ((completed: any[]) => void) | undefined;
      vi.spyOn(component['loadingTasksSvc'], 'startProgressPolling').mockImplementation(
        (_taskId?: string, cb?: (completed: any[]) => void) => {
          onComplete = cb;
        },
      );

      component.onCombineStarted({ taskId: 'tc', numSources: 2, totalItems: 80 });
      onComplete!([{ task_id: 'tc', status: 'idle' } as any]);

      // No dataset id → no registry fetch and no toast.
      httpMock.expectNone('/api/datasets/registry');
      expect(success).not.toHaveBeenCalled();
    });
  });
});
