import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { DashboardComponent } from './dashboard.component';
import { LabelSessionService } from '../../services/label-session.service';
import { NewThingFlowsService } from '../../services/new-thing-flows.service';

describe('DashboardComponent', () => {
  let component: DashboardComponent;
  let fixture: ComponentFixture<DashboardComponent>;
  let httpMock: HttpTestingController;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DashboardComponent],
      providers: [provideHttpClient(), provideHttpClientTesting(), provideRouter([])],
    }).compileComponents();

    fixture = TestBed.createComponent(DashboardComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  function flushInitialRequests(
    datasets: any[] = [],
    detectors: any[] = [],
    importers: any[] = [],
  ): void {
    fixture.detectChanges();
    httpMock.expectOne('/api/datasets/registry').flush({ datasets });
    httpMock.expectOne('/api/detectors/registry').flush({ detectors });
    httpMock.expectOne('/api/dataset/all-importers').flush({ importers, tabs: [] });
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
    expect(component.selectedDatasetIds.has('d1')).toBeTrue();
  });

  it('should auto-select single model', () => {
    const models = [{ id: 'm1', name: 'Only One' }];
    flushInitialRequests([], models);
    expect(component.selectedDetectorIds.has('m1')).toBeTrue();
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
    expect(component.selectedDatasetIds.has('d1')).toBeTrue();

    // Simulate adding a second dataset via refresh
    component.refresh();
    httpMock.expectOne('/api/datasets/registry').flush({
      datasets: [
        { id: 'd1', name: 'First' },
        { id: 'd2', name: 'Second' },
      ],
    });
    httpMock.expectOne('/api/detectors/registry').flush({ detectors: [] });

    expect(component.selectedDatasetIds.has('d1')).toBeFalse();
    expect(component.selectedDatasetIds.has('d2')).toBeTrue();
  });

  it('should auto-select newly added model', () => {
    const models = [{ id: 'm1', name: 'First' }];
    flushInitialRequests([], models);
    expect(component.selectedDetectorIds.has('m1')).toBeTrue();

    // Simulate adding a second model via refresh
    component.refresh();
    httpMock.expectOne('/api/datasets/registry').flush({ datasets: [] });
    httpMock.expectOne('/api/detectors/registry').flush({
      detectors: [
        { id: 'm1', name: 'First' },
        { id: 'm2', name: 'Second' },
      ],
    });

    expect(component.selectedDetectorIds.has('m1')).toBeTrue();
    expect(component.selectedDetectorIds.has('m2')).toBeTrue();
  });

  it('should toggle dataset selection on click', () => {
    flushInitialRequests();
    const event = new MouseEvent('click');
    component.toggleDatasetSelection('d1', event);
    expect(component.isDatasetSelected('d1')).toBeTrue();
    component.toggleDatasetSelection('d1', event);
    expect(component.isDatasetSelected('d1')).toBeFalse();
  });

  it('should support multi-select with ctrl key', () => {
    flushInitialRequests();
    const ctrlEvent = new MouseEvent('click', { ctrlKey: true });
    component.toggleDatasetSelection('d1', new MouseEvent('click'));
    component.toggleDatasetSelection('d2', ctrlEvent);
    expect(component.isDatasetSelected('d1')).toBeTrue();
    expect(component.isDatasetSelected('d2')).toBeTrue();
  });

  it('should replace selection without ctrl key', () => {
    flushInitialRequests();
    component.toggleDatasetSelection('d1', new MouseEvent('click'));
    component.toggleDatasetSelection('d2', new MouseEvent('click'));
    expect(component.isDatasetSelected('d1')).toBeFalse();
    expect(component.isDatasetSelected('d2')).toBeTrue();
  });

  it('should sort datasets by column', () => {
    const datasets = [
      { id: 'd1', name: 'Bravo', num_items: 5 },
      { id: 'd2', name: 'Alpha', num_items: 10 },
    ];
    flushInitialRequests(datasets);

    component.datasetCols.sortBy('name');
    expect(component.sortedDatasets[0].name).toBe('Alpha');

    component.datasetCols.sortBy('name');
    expect(component.sortedDatasets[0].name).toBe('Bravo');
  });

  it('should sort models by column', () => {
    const models = [
      { id: 'm1', name: 'Zeta', num_training: 5 },
      { id: 'm2', name: 'Alpha', num_training: 10 },
    ];
    flushInitialRequests([], models);

    component.detectorCols.sortBy('name');
    expect(component.sortedDetectors[0].name).toBe('Alpha');
  });

  it('should show sort indicators', () => {
    flushInitialRequests();
    component.datasetCols.sortBy('name');
    expect(component.datasetCols.sortIndicator('name')).toContain('\u25B2');
    expect(component.datasetCols.isSortActive('name')).toBeTrue();
    component.datasetCols.sortBy('name');
    expect(component.datasetCols.sortIndicator('name')).toContain('\u25BC');
    expect(component.datasetCols.isSortActive('other')).toBeFalse();
  });

  describe('button state', () => {
    it('should disable Label when nothing selected', () => {
      flushInitialRequests();
      component.selectedDatasetIds.clear();
      component.selectedDetectorIds.clear();
      expect(component.labelEnabled).toBeFalse();
    });

    it('should enable Label with 1 dataset + 1 model', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'audio' }];
      flushInitialRequests(datasets, models);
      // Auto-selected since only 1 each
      expect(component.labelEnabled).toBeTrue();
    });

    it('should disable Label on media type mismatch', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'image' }];
      flushInitialRequests(datasets, models);
      expect(component.labelEnabled).toBeFalse();
    });

    it('should disable Label when model media_type is "any" (not a valid type)', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'any' }];
      flushInitialRequests(datasets, models);
      expect(component.labelEnabled).toBeFalse();
    });

    it('should disable Find with no selections', () => {
      flushInitialRequests();
      component.selectedDatasetIds.clear();
      component.selectedDetectorIds.clear();
      expect(component.findEnabled).toBeFalse();
    });

    it('should enable Find with matching media types', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'audio' }];
      flushInitialRequests(datasets, models);
      expect(component.findEnabled).toBeTrue();
    });

    it('should disable Find on media type mismatch', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'image' }];
      flushInitialRequests(datasets, models);
      expect(component.findEnabled).toBeFalse();
    });

    it('should disable Find when multiple datasets have different media types', () => {
      const datasets = [
        { id: 'd1', name: 'DS1', media_type: 'audio' },
        { id: 'd2', name: 'DS2', media_type: 'image' },
      ];
      const models = [{ id: 'm1', name: 'M', media_type: 'audio' }];
      flushInitialRequests(datasets, models);
      component.selectedDatasetIds.add('d2');
      expect(component.findEnabled).toBeFalse();
    });

    it('should enable Find when all selected items share media type', () => {
      const datasets = [
        { id: 'd1', name: 'DS1', media_type: 'image' },
        { id: 'd2', name: 'DS2', media_type: 'image' },
      ];
      const models = [
        { id: 'm1', name: 'M1', media_type: 'image' },
        { id: 'm2', name: 'M2', media_type: 'image' },
      ];
      flushInitialRequests(datasets, models);
      component.selectedDatasetIds.add('d2');
      component.selectedDetectorIds.add('m2');
      expect(component.findEnabled).toBeTrue();
    });

    it('should disable Find when the selected model has 0 training', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'audio', num_training: 0 }];
      flushInitialRequests(datasets, models);
      expect(component.findEnabled).toBeFalse();
    });

    it('should enable Find for a model with training', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'audio', num_training: 5 }];
      flushInitialRequests(datasets, models);
      expect(component.findEnabled).toBeTrue();
    });
  });

  describe('label hints', () => {
    it('should hint about missing dataset', () => {
      flushInitialRequests();
      component.selectedDatasetIds.clear();
      expect(component.labelHint).toBe('Select a dataset');
    });

    it('should hint about missing model', () => {
      flushInitialRequests();
      component.selectedDatasetIds.add('d1');
      component.selectedDetectorIds.clear();
      expect(component.labelHint).toBe('Select a detector');
    });

    it('should hint about multiple datasets', () => {
      flushInitialRequests();
      component.selectedDatasetIds.add('d1');
      component.selectedDatasetIds.add('d2');
      expect(component.labelHint).toBe('Select exactly 1 dataset');
    });

    it('should hint about multiple models', () => {
      flushInitialRequests();
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
      expect(component.findHint).toBe('Select a dataset and a detector');
    });

    it('should hint about missing dataset', () => {
      flushInitialRequests();
      component.selectedDatasetIds.clear();
      component.selectedDetectorIds.add('m1');
      expect(component.findHint).toBe('Select a dataset');
    });

    it('should hint about missing model', () => {
      flushInitialRequests();
      component.selectedDatasetIds.add('d1');
      component.selectedDetectorIds.clear();
      expect(component.findHint).toBe('Select a detector');
    });

    it('should hint about media type mismatch', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'image' }];
      flushInitialRequests(datasets, models);
      expect(component.findHint).toBe('Media type mismatch');
    });

    it('should return score hint when find is enabled', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', media_type: 'audio' }];
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

  it('should delete a dataset after confirmation', fakeAsync(() => {
    const datasets = [{ id: 'd1', name: 'ToDelete', media_type: 'audio' }];
    flushInitialRequests(datasets);
    component.selectedDatasetIds.add('d1');

    // Mock dialog confirmation
    spyOn(component['dialog'], 'confirmDestructive').and.returnValue(Promise.resolve(true));

    component.deleteDataset(datasets[0]);
    tick();

    const req = httpMock.expectOne('/api/datasets/registry/d1');
    expect(req.request.method).toBe('DELETE');
    req.flush({});

    expect(component.selectedDatasetIds.has('d1')).toBeFalse();

    httpMock.expectOne('/api/datasets/registry').flush({ datasets: [] });
    httpMock.expectOne('/api/detectors/registry').flush({ detectors: [] });
  }));

  it('should open and close importer modal via NewThingFlowsService', () => {
    flushInitialRequests();
    const flows = TestBed.inject(NewThingFlowsService);
    expect(component.importerModalOpen).toBeFalse();
    component.openImporterModal();
    expect(component.importerModalOpen).toBeTrue();
    flows.closeImporter();
    expect(component.importerModalOpen).toBeFalse();
  });

  it('should render empty state when no datasets', () => {
    flushInitialRequests();
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    const empty = el.querySelector('.empty-state');
    expect(empty).toBeTruthy();
    expect(empty?.textContent || '').toContain('No datasets yet. Click + to add one.');
  });

  it('should render dataset table when datasets exist', () => {
    const datasets = [{ id: 'd1', name: 'Test', media_type: 'audio', num_items: 5 }];
    flushInitialRequests(datasets);
    fixture.detectChanges();
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

      const routerSpy = spyOn(component['router'], 'navigate');
      component.onLabel();

      expect(routerSpy).toHaveBeenCalledWith(['/label', 'd1', 'm1']);
    });

    it('stores the selected model text_query in LabelSessionService before navigating', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio', loaded: true }];
      const models = [{ id: 'm1', name: 'M', media_type: 'audio', text_query: 'dog barking' }];
      flushInitialRequests(datasets, models);

      const session = TestBed.inject(LabelSessionService);
      spyOn(component['router'], 'navigate');
      component.onLabel();

      expect(session.textQuery).toBe('dog barking');
    });

    it('does nothing when no dataset is selected', () => {
      flushInitialRequests();
      component.selectedDatasetIds.clear();
      const routerSpy = spyOn(component['router'], 'navigate');
      component.onLabel();
      expect(routerSpy).not.toHaveBeenCalled();
    });

    it('opens the new-detector modal (no navigation) when no model is selected', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio', loaded: true }];
      flushInitialRequests(datasets, []);
      component.selectedDetectorIds.clear();
      const routerSpy = spyOn(component['router'], 'navigate');

      component.onLabel();

      expect(routerSpy).not.toHaveBeenCalled();
      expect(component.newDetectorModalOpen).toBeTrue();
      expect(component.trainAfterModelCreation).toBeTrue();
    });
  });

  it('should continue polling after HTTP error on progress endpoint', fakeAsync(() => {
    flushInitialRequests();
    const flows = TestBed.inject(NewThingFlowsService);
    flows.emitDemoSelected({ name: 'gtzan', label: 'GTZAN' } as any);

    const demoReq = httpMock.expectOne('/api/dataset/load-demo');
    demoReq.flush({});

    // First progress poll fails with a server error
    const failedReq = httpMock.expectOne('/api/dataset/progress');
    failedReq.error(new ProgressEvent('error'), { status: 500, statusText: 'Internal Server Error' });

    // Component should still be in loading state
    expect(component.loading).toBeTrue();

    // Advance timer to trigger next poll - polling should survive the error
    tick(1000);
    const retryReq = httpMock.expectOne('/api/dataset/progress');
    retryReq.flush({ status: 'idle' });

    expect(component.loading).toBeFalse();

    // Refresh after completion
    httpMock.expectOne('/api/datasets/registry').flush({ datasets: [] });
    httpMock.expectOne('/api/detectors/registry').flush({ detectors: [] });
  }));

  it('should load demo dataset on demoSelected', () => {
    flushInitialRequests();
    const flows = TestBed.inject(NewThingFlowsService);
    flows.openImporter();
    expect(component.importerModalOpen).toBeTrue();
    const demo = { name: 'gtzan', label: 'GTZAN' } as any;
    flows.emitDemoSelected(demo);
    flows.closeImporter();

    expect(component.importerModalOpen).toBeFalse();
    expect(component.loading).toBeTrue();
    expect(component.progressMessage).toContain('GTZAN');

    const req = httpMock.expectOne('/api/dataset/load-demo');
    expect(req.request.method).toBe('POST');
    expect(req.request.body.name).toBe('gtzan');
    req.flush({});

    // Progress polling starts
    const progressReq = httpMock.expectOne('/api/dataset/progress');
    progressReq.flush({ status: 'idle' });

    // After idle, it should refresh
    httpMock.expectOne('/api/datasets/registry').flush({ datasets: [] });
    httpMock.expectOne('/api/detectors/registry').flush({ detectors: [] });
  });
});
