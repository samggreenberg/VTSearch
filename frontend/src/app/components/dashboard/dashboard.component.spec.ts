import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { DashboardComponent } from './dashboard.component';
import { LabelSessionService } from '../../services/label-session.service';

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
    models: any[] = [],
  ): void {
    fixture.detectChanges();
    httpMock.expectOne('/api/datasets/registry').flush({ datasets });
    httpMock.expectOne('/api/models/registry').flush({ models });
  }

  it('should create', () => {
    flushInitialRequests();
    expect(component).toBeTruthy();
  });

  it('should fetch datasets and models on init', () => {
    const datasets = [{ id: 'd1', name: 'Test Dataset', media_type: 'audio', num_items: 10 }];
    const models = [{ id: 'm1', name: 'Test Model', trainable: true }];
    flushInitialRequests(datasets, models);
    expect(component.datasets.length).toBe(1);
    expect(component.models.length).toBe(1);
  });

  it('should auto-select single dataset', () => {
    const datasets = [{ id: 'd1', name: 'Only One' }];
    flushInitialRequests(datasets);
    expect(component.selectedDatasetIds.has('d1')).toBeTrue();
  });

  it('should auto-select single model', () => {
    const models = [{ id: 'm1', name: 'Only One' }];
    flushInitialRequests([], models);
    expect(component.selectedModelIds.has('m1')).toBeTrue();
  });

  it('should not auto-select when multiple datasets', () => {
    const datasets = [
      { id: 'd1', name: 'First' },
      { id: 'd2', name: 'Second' },
    ];
    flushInitialRequests(datasets);
    expect(component.selectedDatasetIds.size).toBe(0);
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

    component.sortDatasets('name');
    expect(component.sortedDatasets[0].name).toBe('Alpha');

    component.sortDatasets('name');
    expect(component.sortedDatasets[0].name).toBe('Bravo');
  });

  it('should sort models by column', () => {
    const models = [
      { id: 'm1', name: 'Zeta', num_training: 5 },
      { id: 'm2', name: 'Alpha', num_training: 10 },
    ];
    flushInitialRequests([], models);

    component.sortModels('name');
    expect(component.sortedModels[0].name).toBe('Alpha');
  });

  it('should show sort indicators', () => {
    flushInitialRequests();
    component.sortDatasets('name');
    expect(component.datasetSortIndicator('name')).toContain('\u25B2');
    expect(component.isDatasetSortActive('name')).toBeTrue();
    component.sortDatasets('name');
    expect(component.datasetSortIndicator('name')).toContain('\u25BC');
    expect(component.isDatasetSortActive('other')).toBeFalse();
  });

  describe('button state', () => {
    it('should disable Label when nothing selected', () => {
      flushInitialRequests();
      component.selectedDatasetIds.clear();
      component.selectedModelIds.clear();
      expect(component.labelEnabled).toBeFalse();
    });

    it('should enable Label with 1 dataset + 1 trainable model', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', trainable: true, media_type: 'audio' }];
      flushInitialRequests(datasets, models);
      // Auto-selected since only 1 each
      expect(component.labelEnabled).toBeTrue();
    });

    it('should disable Label when model is not trainable', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', trainable: false, media_type: 'audio' }];
      flushInitialRequests(datasets, models);
      expect(component.labelEnabled).toBeFalse();
    });

    it('should disable Label on media type mismatch', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', trainable: true, media_type: 'image' }];
      flushInitialRequests(datasets, models);
      expect(component.labelEnabled).toBeFalse();
    });

    it('should disable Label when model media_type is "any" (not a valid type)', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', trainable: true, media_type: 'any' }];
      flushInitialRequests(datasets, models);
      expect(component.labelEnabled).toBeFalse();
    });

    it('should disable Find with no selections', () => {
      flushInitialRequests();
      component.selectedDatasetIds.clear();
      component.selectedModelIds.clear();
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
      component.selectedModelIds.add('m2');
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
      component.selectedModelIds.clear();
      expect(component.labelHint).toBe('Select a model');
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
      component.selectedModelIds.add('m1');
      component.selectedModelIds.add('m2');
      expect(component.labelHint).toBe('Select exactly 1 model');
    });

    it('should hint about non-trainable model', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', trainable: false, media_type: 'audio' }];
      flushInitialRequests(datasets, models);
      expect(component.labelHint).toBe('Model is not trainable');
    });

    it('should hint about media type mismatch', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', trainable: true, media_type: 'image' }];
      flushInitialRequests(datasets, models);
      expect(component.labelHint).toBe('Media type mismatch');
    });

    it('should return empty hint when label is enabled', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio' }];
      const models = [{ id: 'm1', name: 'M', trainable: true, media_type: 'audio' }];
      flushInitialRequests(datasets, models);
      expect(component.labelHint).toBe('');
    });
  });

  describe('find hints', () => {
    it('should hint about missing dataset and model', () => {
      flushInitialRequests();
      component.selectedDatasetIds.clear();
      component.selectedModelIds.clear();
      expect(component.findHint).toBe('Select a dataset and a model');
    });

    it('should hint about missing dataset', () => {
      flushInitialRequests();
      component.selectedDatasetIds.clear();
      component.selectedModelIds.add('m1');
      expect(component.findHint).toBe('Select a dataset');
    });

    it('should hint about missing model', () => {
      flushInitialRequests();
      component.selectedDatasetIds.add('d1');
      component.selectedModelIds.clear();
      expect(component.findHint).toBe('Select a model');
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
      expect(component.findHint).toBe('Score selected datasets with selected models');
    });
  });

  it('should rename a dataset', () => {
    const datasets = [{ id: 'd1', name: 'Old' }];
    flushInitialRequests(datasets);

    component.renameDataset(datasets[0], 'New');
    const req = httpMock.expectOne('/api/datasets/registry/d1/rename');
    expect(req.request.method).toBe('PUT');
    expect(req.request.body).toEqual({ name: 'New' });
    req.flush({});

    // Refresh calls
    httpMock.expectOne('/api/datasets/registry').flush({ datasets: [] });
    httpMock.expectOne('/api/models/registry').flush({ models: [] });
  });

  it('should delete a dataset after confirmation', fakeAsync(() => {
    const datasets = [{ id: 'd1', name: 'ToDelete' }];
    flushInitialRequests(datasets);
    component.selectedDatasetIds.add('d1');

    // Mock dialog confirmation
    spyOn(component['dialog'], 'confirm').and.returnValue(Promise.resolve(true));

    component.deleteDataset(datasets[0]);
    tick();

    const req = httpMock.expectOne('/api/datasets/registry/d1');
    expect(req.request.method).toBe('DELETE');
    req.flush({});

    expect(component.selectedDatasetIds.has('d1')).toBeFalse();

    httpMock.expectOne('/api/datasets/registry').flush({ datasets: [] });
    httpMock.expectOne('/api/models/registry').flush({ models: [] });
  }));

  it('should open and close importer modal', () => {
    flushInitialRequests();
    expect(component.importerModalOpen).toBeFalse();
    component.openImporterModal();
    expect(component.importerModalOpen).toBeTrue();
    component.closeImporterModal();
    expect(component.importerModalOpen).toBeFalse();
  });

  it('should render empty state when no datasets', () => {
    flushInitialRequests();
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.empty-state')?.textContent).toContain('No datasets yet');
  });

  it('should render dataset table when datasets exist', () => {
    const datasets = [{ id: 'd1', name: 'Test', media_type: 'audio', num_items: 5 }];
    flushInitialRequests(datasets);
    fixture.detectChanges();
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('.dash-table')).toBeTruthy();
  });

  describe('onLabel', () => {
    it('should navigate directly when dataset is already loaded', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio', loaded: true }];
      const models = [{ id: 'm1', name: 'M', trainable: true, media_type: 'audio' }];
      flushInitialRequests(datasets, models);

      const routerSpy = spyOn(component['router'], 'navigate');
      component.onLabel();

      // Flush the loadModel call
      const loadModelReq = httpMock.expectOne('/api/models/registry/load');
      expect(loadModelReq.request.method).toBe('POST');
      expect(loadModelReq.request.body).toEqual({ model_id: 'm1' });
      loadModelReq.flush({ ok: true });

      expect(routerSpy).toHaveBeenCalledWith(['/label']);
      expect(component.loading).toBeFalse();
    });

    it('should store selected model text_query in session before navigating', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio', loaded: true }];
      const models = [{ id: 'm1', name: 'M', trainable: true, media_type: 'audio', text_query: 'dog barking' }];
      flushInitialRequests(datasets, models);

      const session = TestBed.inject(LabelSessionService);
      spyOn(component['router'], 'navigate');
      component.onLabel();

      const loadModelReq = httpMock.expectOne('/api/models/registry/load');
      loadModelReq.flush({ ok: true });

      expect(session.textQuery).toBe('dog barking');
    });

    it('should load dataset first when not loaded, then navigate', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio', loaded: false }];
      const models = [{ id: 'm1', name: 'M', trainable: true, media_type: 'audio' }];
      flushInitialRequests(datasets, models);

      const routerSpy = spyOn(component['router'], 'navigate');
      component.onLabel();

      expect(component.loading).toBeTrue();
      expect(component.progressMessage).toContain('DS');
      expect(routerSpy).not.toHaveBeenCalled();

      // Respond to load request
      const loadReq = httpMock.expectOne('/api/datasets/registry/d1/load');
      expect(loadReq.request.method).toBe('POST');
      loadReq.flush({ ok: true });

      // Progress polling returns idle -> should navigate
      const progressReq = httpMock.expectOne('/api/dataset/progress');
      progressReq.flush({ status: 'idle' });

      // Flush the loadModel call triggered after progress completes
      const loadModelReq = httpMock.expectOne('/api/models/registry/load');
      loadModelReq.flush({ ok: true });

      expect(routerSpy).toHaveBeenCalledWith(['/label']);

      // Refresh after completion
      httpMock.expectOne('/api/datasets/registry').flush({ datasets: [] });
      httpMock.expectOne('/api/models/registry').flush({ models: [] });
    });

    it('should not navigate on load error progress', () => {
      const datasets = [{ id: 'd1', name: 'DS', media_type: 'audio', loaded: false }];
      const models = [{ id: 'm1', name: 'M', trainable: true, media_type: 'audio' }];
      flushInitialRequests(datasets, models);

      const routerSpy = spyOn(component['router'], 'navigate');
      component.onLabel();

      const loadReq = httpMock.expectOne('/api/datasets/registry/d1/load');
      loadReq.flush({ ok: true });

      // Progress polling returns error
      const progressReq = httpMock.expectOne('/api/dataset/progress');
      progressReq.flush({ status: 'error', message: 'Out of memory' });

      expect(routerSpy).not.toHaveBeenCalled();

      // Refresh after error
      httpMock.expectOne('/api/datasets/registry').flush({ datasets: [] });
      httpMock.expectOne('/api/models/registry').flush({ models: [] });
    });

    it('should do nothing when no dataset is selected', () => {
      flushInitialRequests();
      component.selectedDatasetIds.clear();
      const routerSpy = spyOn(component['router'], 'navigate');
      component.onLabel();
      expect(routerSpy).not.toHaveBeenCalled();
    });
  });

  it('should load demo dataset on demoSelected', () => {
    flushInitialRequests();
    component.importerModalOpen = true;
    const demo = { name: 'gtzan', label: 'GTZAN' };
    component.onDemoSelected(demo);

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
    httpMock.expectOne('/api/models/registry').flush({ models: [] });
  });
});
