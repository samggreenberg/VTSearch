import { TestBed, fakeAsync, tick } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { DatasetStateService } from './dataset-state.service';

describe('DatasetStateService', () => {
  let service: DatasetStateService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(DatasetStateService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should start with empty state', () => {
    expect(service.datasets).toEqual([]);
    expect(service.models).toEqual([]);
    expect(service.loading).toBeFalse();
    expect(service.progressMessage).toBe('');
  });

  it('refresh should fetch datasets and models', () => {
    service.refresh();

    const datasetsReq = httpMock.expectOne('/api/datasets/registry');
    const modelsReq = httpMock.expectOne('/api/detectors/registry');

    datasetsReq.flush({ datasets: [{ id: '1', name: 'test' }] });
    modelsReq.flush({ models: [{ id: 'm1', name: 'model1' }] });

    expect(service.datasets.length).toBe(1);
    expect(service.datasets[0].name).toBe('test');
    expect(service.models.length).toBe(1);
    expect(service.models[0].name).toBe('model1');
  });

  it('rapid refresh should cancel stale in-flight requests', fakeAsync(() => {
    // First refresh — will be cancelled by the second
    service.refresh();
    tick();
    const staleDs = httpMock.expectOne('/api/datasets/registry');
    const staleModels = httpMock.expectOne('/api/detectors/registry');

    // Second refresh — this one should win
    service.refresh();
    tick();
    const freshDs = httpMock.expectOne('/api/datasets/registry');
    const freshModels = httpMock.expectOne('/api/detectors/registry');

    // The first (stale) requests were cancelled by switchMap
    expect(staleDs.cancelled).toBeTrue();
    expect(staleModels.cancelled).toBeTrue();

    // Flush the fresh responses
    freshDs.flush({ datasets: [{ id: '1', name: 'fresh' }] });
    freshModels.flush({ models: [] });

    expect(service.datasets.length).toBe(1);
    expect(service.datasets[0].name).toBe('fresh');
  }));

  it('setLoading and setProgressMessage should update state', () => {
    service.setLoading(true);
    service.setProgressMessage('Loading...');

    expect(service.loading).toBeTrue();
    expect(service.progressMessage).toBe('Loading...');
  });

  it('clear should reset all state', () => {
    service.setLoading(true);
    service.setProgressMessage('test');
    service.clear();

    expect(service.datasets).toEqual([]);
    expect(service.models).toEqual([]);
    expect(service.loading).toBeFalse();
    expect(service.progressMessage).toBe('');
  });
});
