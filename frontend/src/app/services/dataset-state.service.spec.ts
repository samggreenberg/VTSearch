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
    expect(service.detectors).toEqual([]);
    expect(service.loading).toBeFalse();
    expect(service.progressMessage).toBe('');
    expect(service.loaded).toBeFalse();
  });

  it('flips loaded$ once the first registry fetch returns (success)', () => {
    expect(service.loaded).toBeFalse();
    service.refresh();
    httpMock.expectOne('/api/datasets/registry').flush({ datasets: [] });
    httpMock.expectOne('/api/detectors/registry').flush({ detectors: [] });
    expect(service.loaded).toBeTrue();
  });

  it('flips loaded$ once the first registry fetch returns (error)', () => {
    expect(service.loaded).toBeFalse();
    service.refresh();
    httpMock.expectOne('/api/datasets/registry').error(new ProgressEvent('Network'));
    httpMock.expectOne('/api/detectors/registry').error(new ProgressEvent('Network'));
    expect(service.loaded).toBeTrue();
  });

  it('refresh should fetch datasets and detectors', () => {
    service.refresh();

    const datasetsReq = httpMock.expectOne('/api/datasets/registry');
    const detectorsReq = httpMock.expectOne('/api/detectors/registry');

    datasetsReq.flush({ datasets: [{ id: '1', name: 'test' }] });
    detectorsReq.flush({ detectors: [{ id: 'm1', name: 'detector1' }] });

    expect(service.datasets.length).toBe(1);
    expect(service.datasets[0].name).toBe('test');
    expect(service.detectors.length).toBe(1);
    expect(service.detectors[0].name).toBe('detector1');
  });

  it('rapid refresh should cancel stale in-flight requests', fakeAsync(() => {
    // First refresh - will be cancelled by the second
    service.refresh();
    tick();
    const staleDs = httpMock.expectOne('/api/datasets/registry');
    const staleDetectors = httpMock.expectOne('/api/detectors/registry');

    // Second refresh - this one should win
    service.refresh();
    tick();
    const freshDs = httpMock.expectOne('/api/datasets/registry');
    const freshDetectors = httpMock.expectOne('/api/detectors/registry');

    // The first (stale) requests were cancelled by switchMap
    expect(staleDs.cancelled).toBeTrue();
    expect(staleDetectors.cancelled).toBeTrue();

    // Flush the fresh responses
    freshDs.flush({ datasets: [{ id: '1', name: 'fresh' }] });
    freshDetectors.flush({ detectors: [] });

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
    expect(service.detectors).toEqual([]);
    expect(service.loading).toBeFalse();
    expect(service.progressMessage).toBe('');
    expect(service.error).toBeNull();
  });

  it('refresh should record error on registry fetch failure and clear on retry success', () => {
    const errors: (string | null)[] = [];
    service.error$.subscribe((e) => errors.push(e));

    service.refresh();
    httpMock.expectOne('/api/datasets/registry').error(new ProgressEvent('Network error'));
    httpMock.expectOne('/api/detectors/registry').error(new ProgressEvent('Network error'));
    expect(service.error).toBe("Couldn't load datasets and detectors.");

    // A second refresh after the failure should still work - the
    // pipeline survives errors via catchError().
    service.refresh();
    httpMock.expectOne('/api/datasets/registry').flush({ datasets: [{ id: '1', name: 'ok' }] });
    httpMock.expectOne('/api/detectors/registry').flush({ detectors: [] });
    expect(service.datasets.length).toBe(1);
    expect(service.error).toBeNull();
    // Sanity: we saw both error and success emissions.
    expect(errors).toContain("Couldn't load datasets and detectors.");
    expect(errors[errors.length - 1]).toBeNull();
  });
});
