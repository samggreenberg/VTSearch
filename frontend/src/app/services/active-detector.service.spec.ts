import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { ActiveContextService } from './active-context.service';
import { ActiveDetectorService } from './active-detector.service';
import { DatasetStateService } from './dataset-state.service';
import { provideHttpTesting } from '../testing/test-providers';

describe('ActiveDetectorService', () => {
  let service: ActiveDetectorService;
  let activeContext: ActiveContextService;
  let datasetState: DatasetStateService;
  let httpMock: HttpTestingController;

  /** Land a detector registry payload, as the app-level refresh would. */
  function flushRegistry(detectors: { id: string; name: string }[]): void {
    datasetState.refresh();
    httpMock.expectOne('/api/datasets/registry').flush({ datasets: [] });
    httpMock.expectOne('/api/detectors/registry').flush({ detectors });
  }

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [...provideHttpTesting()],
    });
    service = TestBed.inject(ActiveDetectorService);
    activeContext = TestBed.inject(ActiveContextService);
    datasetState = TestBed.inject(DatasetStateService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('reports no detector before anything is selected', () => {
    expect(service.detectorId()).toBe('');
    expect(service.detector()).toBeNull();
    expect(service.detectorName()).toBe('');
  });

  it('resolves the active id to its registry name', () => {
    flushRegistry([
      { id: 'd1', name: 'Birdsong' },
      { id: 'd2', name: 'Sirens' },
    ]);
    activeContext.setActivePair('ds1', 'd2');
    expect(service.detectorId()).toBe('d2');
    expect(service.detector()?.name).toBe('Sirens');
    expect(service.detectorName()).toBe('Sirens');
  });

  // The lifecycle gap behind issue #2819: a consumer reading the name before
  // the registry lands used to latch '' forever. As a signal it fills in.
  it('fills in the name when the registry lands after the selection', () => {
    activeContext.setActivePair('ds1', 'd1');
    expect(service.detectorName()).toBe('');

    flushRegistry([{ id: 'd1', name: 'Birdsong' }]);
    expect(service.detectorName()).toBe('Birdsong');
  });

  it('names the user pick while the switch is still loading', () => {
    flushRegistry([{ id: 'd1', name: 'Birdsong' }]);
    // Intent leads active for the duration of the dataset/detector load.
    activeContext.setIntent('ds1', 'd1');
    expect(service.activeId()).toBe('');
    expect(service.intentId()).toBe('d1');
    expect(service.detectorName()).toBe('Birdsong');
  });

  it('prefers the active detector over an in-flight switch to another one', () => {
    flushRegistry([
      { id: 'd1', name: 'Birdsong' },
      { id: 'd2', name: 'Sirens' },
    ]);
    activeContext.setActivePair('ds1', 'd1');
    activeContext.setIntent('ds1', 'd2');
    // Until the load promotes d2, the loaded detector is still d1.
    expect(service.detectorName()).toBe('Birdsong');

    activeContext.setActive('ds1', 'd2');
    expect(service.detectorName()).toBe('Sirens');
  });

  it('clears the name when the selection is cleared', () => {
    flushRegistry([{ id: 'd1', name: 'Birdsong' }]);
    activeContext.setActivePair('ds1', 'd1');
    activeContext.clear();
    expect(service.detectorId()).toBe('');
    expect(service.detectorName()).toBe('');
  });

  it('reports an empty name for an id the registry does not know', () => {
    flushRegistry([{ id: 'd1', name: 'Birdsong' }]);
    activeContext.setActivePair('ds1', 'gone');
    expect(service.detector()).toBeNull();
    expect(service.detectorName()).toBe('');
  });
});
