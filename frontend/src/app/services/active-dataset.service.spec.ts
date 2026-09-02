import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { ActiveContextService } from './active-context.service';
import { ActiveDatasetService } from './active-dataset.service';
import { DatasetStateService } from './dataset-state.service';
import { provideHttpTesting } from '../testing/test-providers';

describe('ActiveDatasetService', () => {
  let service: ActiveDatasetService;
  let activeContext: ActiveContextService;
  let datasetState: DatasetStateService;
  let httpMock: HttpTestingController;

  /** Land a dataset registry payload, as the app-level refresh would. */
  function flushRegistry(datasets: { id: string; name: string }[]): void {
    datasetState.refresh();
    httpMock.expectOne('/api/datasets/registry').flush({ datasets });
    httpMock.expectOne('/api/detectors/registry').flush({ detectors: [] });
  }

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [...provideHttpTesting()],
    });
    service = TestBed.inject(ActiveDatasetService);
    activeContext = TestBed.inject(ActiveContextService);
    datasetState = TestBed.inject(DatasetStateService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('reports no dataset before anything is selected', () => {
    expect(service.datasetId()).toBe('');
    expect(service.dataset()).toBeNull();
    expect(service.datasetName()).toBe('');
  });

  it('resolves the active id to its registry name', () => {
    flushRegistry([
      { id: 'ds1', name: 'Field recordings' },
      { id: 'ds2', name: 'Street noise' },
    ]);
    activeContext.setActivePair('ds2', 'd1');
    expect(service.datasetId()).toBe('ds2');
    expect(service.dataset()?.name).toBe('Street noise');
    expect(service.datasetName()).toBe('Street noise');
  });

  // The dataset half of the lifecycle gap behind issue #2819: a consumer
  // reading the entry before the registry lands used to latch null forever.
  it('fills in the name when the registry lands after the selection', () => {
    activeContext.setActivePair('ds1', '');
    expect(service.datasetName()).toBe('');

    flushRegistry([{ id: 'ds1', name: 'Field recordings' }]);
    expect(service.datasetName()).toBe('Field recordings');
  });

  it('names the user pick while the switch is still loading', () => {
    flushRegistry([{ id: 'ds1', name: 'Field recordings' }]);
    // Intent leads active for the duration of the dataset/detector load.
    activeContext.setIntent('ds1', 'd1');
    expect(service.activeId()).toBe('');
    expect(service.intentId()).toBe('ds1');
    expect(service.datasetName()).toBe('Field recordings');
  });

  it('prefers the active dataset over an in-flight switch to another one', () => {
    flushRegistry([
      { id: 'ds1', name: 'Field recordings' },
      { id: 'ds2', name: 'Street noise' },
    ]);
    activeContext.setActivePair('ds1', 'd1');
    activeContext.setIntent('ds2', 'd1');
    // Until the load promotes ds2, the loaded dataset is still ds1.
    expect(service.datasetName()).toBe('Field recordings');

    activeContext.setActive('ds2', 'd1');
    expect(service.datasetName()).toBe('Street noise');
  });

  it('clears the name when the selection is cleared', () => {
    flushRegistry([{ id: 'ds1', name: 'Field recordings' }]);
    activeContext.setActivePair('ds1', 'd1');
    activeContext.clear();
    expect(service.datasetId()).toBe('');
    expect(service.datasetName()).toBe('');
  });

  it('reports an empty name for an id the registry does not know', () => {
    flushRegistry([{ id: 'ds1', name: 'Field recordings' }]);
    activeContext.setActivePair('gone', 'd1');
    expect(service.dataset()).toBeNull();
    expect(service.datasetName()).toBe('');
  });
});
