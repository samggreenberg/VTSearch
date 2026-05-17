import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { ActiveContextWatcherService } from './active-context-watcher.service';
import { ActiveContextService } from './active-context.service';
import { DatasetStateService } from './dataset-state.service';
import { ToastService } from './toast.service';
import { DatasetRegistryEntry, DetectorRegistryEntry } from '../models/api.models';

describe('ActiveContextWatcherService', () => {
  let watcher: ActiveContextWatcherService;
  let activeContext: ActiveContextService;
  let datasetState: DatasetStateService;
  let toast: ToastService;

  function makeDataset(id: string, name: string): DatasetRegistryEntry {
    return { id, name, media_type: 'audio' } as DatasetRegistryEntry;
  }

  function makeDetector(id: string, name: string): DetectorRegistryEntry {
    return { id, name, media_type: 'audio' } as DetectorRegistryEntry;
  }

  function setDatasets(entries: DatasetRegistryEntry[]): void {
    // BehaviorSubject is private; cast through any for test setup.
    (datasetState as unknown as { datasetsSubject: { next: (v: unknown) => void } })
      .datasetsSubject.next(entries);
  }

  function setDetectors(entries: DetectorRegistryEntry[]): void {
    (datasetState as unknown as { detectorsSubject: { next: (v: unknown) => void } })
      .detectorsSubject.next(entries);
  }

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting(), provideRouter([])],
    });
    watcher = TestBed.inject(ActiveContextWatcherService);
    activeContext = TestBed.inject(ActiveContextService);
    datasetState = TestBed.inject(DatasetStateService);
    toast = TestBed.inject(ToastService);
    watcher.start();
  });

  it('does not toast when the registry is empty (initial-load state)', () => {
    activeContext.setActivePair('d1', 'm1');
    // Registry still empty — could just be loading.
    expect(toast.toasts.length).toBe(0);
    expect(activeContext.datasetId).toBe('d1');
    expect(activeContext.modelId).toBe('m1');
  });

  it('toasts and clears the dataset half when the active dataset disappears', () => {
    setDatasets([makeDataset('d1', 'Audio Foo')]);
    setDetectors([makeDetector('m1', 'Det')]);
    activeContext.setActivePair('d1', 'm1');
    expect(toast.toasts.length).toBe(0);

    // d1 deleted from another tab / session.
    setDatasets([]);
    setDetectors([makeDetector('m1', 'Det')]);

    expect(activeContext.datasetId).toBe('');
    expect(activeContext.modelId).toBe('m1');
    expect(toast.toasts.length).toBe(1);
    expect(toast.toasts[0].message).toContain('Audio Foo');
  });

  it('toasts and clears the detector half when the active detector disappears', () => {
    setDatasets([makeDataset('d1', 'DS')]);
    setDetectors([makeDetector('m1', 'CatDet')]);
    activeContext.setActivePair('d1', 'm1');

    setDetectors([]);

    expect(activeContext.modelId).toBe('');
    expect(toast.toasts.length).toBe(1);
    expect(toast.toasts[0].message).toContain('CatDet');
  });

  it('does not toast when the user switches to a different active item', () => {
    setDatasets([makeDataset('d1', 'A'), makeDataset('d2', 'B')]);
    setDetectors([makeDetector('m1', 'M')]);
    activeContext.setActivePair('d1', 'm1');

    activeContext.setActivePair('d2', 'm1');

    expect(toast.toasts.length).toBe(0);
    expect(activeContext.datasetId).toBe('d2');
  });

  it('is idempotent — calling start() twice does not double-subscribe', () => {
    watcher.start();
    setDatasets([makeDataset('d1', 'X')]);
    setDetectors([]);
    activeContext.setActivePair('d1', '');
    setDatasets([]);

    expect(toast.toasts.length).toBe(1);
  });
});
