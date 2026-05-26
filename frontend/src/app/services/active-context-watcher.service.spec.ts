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
    // Registry still empty; could just be loading.
    expect(toast.toasts.length).toBe(0);
    expect(activeContext.datasetId).toBe('d1');
    expect(activeContext.modelId).toBe('m1');
  });

  it('toasts and clears the dataset half when the active dataset disappears', () => {
    setDatasets([makeDataset('d1', 'Audio Foo')]);
    setDetectors([makeDetector('m1', 'Det')]);
    activeContext.setActivePair('d1', 'm1');
    expect(toast.toasts.length).toBe(0);

    // d1 deleted from another tab / session; detectors still present so the
    // detectors.length > 0 guard confirms the registry is loaded.
    setDatasets([]);
    // Re-emit detectors to trigger combineLatest with detectors.length > 0.
    setDetectors([makeDetector('m1', 'Det')]);

    expect(activeContext.datasetId).toBe('');
    expect(activeContext.modelId).toBe('m1');
    expect(toast.toasts.length).toBe(1);
    expect(toast.toasts[0].message).toContain('Audio Foo');
  });

  it('toasts and clears the detector half when the active detector disappears', () => {
    setDatasets([makeDataset('d1', 'DS')]);
    setDetectors([makeDetector('m1', 'CatDet'), makeDetector('m2', 'Other')]);
    activeContext.setActivePair('d1', 'm1');

    // m1 deleted; m2 still present (registry confirmed loaded via datasets.length > 0).
    setDetectors([makeDetector('m2', 'Other')]);

    expect(activeContext.modelId).toBe('');
    expect(toast.toasts.length).toBe(1);
    expect(toast.toasts[0].message).toContain('CatDet');
  });

  it('toasts when the only detector is deleted (datasets still present)', () => {
    setDatasets([makeDataset('d1', 'DS')]);
    setDetectors([makeDetector('m1', 'Solo')]);
    activeContext.setActivePair('d1', 'm1');

    // m1 is the only detector; datasets still present so datasets.length > 0 passes.
    setDetectors([]);

    expect(activeContext.modelId).toBe('');
    expect(toast.toasts.length).toBe(1);
    expect(toast.toasts[0].message).toContain('Solo');
  });

  it('does not toast when the user switches to a different active item', () => {
    setDatasets([makeDataset('d1', 'A'), makeDataset('d2', 'B')]);
    setDetectors([makeDetector('m1', 'M')]);
    activeContext.setActivePair('d1', 'm1');

    activeContext.setActivePair('d2', 'm1');

    expect(toast.toasts.length).toBe(0);
    expect(activeContext.datasetId).toBe('d2');
  });

  it('does not toast when intent has already moved to a different detector', () => {
    // Simulates the delete-then-create-and-train flow: user deletes m1, creates
    // m2, navigates to the train view. The route guard sets intent=m2 but the
    // active is still m1 when the registry refresh (with m1 gone, m2 present)
    // arrives. The watcher must not fire "m1 removed" in this case.
    setDatasets([makeDataset('d1', 'DS')]);
    setDetectors([makeDetector('m1', 'OldDet'), makeDetector('m2', 'NewDet')]);
    activeContext.setActivePair('d1', 'm1');

    // Intent moves to m2 (route guard called setIntent); active still m1.
    activeContext.setIntent('d1', 'm2');

    // Registry arrives: m1 gone, m2 present.
    setDetectors([makeDetector('m2', 'NewDet')]);

    // Must NOT fire: user already moved on.
    expect(toast.toasts.length).toBe(0);
    expect(activeContext.modelId).toBe('m1'); // active unchanged; route guard promotes it
  });

  it('does not toast when intent has already moved to a different dataset', () => {
    setDatasets([makeDataset('d1', 'OldDS'), makeDataset('d2', 'NewDS')]);
    setDetectors([makeDetector('m1', 'Det')]);
    activeContext.setActivePair('d1', 'm1');

    // Intent moves to d2; active still d1.
    activeContext.setIntent('d2', 'm1');

    // d1 disappears, d2 remains.
    setDatasets([makeDataset('d2', 'NewDS')]);
    setDetectors([makeDetector('m1', 'Det')]);

    expect(toast.toasts.length).toBe(0);
    expect(activeContext.datasetId).toBe('d1'); // active unchanged
  });

  it('is idempotent: calling start() twice does not double-subscribe', () => {
    watcher.start(); // second call, should be a no-op
    setDatasets([makeDataset('d1', 'X')]);
    setDetectors([makeDetector('m1', 'Det')]);
    activeContext.setActivePair('d1', 'm1');

    // m1 deleted; datasets still present so datasets.length > 0 guard passes.
    setDetectors([]);

    // If start() double-subscribed we'd see 2 toasts; idempotency gives exactly 1.
    expect(toast.toasts.length).toBe(1);
  });
});
