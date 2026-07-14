import { TestBed } from '@angular/core/testing';
import { NewThingFlowsService } from './new-thing-flows.service';
import { DemoDatasetEntry } from '../generated/api-client/models/demo-dataset-entry';

describe('NewThingFlowsService', () => {
  let service: NewThingFlowsService;

  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [NewThingFlowsService] });
    service = TestBed.inject(NewThingFlowsService);
  });

  it('both flows start closed', () => {
    expect(service.importer.open).toBe(false);
    expect(service.newDetector.open).toBe(false);
  });

  it('openImporter with no options defaults every field to empty', () => {
    service.openImporter();
    expect(service.importer).toEqual({
      open: true,
      initialTab: '',
      guessedMediaType: '',
      guessedMediaEmbedder: '',
    });
  });

  it('openImporter carries the supplied guesses', () => {
    service.openImporter({ initialTab: 'folder', guessedMediaType: 'audio', guessedMediaEmbedder: 'clap' });
    expect(service.importer).toEqual({
      open: true,
      initialTab: 'folder',
      guessedMediaType: 'audio',
      guessedMediaEmbedder: 'clap',
    });
  });

  it('closeImporter resets every field', () => {
    service.openImporter({ initialTab: 'folder' });
    service.closeImporter();
    expect(service.importer).toEqual({
      open: false,
      initialTab: '',
      guessedMediaType: '',
      guessedMediaEmbedder: '',
    });
  });

  it('openNewDetector defaults optional seed fields to undefined', () => {
    service.openNewDetector();
    expect(service.newDetector).toEqual({
      open: true,
      defaultMediaType: '',
      datasetEmbedder: '',
      seedMediaId: undefined,
      seedCropParams: undefined,
    });
  });

  it('openNewDetector carries seed media + crop params', () => {
    const seedCropParams = { box: [0, 0, 10, 10] };
    service.openNewDetector({
      defaultMediaType: 'image',
      datasetEmbedder: 'siglip',
      seedMediaId: 42,
      seedCropParams,
    });
    expect(service.newDetector).toEqual({
      open: true,
      defaultMediaType: 'image',
      datasetEmbedder: 'siglip',
      seedMediaId: 42,
      seedCropParams,
    });
  });

  it('closeNewDetector resets every field', () => {
    service.openNewDetector({ seedMediaId: 7 });
    service.closeNewDetector();
    expect(service.newDetector).toEqual({
      open: false,
      defaultMediaType: '',
      datasetEmbedder: '',
      seedMediaId: undefined,
      seedCropParams: undefined,
    });
  });

  it('emitDetectorCreated fires created$ with kind detector', () => {
    const events: unknown[] = [];
    service.created$.subscribe((e) => events.push(e));
    service.emitDetectorCreated('det-9');
    expect(events).toEqual([{ kind: 'detector', id: 'det-9' }]);
  });

  it('emitDatasetCreated fires created$ with kind dataset', () => {
    const events: unknown[] = [];
    service.created$.subscribe((e) => events.push(e));
    service.emitDatasetCreated('ds-9');
    expect(events).toEqual([{ kind: 'dataset', id: 'ds-9' }]);
  });

  it('emitImportStarted fires importStarted$', () => {
    const fired = vi.fn();
    service.importStarted$.subscribe(fired);
    service.emitImportStarted();
    expect(fired).toHaveBeenCalledTimes(1);
  });

  it('emitDemoSelected fires demoSelected$ with the demo entry', () => {
    const demo = { name: 'gtzan' } as DemoDatasetEntry;
    const events: unknown[] = [];
    service.demoSelected$.subscribe((e) => events.push(e));
    service.emitDemoSelected(demo);
    expect(events).toEqual([{ demo }]);
  });

  it('created$ is a plain Subject: late subscribers miss prior emissions', () => {
    service.emitDetectorCreated('early');
    const events: unknown[] = [];
    service.created$.subscribe((e) => events.push(e));
    expect(events).toEqual([]);
  });
});
