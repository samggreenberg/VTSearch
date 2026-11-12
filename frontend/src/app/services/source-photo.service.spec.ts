import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { ActiveContextService } from './active-context.service';
import { DatasetStateService } from './dataset-state.service';
import { MediaTypeCapabilityService } from './media-type-capability.service';
import { SourceLookup, SourcePhotoService, WIDER_SCALE, widerBox, widerView } from './source-photo.service';
import { SKIP_ERROR_TOAST } from '../interceptors/error.interceptor';
import { provideHttpTesting } from '../testing/test-providers';

/** Show in photo (#4750): which datasets offer it, and what the lookup reads. */
describe('SourcePhotoService', () => {
  let service: SourcePhotoService;
  let activeContext: ActiveContextService;
  let datasetState: DatasetStateService;
  let httpMock: HttpTestingController;

  const registry = [
    { id: 'faces', name: 'Photos – Face', media_type: 'face', import_group: 'g1' },
    { id: 'photos', name: 'Photos – Image', media_type: 'image', import_group: 'g1', loaded: true },
    { id: 'pages', name: 'Scans – Document', media_type: 'document', import_group: 'g2' },
    { id: 'holiday-pages', name: 'Holiday – Document', media_type: 'document', import_group: 'g3' },
    { id: 'holiday-photos', name: 'Holiday – Image', media_type: 'image', import_group: 'g3' },
    { id: 'birds', name: 'Birds', media_type: 'image' },
    { id: 'trip-faces', name: 'Trip – Face', media_type: 'face', import_group: 'g4' },
    { id: 'trip-photos', name: 'Trip – Image', media_type: 'image', import_group: 'g4', loaded: false },
  ];

  function flushRegistry(): void {
    datasetState.refresh();
    httpMock.expectOne('/api/datasets/registry').flush({ datasets: registry });
    httpMock.expectOne('/api/detectors/registry').flush({ detectors: [] });
  }

  function lookup(id: number): SourceLookup[] {
    const seen: SourceLookup[] = [];
    service.lookup(id).subscribe((v) => seen.push(v));
    return seen;
  }

  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [...provideHttpTesting()] });
    service = TestBed.inject(SourcePhotoService);
    activeContext = TestBed.inject(ActiveContextService);
    datasetState = TestBed.inject(DatasetStateService);
    httpMock = TestBed.inject(HttpTestingController);
    // The media-type registry, which the centre pane loads: `face` is convert-in.
    TestBed.inject(MediaTypeCapabilityService).ensureLoaded();
    httpMock.expectOne('/api/media-types').flush({
      media_types: [
        { type_id: 'face', name: 'Face', importable: false },
        { type_id: 'image', name: 'Image', importable: true },
        { type_id: 'document', name: 'Document', importable: true },
      ],
    });
    flushRegistry();
  });

  afterEach(() => httpMock.verify());

  describe('offered', () => {
    it('is on for a dataset whose import also produced an Image dataset', () => {
      activeContext.setActivePair('faces', 'det');
      expect(service.offered()).toBe(true);
    });

    it('is off for the Image dataset itself, and for an ungrouped dataset', () => {
      activeContext.setActivePair('photos', 'det');
      expect(service.offered()).toBe(false);
      activeContext.setActivePair('birds', 'det');
      expect(service.offered()).toBe(false);
    });

    it('is off when the group has no Image dataset', () => {
      activeContext.setActivePair('pages', 'det');
      expect(service.offered()).toBe(false);
    });

    it('is off for a Document dataset beside an Image one: pages have no source item', () => {
      activeContext.setActivePair('holiday-pages', 'det');
      expect(service.offered()).toBe(false);
    });
  });

  describe('photosLoaded', () => {
    it('is on once the Image dataset beside the faces is in memory', () => {
      activeContext.setActivePair('faces', 'det');
      expect(service.photosLoaded()).toBe(true);
    });

    it('is off while that dataset is registered but not loaded', () => {
      activeContext.setActivePair('trip-faces', 'det');
      expect(service.offered()).toBe(true);
      expect(service.photosLoaded()).toBe(false);
    });

    it('is off wherever Show in photo is, loaded photos or not', () => {
      activeContext.setActivePair('photos', 'det');
      expect(service.photosLoaded()).toBe(false);
    });
  });

  describe('open / close', () => {
    it('records the item and the dataset Back returns to', () => {
      activeContext.setActivePair('faces', 'det');
      service.open(7);
      expect(service.request()).toEqual({ mediaId: 7, fromName: 'Photos – Face' });
      service.close();
      expect(service.request()).toBeNull();
    });
  });

  describe('lookup', () => {
    it('reads a 200 as the item found, with its box', () => {
      const seen = lookup(7);
      const req = httpMock.expectOne('/api/medias/7/source');
      expect(req.request.context.get(SKIP_ERROR_TOAST)).toBe(true);
      req.flush({ dataset_id: 'photos', media_id: 2, box: [0.1, 0.2, 0.3, 0.4] });
      expect(seen).toEqual([{ kind: 'found', datasetId: 'photos', mediaId: 2, box: [0.1, 0.2, 0.3, 0.4] }]);
    });

    it('reads a missing or malformed box as none', () => {
      const seen = lookup(7);
      httpMock.expectOne('/api/medias/7/source').flush({ dataset_id: 'photos', media_id: 2, box: null });
      expect(seen[0]).toEqual({ kind: 'found', datasetId: 'photos', mediaId: 2, box: null });
    });

    it('reads a source_not_loaded 409 as the sibling to load', () => {
      const seen = lookup(7);
      httpMock
        .expectOne('/api/medias/7/source')
        .flush(
          { code: 409, message: 'Load the source dataset', error_code: 'source_not_loaded', dataset_id: 'photos' },
          { status: 409, statusText: 'Conflict' },
        );
      expect(seen).toEqual([{ kind: 'not-loaded', datasetId: 'photos' }]);
    });

    it('reads a 404 as no source, keeping the server\'s reason', () => {
      const seen = lookup(7);
      httpMock
        .expectOne('/api/medias/7/source')
        .flush(
          { code: 404, message: 'This item was not made by a converter.', error_code: 'not_derived' },
          { status: 404, statusText: 'Not Found' },
        );
      expect(seen).toEqual([{ kind: 'none', message: 'This item was not made by a converter.' }]);
    });

    it('reads anything else as an error rather than throwing', () => {
      const seen = lookup(7);
      httpMock
        .expectOne('/api/medias/7/source')
        .flush({ code: 500, message: 'boom' }, { status: 500, statusText: 'Server Error' });
      expect(seen).toEqual([{ kind: 'error', message: 'boom' }]);
    });
  });

  describe('the Wider view (#5645)', () => {
    it('widens a box about its own centre, past the photo\'s edge if it must', () => {
      expect(WIDER_SCALE).toBe(2);
      const box = widerBox([0, 0.4, 0.2, 0.6]);
      [-0.1, 0.3, 0.3, 0.7].forEach((v, i) => expect(box[i]).toBeCloseTo(v, 9));
    });

    it('is pending while the lookup is out', () => {
      expect(widerView(undefined, true)).toEqual({ kind: 'pending' });
    });

    it('is the photo, framed wide on the face, once it is found', () => {
      const found: SourceLookup = { kind: 'found', datasetId: 'photos', mediaId: 2, box: [0.4, 0.4, 0.6, 0.6] };
      const view = widerView(found, false, 'a.jpg→face_0.jpg');
      expect(view.kind).toBe('photo');
      if (view.kind !== 'photo') return;
      expect(view.datasetId).toBe('photos');
      expect(view.media).toEqual({ id: 2, media_type: 'image', filename: 'a.jpg→face_0.jpg' });
      [0.3, 0.3, 0.7, 0.7].forEach((v, i) => expect(view.fitBox[i]).toBeCloseTo(v, 9));
    });

    it('stays on the crop when there is no photo, or no box to frame on it', () => {
      expect(widerView(undefined, false)).toEqual({ kind: 'crop' });
      expect(widerView({ kind: 'found', datasetId: 'photos', mediaId: 2, box: null }, false)).toEqual({ kind: 'crop' });
      expect(widerView({ kind: 'not-loaded', datasetId: 'photos' }, false)).toEqual({ kind: 'crop' });
      expect(widerView({ kind: 'none', message: 'no' }, false)).toEqual({ kind: 'crop' });
      expect(widerView({ kind: 'error', message: 'boom' }, false)).toEqual({ kind: 'crop' });
    });
  });
});
