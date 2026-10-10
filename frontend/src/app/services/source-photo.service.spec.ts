import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { ActiveContextService } from './active-context.service';
import { DatasetStateService } from './dataset-state.service';
import { SourceLookup, SourcePhotoService } from './source-photo.service';
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
    { id: 'photos', name: 'Photos – Image', media_type: 'image', import_group: 'g1' },
    { id: 'pages', name: 'Scans – Document', media_type: 'document', import_group: 'g2' },
    { id: 'birds', name: 'Birds', media_type: 'image' },
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
});
