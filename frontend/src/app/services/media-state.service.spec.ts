import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { MediaStateService } from './media-state.service';
import { Media } from '../models/api.models';
import { settleResource } from '../testing/settle-resource';
import { provideHttpTesting } from '../testing/test-providers';

describe('MediaStateService', () => {
  let service: MediaStateService;
  let httpMock: HttpTestingController;

  const mockMedias: Media[] = [
    { id: 1, media_type: 'audio', filename: 'a.wav', md5: 'abc', custom_metadata: {} },
    { id: 2, media_type: 'image', filename: 'b.png', md5: 'def', custom_metadata: {} },
  ];

  function load() {
    service.loadMedias();
    TestBed.tick();
    httpMock.expectOne('/api/medias/ids').flush(mockMedias);
  }

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [...provideHttpTesting()],
    });
    service = TestBed.inject(MediaStateService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should start with empty medias and not fetch until loadMedias()', () => {
    TestBed.tick();
    expect(service.mediasSignal()).toEqual([]);
    expect(service.selectedId()).toBeNull();
    expect(service.selectedMedia).toBeNull();
    httpMock.expectNone('/api/medias/ids');
  });

  it('loadMedias should fetch and store medias', async () => {
    load();
    await settleResource();
    expect(service.mediasSignal()).toEqual(mockMedias);
  });

  it('selectMedia should update selectedId', async () => {
    load();
    await settleResource();

    service.selectMedia(2);
    expect(service.selectedId()).toBe(2);
    expect(service.selectedMedia?.filename).toBe('b.png');
  });

  it('selectedMedia should return null for unknown id', async () => {
    load();
    await settleResource();

    service.selectMedia(999);
    expect(service.selectedMedia).toBeNull();
  });

  it('clear should reset all state', async () => {
    load();
    await settleResource();
    service.selectMedia(1);

    service.clear();
    TestBed.tick();
    expect(service.mediasSignal()).toEqual([]);
    expect(service.selectedId()).toBeNull();
  });

  it('clearSelection should drop the selection but keep the loaded medias', async () => {
    load();
    await settleResource();
    service.selectMedia(1);
    expect(service.selectedMedia).not.toBeNull();

    // The pair-change reset wants only this half: the stub list is re-fetched
    // by `loadMedias()` rather than blanked, so the grid never flashes empty
    // (issue #3489).
    service.clearSelection();
    TestBed.tick();
    expect(service.selectedId()).toBeNull();
    expect(service.selectedMedia).toBeNull();
    expect(service.mediasSignal()).toEqual(mockMedias);
  });

  it('loadMedias should restart an in-flight fetch instead of dropping the reload', async () => {
    // Pair switch A->B issues the first fetch (stamped with B's X-Dataset-Id)...
    service.loadMedias();
    TestBed.tick();
    const stale = httpMock.expectOne('/api/medias/ids');

    // ...and a rapid B->C switch reloads while it is still in flight. The
    // reload must not be swallowed: the in-flight response carries the *old*
    // pair's id list, and nothing else would ever re-fetch (issue #2944).
    service.loadMedias();
    TestBed.tick();

    expect(stale.cancelled).toBe(true);
    const fresh = httpMock.expectOne('/api/medias/ids');
    fresh.flush(mockMedias);
    await settleResource();
    expect(service.mediasSignal()).toEqual(mockMedias);
  });

  it('mediasSignal should reflect the loaded medias', async () => {
    expect(service.mediasSignal()).toEqual([]);
    load();
    await settleResource();
    expect(service.mediasSignal()).toEqual(mockMedias);
  });

  /**
   * #5661: an `rxResource`'s `value()` throws once its loader fails, so reading
   * it straight through left every `mediasSignal()` reader throwing after a
   * failed `GET /api/medias/ids`, and this store is app-wide.
   */
  describe('after a failed load (#5661)', () => {
    async function failLoad() {
      service.loadMedias();
      TestBed.tick();
      httpMock
        .expectOne('/api/medias/ids')
        .flush({ message: 'boom' }, { status: 500, statusText: 'Internal Server Error' });
      await settleResource();
    }

    it('a failed first load reads as empty, without throwing', async () => {
      await failLoad();
      expect(service.isLoading()).toBe(false);
      expect(service.mediasSignal()).toEqual([]);
      expect(service.getMedia(1)).toBeNull();
    });

    // A reload is nearly always a pair switch, so the kept list would be the
    // last pair's ids rendered against this one (#3489).
    it('a failed reload reads as empty rather than keeping the last list', async () => {
      load();
      await settleResource();
      expect(service.mediasSignal()).toEqual(mockMedias);

      await failLoad();
      expect(service.mediasSignal()).toEqual([]);
    });

    it('a reload after a failed load brings the list in', async () => {
      await failLoad();
      load();
      await settleResource();
      expect(service.mediasSignal()).toEqual(mockMedias);
    });
  });

  it('selectedId signal should update on select', () => new Promise<void>((done) => {
    expect(service.selectedId()).toBeNull();
    service.selectMedia(5);
    expect(service.selectedId()).toBe(5);

    // selectMedia() also schedules a debounced metadata batch fetch
    // (POST /api/medias/batch) via the metadata cache; drain it so the
    // afterEach httpMock.verify() doesn't see an open request.
    setTimeout(() => {
      for (const req of httpMock.match('/api/medias/batch')) {
        req.flush([]);
      }
      done();
    });
  }));
});
