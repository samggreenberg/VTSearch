import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { MediaStateService } from './media-state.service';
import { Media } from '../models/api.models';
import { settleResource } from '../testing/settle-resource';

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
      providers: [provideHttpClient(), provideHttpClientTesting()],
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

  it('mediasSignal should reflect the loaded medias', async () => {
    expect(service.mediasSignal()).toEqual([]);
    load();
    await settleResource();
    expect(service.mediasSignal()).toEqual(mockMedias);
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
