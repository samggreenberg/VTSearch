import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { MediaStateService } from './media-state.service';
import { MediaIdsResponse, MediaItem } from '../models/api.models';

describe('MediaStateService', () => {
  let service: MediaStateService;
  let httpMock: HttpTestingController;

  const mockIdsResp: MediaIdsResponse = {
    ids: [1, 2],
    type: 'audio',
    embedder: 'laion-clap-large',
  };
  const mockBatch: MediaItem[] = [
    { id: 1, type: 'audio', filename: 'a.wav', md5: 'abc', custom_metadata: {} },
    { id: 2, type: 'image', filename: 'b.png', md5: 'def', custom_metadata: {} },
  ];

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

  it('should start with empty state', () => {
    expect(service.mediaIds).toEqual([]);
    expect(service.mediaType).toBe('');
    expect(service.embedder).toBe('');
    expect(service.selectedId).toBeNull();
    expect(service.selectedMedia).toBeNull();
  });

  it('loadMedias should fetch ids and dataset metadata', () => {
    service.loadMedias();
    const req = httpMock.expectOne('/api/medias/ids');
    req.flush(mockIdsResp);
    expect(service.mediaIds).toEqual([1, 2]);
    expect(service.mediaType).toBe('audio');
    expect(service.embedder).toBe('laion-clap-large');
  });

  it('selectMedia should update selectedId and trigger metadata fetch', () => {
    service.loadMedias();
    httpMock.expectOne('/api/medias/ids').flush(mockIdsResp);

    service.selectMedia(2);
    expect(service.selectedId).toBe(2);
    // selectMedia triggers a batch fetch via the metadata cache.
    const batchReq = httpMock.expectOne('/api/medias/batch');
    batchReq.flush(mockBatch.filter((m) => m.id === 2));
    expect(service.selectedMedia?.filename).toBe('b.png');
  });

  it('selectedMedia should return null when metadata not yet cached', () => {
    service.loadMedias();
    httpMock.expectOne('/api/medias/ids').flush(mockIdsResp);

    service.selectMedia(999);
    // Cache flush — even if request errors, selectedMedia stays null until cached.
    httpMock.expectOne('/api/medias/batch').flush([]);
    expect(service.selectedMedia).toBeNull();
  });

  it('clear should reset all state', () => {
    service.loadMedias();
    httpMock.expectOne('/api/medias/ids').flush(mockIdsResp);
    service.selectMedia(1);
    httpMock.expectOne('/api/medias/batch').flush(mockBatch.filter((m) => m.id === 1));

    service.clear();
    expect(service.mediaIds).toEqual([]);
    expect(service.mediaType).toBe('');
    expect(service.embedder).toBe('');
    expect(service.selectedId).toBeNull();
  });

  it('mediaIds$ should emit on load', (done) => {
    const emissions: number[][] = [];
    service.mediaIds$.subscribe((ids) => emissions.push(ids));

    service.loadMedias();
    httpMock.expectOne('/api/medias/ids').flush(mockIdsResp);

    setTimeout(() => {
      expect(emissions.length).toBeGreaterThanOrEqual(2);
      expect(emissions[emissions.length - 1]).toEqual([1, 2]);
      done();
    });
  });

  it('selectedId$ should emit on select', (done) => {
    const ids: (number | null)[] = [];
    service.selectedId$.subscribe((id) => ids.push(id));

    service.selectMedia(5);
    // selectMedia triggers a batch fetch; flush it so afterEach is satisfied.
    httpMock.expectOne('/api/medias/batch').flush([]);

    setTimeout(() => {
      expect(ids).toContain(5);
      done();
    });
  });
});
