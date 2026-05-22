import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { MediaStateService } from './media-state.service';
import { Media } from '../models/api.models';

describe('MediaStateService', () => {
  let service: MediaStateService;
  let httpMock: HttpTestingController;

  const mockMedias: Media[] = [
    { id: 1, media_type: 'audio', filename: 'a.wav', md5: 'abc', custom_metadata: {} },
    { id: 2, media_type: 'image', filename: 'b.png', md5: 'def', custom_metadata: {} },
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

  it('should start with empty medias', () => {
    expect(service.medias).toEqual([]);
    expect(service.selectedId).toBeNull();
    expect(service.selectedMedia).toBeNull();
  });

  it('loadMedias should fetch and store medias', () => {
    service.loadMedias();
    const req = httpMock.expectOne('/api/medias/ids');
    req.flush(mockMedias);
    expect(service.medias).toEqual(mockMedias);
  });

  it('selectMedia should update selectedId', () => {
    service.loadMedias();
    httpMock.expectOne('/api/medias/ids').flush(mockMedias);

    service.selectMedia(2);
    expect(service.selectedId).toBe(2);
    expect(service.selectedMedia?.filename).toBe('b.png');
  });

  it('selectedMedia should return null for unknown id', () => {
    service.loadMedias();
    httpMock.expectOne('/api/medias/ids').flush(mockMedias);

    service.selectMedia(999);
    expect(service.selectedMedia).toBeNull();
  });

  it('clear should reset all state', () => {
    service.loadMedias();
    httpMock.expectOne('/api/medias/ids').flush(mockMedias);
    service.selectMedia(1);

    service.clear();
    expect(service.medias).toEqual([]);
    expect(service.selectedId).toBeNull();
  });

  it('medias$ should emit on load', (done) => {
    const emissions: Media[][] = [];
    service.medias$.subscribe((m) => emissions.push(m));

    service.loadMedias();
    httpMock.expectOne('/api/medias/ids').flush(mockMedias);

    setTimeout(() => {
      expect(emissions.length).toBeGreaterThanOrEqual(2); // initial [] + loaded
      expect(emissions[emissions.length - 1]).toEqual(mockMedias);
      done();
    });
  });

  it('selectedId$ should emit on select', (done) => {
    const ids: (number | null)[] = [];
    service.selectedId$.subscribe((id) => ids.push(id));

    service.selectMedia(5);

    setTimeout(() => {
      expect(ids).toContain(5);
      done();
    });
  });
});
