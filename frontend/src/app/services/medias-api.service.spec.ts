import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { MediasApiService } from './medias-api.service';

describe('MediasApiService', () => {
  let service: MediasApiService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(MediasApiService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('getMediaIds should GET /api/medias/ids', () => {
    const mock = { ids: [1, 2], type: 'audio', embedder: 'laion-clap-large' };
    service.getMediaIds().subscribe(data => expect(data).toEqual(mock));
    const req = httpMock.expectOne('/api/medias/ids');
    expect(req.request.method).toBe('GET');
    req.flush(mock);
  });

  it('getMediasBatch should POST ids to /api/medias/batch', () => {
    const mock = [{ id: 1, type: 'audio', filename: 'a.wav', md5: 'abc', custom_metadata: {} }];
    service.getMediasBatch([1]).subscribe(data => expect(data).toEqual(mock));
    const req = httpMock.expectOne('/api/medias/batch');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ ids: [1] });
    req.flush(mock);
  });

  it('getAudio should GET blob', () => {
    service.getAudio(1).subscribe(data => expect(data instanceof Blob).toBeTrue());
    const req = httpMock.expectOne('/api/medias/1/audio');
    expect(req.request.method).toBe('GET');
    expect(req.request.responseType).toBe('blob');
    req.flush(new Blob());
  });

  it('getVideo should GET blob', () => {
    service.getVideo(2).subscribe();
    const req = httpMock.expectOne('/api/medias/2/video');
    expect(req.request.method).toBe('GET');
    req.flush(new Blob());
  });

  it('getImage should GET blob', () => {
    service.getImage(3).subscribe();
    const req = httpMock.expectOne('/api/medias/3/image');
    expect(req.request.method).toBe('GET');
    req.flush(new Blob());
  });

  it('getText should GET json', () => {
    service.getText(4).subscribe(data => expect(data.content).toBe('hello'));
    const req = httpMock.expectOne('/api/medias/4/text');
    expect(req.request.method).toBe('GET');
    req.flush({ content: 'hello', word_count: 1, character_count: 5 });
  });

  it('getMedia should GET blob', () => {
    service.getMedia(5).subscribe();
    const req = httpMock.expectOne('/api/medias/5/media');
    expect(req.request.method).toBe('GET');
    req.flush(new Blob());
  });

  it('vote should POST with label', () => {
    service.vote(1, 'good').subscribe(data => expect(data.ok).toBeTrue());
    const req = httpMock.expectOne('/api/medias/1/vote');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ vote: 'good' });
    req.flush({ ok: true });
  });
});
