import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { DetectorsApiService } from './detectors-api.service';

describe('DetectorsApiService', () => {
  let service: DetectorsApiService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(DetectorsApiService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('getAutorunDetectors should GET', () => {
    service.getAutorunDetectors().subscribe(data => expect(data.detectors).toBeDefined());
    const req = httpMock.expectOne('/api/autorun-detectors');
    expect(req.request.method).toBe('GET');
    req.flush({ detectors: [] });
  });

  it('createDetector should POST', () => {
    service.createDetector({ name: 'det1', media_type: 'audio' }).subscribe(data => {
      expect(data.success).toBeTrue();
    });
    const req = httpMock.expectOne('/api/autorun-detectors');
    expect(req.request.method).toBe('POST');
    expect(req.request.body.name).toBe('det1');
    req.flush({ success: true, name: 'det1' });
  });

  it('deleteDetector should DELETE', () => {
    service.deleteDetector('det1').subscribe(data => expect(data.success).toBeTrue());
    const req = httpMock.expectOne('/api/autorun-detectors/det1');
    expect(req.request.method).toBe('DELETE');
    req.flush({ success: true });
  });

  it('renameDetector should PUT', () => {
    service.renameDetector('old', 'new').subscribe(data => expect(data.new_name).toBe('new'));
    const req = httpMock.expectOne('/api/autorun-detectors/old/rename');
    expect(req.request.method).toBe('PUT');
    expect(req.request.body).toEqual({ new_name: 'new' });
    req.flush({ success: true, new_name: 'new' });
  });

  it('detectorSort should POST', () => {
    service.detectorSort({ detector: 'det1' }).subscribe();
    const req = httpMock.expectOne('/api/detector-sort');
    expect(req.request.method).toBe('POST');
    req.flush({ results: [], threshold: 0.5 });
  });

  it('autoDetect should POST', () => {
    service.autoDetect({}).subscribe();
    const req = httpMock.expectOne('/api/auto-detect');
    expect(req.request.method).toBe('POST');
    req.flush({});
  });

  it('getServerFiles should GET', () => {
    service.getServerFiles().subscribe();
    const req = httpMock.expectOne('/api/detector/server-files');
    expect(req.request.method).toBe('GET');
    req.flush({ files: [] });
  });

  it('getAutorunExtractors should GET', () => {
    service.getAutorunExtractors().subscribe();
    const req = httpMock.expectOne('/api/autorun-extractors');
    expect(req.request.method).toBe('GET');
    req.flush({ extractors: [] });
  });

  it('getAutorunLocalizers should GET', () => {
    service.getAutorunLocalizers().subscribe();
    const req = httpMock.expectOne('/api/autorun-localizers');
    expect(req.request.method).toBe('GET');
    req.flush({ localizers: [] });
  });

  it('find should POST', () => {
    service.find({ datasets: [], models: [] }).subscribe();
    const req = httpMock.expectOne('/api/find');
    expect(req.request.method).toBe('POST');
    req.flush({});
  });
});
