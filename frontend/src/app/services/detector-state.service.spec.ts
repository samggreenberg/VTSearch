import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { DetectorStateService } from './detector-state.service';

describe('DetectorStateService', () => {
  let service: DetectorStateService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(DetectorStateService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should start with empty state', () => {
    expect(service.detectors).toEqual([]);
    expect(service.extractors).toEqual([]);
    expect(service.localizers).toEqual([]);
  });

  it('loadDetectors should fetch and store detectors', () => {
    service.loadDetectors();
    const req = httpMock.expectOne('/api/autorun-detectors');
    req.flush({ detectors: [{ name: 'det1' }] });
    expect(service.detectors.length).toBe(1);
    expect(service.detectors[0].name).toBe('det1');
  });

  it('loadExtractors should fetch and store extractors', () => {
    service.loadExtractors();
    const req = httpMock.expectOne('/api/autorun-extractors');
    req.flush({ extractors: [{ name: 'ext1' }] });
    expect(service.extractors.length).toBe(1);
  });

  it('loadLocalizers should fetch and store localizers', () => {
    service.loadLocalizers();
    const req = httpMock.expectOne('/api/autorun-localizers');
    req.flush({ localizers: [{ name: 'loc1' }] });
    expect(service.localizers.length).toBe(1);
  });

  it('loadAll should fetch all three', () => {
    service.loadAll();
    httpMock.expectOne('/api/autorun-detectors').flush({ detectors: [] });
    httpMock.expectOne('/api/autorun-extractors').flush({ extractors: [] });
    httpMock.expectOne('/api/autorun-localizers').flush({ localizers: [] });
  });

  it('clear should reset all state', () => {
    service.loadDetectors();
    httpMock.expectOne('/api/autorun-detectors').flush({ detectors: [{ name: 'x' }] });

    service.clear();
    expect(service.detectors).toEqual([]);
    expect(service.extractors).toEqual([]);
    expect(service.localizers).toEqual([]);
  });
});
