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
    expect(service.extractors).toEqual([]);
    expect(service.localizers).toEqual([]);
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

  it('loadAll should fetch extractors and localizers', () => {
    service.loadAll();
    httpMock.expectOne('/api/autorun-extractors').flush({ extractors: [] });
    httpMock.expectOne('/api/autorun-localizers').flush({ localizers: [] });
  });

  it('clear should reset all state', () => {
    service.loadExtractors();
    httpMock.expectOne('/api/autorun-extractors').flush({ extractors: [{ name: 'x' }] });

    service.clear();
    expect(service.extractors).toEqual([]);
    expect(service.localizers).toEqual([]);
  });
});
