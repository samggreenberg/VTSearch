import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { ProcessorImportersApiService } from './processor-importers-api.service';

describe('ProcessorImportersApiService', () => {
  let service: ProcessorImportersApiService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(ProcessorImportersApiService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('list should GET', () => {
    service.list().subscribe(data => expect(data.length).toBe(1));
    const req = httpMock.expectOne('/api/processor-importers');
    expect(req.request.method).toBe('GET');
    req.flush([{ name: 'server_detector_file' }]);
  });

  it('runImport should POST', () => {
    service.runImport('server_detector_file', { name: 'det' }).subscribe();
    const req = httpMock.expectOne('/api/processor-importers/import/server_detector_file');
    expect(req.request.method).toBe('POST');
    req.flush({});
  });
});
