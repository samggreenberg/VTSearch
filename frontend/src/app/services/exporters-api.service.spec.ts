import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideHttpClient } from '@angular/common/http';
import { ExportersApiService } from './exporters-api.service';

describe('ExportersApiService', () => {
  let service: ExportersApiService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(ExportersApiService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('getExporters should GET', () => {
    service.getExporters().subscribe(data => expect(data.length).toBe(1));
    const req = httpMock.expectOne('/api/exporters');
    expect(req.request.method).toBe('GET');
    req.flush([{ name: 'json' }]);
  });

  it('runExport should POST', () => {
    service.runExport({ exporter_name: 'json' }).subscribe();
    const req = httpMock.expectOne('/api/exporters/export');
    expect(req.request.method).toBe('POST');
    expect(req.request.body.exporter_name).toBe('json');
    req.flush({});
  });
});
