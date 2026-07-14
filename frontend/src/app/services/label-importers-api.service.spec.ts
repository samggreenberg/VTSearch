import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { LabelImportersApiService } from './label-importers-api.service';
import { provideHttpTesting } from '../testing/test-providers';

describe('LabelImportersApiService', () => {
  let service: LabelImportersApiService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [...provideHttpTesting()],
    });
    service = TestBed.inject(LabelImportersApiService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('list should GET', () => {
    service.list().subscribe(data => expect(data.length).toBe(1));
    const req = httpMock.expectOne('/api/label-importers');
    expect(req.request.method).toBe('GET');
    req.flush([{ name: 'server_json_file' }]);
  });

  it('runImport should POST', () => {
    service.runImport('server_json_file', { path: '/data/labels.json' }).subscribe();
    const req = httpMock.expectOne('/api/label-importers/import/server_json_file');
    expect(req.request.method).toBe('POST');
    req.flush({});
  });

  it('ingestMissing should POST', () => {
    service.ingestMissing([]).subscribe();
    const req = httpMock.expectOne('/api/label-importers/ingest-missing');
    expect(req.request.method).toBe('POST');
    req.flush({});
  });
});
