import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { SeedImportersApiService } from './seed-importers-api.service';
import { provideHttpTesting } from '../testing/test-providers';

describe('SeedImportersApiService', () => {
  let service: SeedImportersApiService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [...provideHttpTesting()],
    });
    service = TestBed.inject(SeedImportersApiService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('list should GET the roster', () => {
    service.list().subscribe((res) => expect(res.importers.length).toBe(1));
    const req = httpMock.expectOne('/api/seed-importers');
    expect(req.request.method).toBe('GET');
    req.flush({ importers: [{ name: 'holder' }] });
  });

  it('run should POST the field values as JSON', () => {
    service.run('holder', { cluster: 'c1' }).subscribe((res) => expect(res.count).toBe(2));
    const req = httpMock.expectOne('/api/seed-import/holder');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ cluster: 'c1' });
    req.flush({ items: [{ filename: 'a' }, { filename: 'b' }], count: 2, truncated: false });
  });

  it('run should POST multipart when a file field is supplied', () => {
    const file = new File([new Uint8Array([1])], 'list.txt');
    service.run('holder', { listing: 'list.txt', cluster: 'c1' }, file, 'listing').subscribe();

    const req = httpMock.expectOne('/api/seed-import/holder');
    const body = req.request.body as FormData;
    expect(body instanceof FormData).toBe(true);
    // FormData.append clones the File under jsdom, so compare by name.
    expect((body.get('listing') as File).name).toBe('list.txt');
    expect(body.get('cluster')).toBe('c1');
    req.flush({ items: [], count: 0, truncated: false });
  });

  it('getFieldOptions should POST the dependency snapshot', () => {
    service.getFieldOptions('holder', 'cluster', { region: 'eu' }).subscribe();
    const req = httpMock.expectOne('/api/seed-import/holder/options');
    expect(req.request.body).toEqual({ field_key: 'cluster', values: { region: 'eu' } });
    req.flush({ options: [] });
  });

  it('encodes the plugin name into the URL', () => {
    service.run('odd name/x', {}).subscribe();
    httpMock.expectOne('/api/seed-import/odd%20name%2Fx').flush({
      items: [],
      count: 0,
      truncated: false,
    });
  });
});
