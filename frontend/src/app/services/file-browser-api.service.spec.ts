import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';

import { FileBrowserApiService } from './file-browser-api.service';
import { provideHttpTesting } from '../testing/test-providers';

describe('FileBrowserApiService', () => {
  let service: FileBrowserApiService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [...provideHttpTesting()],
    });
    service = TestBed.inject(FileBrowserApiService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('browse() GETs /api/browse with the path query param', () => {
    let result: unknown;
    service.browse('sub/dir').subscribe(r => (result = r));

    const req = httpMock.expectOne(r => r.url === '/api/browse');
    expect(req.request.method).toBe('GET');
    expect(req.request.params.get('path')).toBe('sub/dir');
    // No extensions passed -> the param is omitted entirely.
    expect(req.request.params.has('extensions')).toBe(false);

    const body = { directories: [], files: [], rootPath: '/data', currentPath: 'sub/dir' };
    req.flush(body);
    expect(result).toEqual(body);
  });

  it('browse() forwards the extensions filter when supplied', () => {
    service.browse('', '.wav,.mp3').subscribe();

    const req = httpMock.expectOne(r => r.url === '/api/browse');
    expect(req.request.params.get('path')).toBe('');
    expect(req.request.params.get('extensions')).toBe('.wav,.mp3');
    req.flush({ directories: [], files: [] });
  });

  it('browse() maps the HttpResponse to its body', () => {
    let result: unknown;
    service.browse('x').subscribe(r => (result = r));

    const payload = {
      directories: [{ name: 'a', path: 'x/a' }],
      files: [{ name: 'f.txt', path: 'x/f.txt', size_bytes: 3 }],
    };
    httpMock.expectOne(r => r.url === '/api/browse').flush(payload);

    // The service unwraps `r.body`, so subscribers get the payload directly,
    // not the wrapping HttpResponse.
    expect(result).toEqual(payload);
  });
});
