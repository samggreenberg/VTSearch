import { TestBed } from '@angular/core/testing';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { HttpClient, provideHttpClient, withInterceptors } from '@angular/common/http';

import { activeContextInterceptor } from './active-context.interceptor';
import { ActiveContextService } from '../services/active-context.service';

/**
 * The active-context interceptor stamps `X-Dataset-Id` / `X-Detector-Id` on
 * every outgoing request — the backbone of the multi-context state model. A
 * regression here silently mistargets every mutation, so pin the exact
 * header behaviour: only add a header when its id is non-empty.
 */
describe('activeContextInterceptor', () => {
  let http: HttpClient;
  let httpMock: HttpTestingController;
  let ctx: ActiveContextService;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(withInterceptors([activeContextInterceptor])),
        provideHttpClientTesting(),
      ],
    });
    http = TestBed.inject(HttpClient);
    httpMock = TestBed.inject(HttpTestingController);
    ctx = TestBed.inject(ActiveContextService);
  });

  afterEach(() => httpMock.verify());

  it('adds no context headers when nothing is selected', () => {
    http.get('/api/foo').subscribe();
    const req = httpMock.expectOne('/api/foo');
    expect(req.request.headers.has('X-Dataset-Id')).toBe(false);
    expect(req.request.headers.has('X-Detector-Id')).toBe(false);
    req.flush({});
  });

  it('stamps only X-Dataset-Id when a dataset but no detector is active', () => {
    ctx.setActive('ds-1', '');
    http.get('/api/foo').subscribe();
    const req = httpMock.expectOne('/api/foo');
    expect(req.request.headers.get('X-Dataset-Id')).toBe('ds-1');
    expect(req.request.headers.has('X-Detector-Id')).toBe(false);
    req.flush({});
  });

  it('stamps only X-Detector-Id when a detector but no dataset is active', () => {
    ctx.setActive('', 'mdl-7');
    http.get('/api/foo').subscribe();
    const req = httpMock.expectOne('/api/foo');
    expect(req.request.headers.has('X-Dataset-Id')).toBe(false);
    expect(req.request.headers.get('X-Detector-Id')).toBe('mdl-7');
    req.flush({});
  });

  it('stamps both headers when a dataset and detector are active', () => {
    ctx.setActive('ds-1', 'mdl-7');
    http.get('/api/foo').subscribe();
    const req = httpMock.expectOne('/api/foo');
    expect(req.request.headers.get('X-Dataset-Id')).toBe('ds-1');
    expect(req.request.headers.get('X-Detector-Id')).toBe('mdl-7');
    req.flush({});
  });

  it('preserves headers already on the request', () => {
    ctx.setActive('ds-1', 'mdl-7');
    http.get('/api/foo', { headers: { 'X-Custom': 'keep-me' } }).subscribe();
    const req = httpMock.expectOne('/api/foo');
    expect(req.request.headers.get('X-Custom')).toBe('keep-me');
    expect(req.request.headers.get('X-Dataset-Id')).toBe('ds-1');
    req.flush({});
  });
});
