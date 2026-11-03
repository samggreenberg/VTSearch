import { TestBed } from '@angular/core/testing';
import { HttpTestingController } from '@angular/common/http/testing';
import { HttpClient } from '@angular/common/http';

import { metricsInterceptor } from './metrics.interceptor';
import { MetricsService } from '../services/metrics.service';
import { provideHttpTesting } from '../testing/test-providers';

/**
 * The metrics interceptor turns a successful call to an action endpoint into
 * a usage event. Pin the gate (method + tracked path + 2xx) and what each row
 * reads out of the request.
 */
describe('metricsInterceptor', () => {
  let http: HttpClient;
  let httpMock: HttpTestingController;
  let metrics: { logEvent: ReturnType<typeof vi.fn> };

  beforeEach(() => {
    metrics = { logEvent: vi.fn() };
    TestBed.configureTestingModule({
      providers: [...provideHttpTesting(metricsInterceptor), { provide: MetricsService, useValue: metrics }],
    });
    http = TestBed.inject(HttpClient);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  function post(url: string, body: unknown): void {
    http.post(url, body).subscribe();
    httpMock.expectOne(url).flush({ ok: true });
  }

  const cases: { url: string; body?: unknown; group: string; attributes: Record<string, unknown> }[] = [
    {
      url: '/api/dataset/import/server_folder',
      body: { path: '/data/x', media_type: 'image' },
      group: 'datasetImported',
      attributes: { importer: 'server_folder', media_type: 'image' },
    },
    {
      url: '/api/dataset/load-demo',
      body: { name: 'esc50' },
      group: 'datasetImported',
      attributes: { importer: 'demo', demo: 'esc50' },
    },
    { url: '/api/dataset/load-file', body: new FormData(), group: 'datasetImported', attributes: { importer: 'pickle_file' } },
    { url: '/api/dataset/combine', body: {}, group: 'datasetsCombined', attributes: {} },
    { url: '/api/dataset/promote', body: {}, group: 'datasetPromoted', attributes: {} },
    {
      url: '/api/exporters/export',
      body: { exporter_name: 'csv', payload_kind: 'find_results', results: {} },
      group: 'resultsExported',
      attributes: { exporter: 'csv', payload: 'find_results' },
    },
    {
      url: '/api/label-importers/import/json_file',
      body: {},
      group: 'labelsImported',
      attributes: { importer: 'json_file', into: 'new_detector' },
    },
    {
      url: '/api/detectors/dogs/import-labels/csv_file',
      body: {},
      group: 'labelsImported',
      attributes: { importer: 'csv_file', into: 'existing_detector' },
    },
    {
      url: '/api/detectors',
      body: { name: 'dogs', media_type: 'audio', embedder_type: 'semantic', text_query: 'bark' },
      group: 'detectorCreated',
      attributes: { source: 'new_detector_form', media_type: 'audio', embedder_type: 'semantic' },
    },
    {
      url: '/api/detectors/registry/from-labelset/json_file',
      body: {},
      group: 'detectorCreated',
      attributes: { source: 'labelset', importer: 'json_file' },
    },
    { url: '/api/detectors/d1/portable-bundle', body: {}, group: 'detectorExported', attributes: {} },
    { url: '/api/find', body: {}, group: 'find', attributes: {} },
    {
      url: '/api/settings-importers/import/json_file',
      body: {},
      group: 'settingsImported',
      attributes: { importer: 'json_file' },
    },
    {
      url: '/api/settings-exporters/export',
      body: { exporter_name: 'json_file' },
      group: 'settingsExported',
      attributes: { exporter: 'json_file' },
    },
  ];

  for (const c of cases) {
    it(`logs ${c.group} after a successful POST to ${c.url}`, () => {
      post(c.url, c.body);
      expect(metrics.logEvent).toHaveBeenCalledExactlyOnceWith({ group: c.group, attributes: c.attributes });
    });
  }

  it('reads a multipart body field', () => {
    const form = new FormData();
    form.append('media_type', 'audio');
    post('/api/dataset/import-local-folder', form);
    expect(metrics.logEvent).toHaveBeenCalledExactlyOnceWith({
      group: 'datasetImported',
      attributes: { importer: 'local_folder', media_type: 'audio' },
    });
  });

  it('decodes a percent-encoded plugin name', () => {
    post('/api/dataset/import/my%20importer', {});
    expect(metrics.logEvent).toHaveBeenCalledWith({ group: 'datasetImported', attributes: { importer: 'my importer' } });
  });

  it('logs a dataset download on GET', () => {
    http.get('/api/dataset/export', { responseType: 'blob' }).subscribe();
    httpMock.expectOne('/api/dataset/export').flush(new Blob(['x']));
    expect(metrics.logEvent).toHaveBeenCalledExactlyOnceWith({ group: 'datasetExported', attributes: {} });
  });

  it('does not log a tracked call that fails', () => {
    http.post('/api/exporters/export', { exporter_name: 'csv' }).subscribe({ error: () => {} });
    httpMock.expectOne('/api/exporters/export').flush({ message: 'boom' }, { status: 400, statusText: 'Bad Request' });
    expect(metrics.logEvent).not.toHaveBeenCalled();
  });

  it('ignores the wrong method on a tracked path', () => {
    http.get('/api/exporters/export').subscribe();
    httpMock.expectOne('/api/exporters/export').flush({});
    expect(metrics.logEvent).not.toHaveBeenCalled();
  });

  it('ignores untracked endpoints, including the metrics route itself', () => {
    post('/api/metrics', { events: [] });
    post('/api/medias/1/vote', {});
    expect(metrics.logEvent).not.toHaveBeenCalled();
  });

  it('matches a tracked path with a query string but not a superstring of it', () => {
    post('/api/find?dry=1', {});
    post('/api/find/cancel', {});
    expect(metrics.logEvent).toHaveBeenCalledExactlyOnceWith({ group: 'find', attributes: {} });
  });
});
