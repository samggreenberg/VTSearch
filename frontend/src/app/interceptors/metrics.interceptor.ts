import { HttpInterceptorFn, HttpRequest, HttpResponse } from '@angular/common/http';
import { inject } from '@angular/core';
import { tap } from 'rxjs/operators';
import { MetricsAttributeValue, MetricsService } from '../services/metrics.service';

type Attributes = Record<string, MetricsAttributeValue | undefined>;
type AttributesFn = (req: HttpRequest<unknown>, match: RegExpMatchArray) => Attributes;

/** One action endpoint whose success is a usage event (#5475). */
interface TrackedRoute {
  method: 'GET' | 'POST';
  /** Anchored, matched against the path after `/api/`, without its query string. */
  pattern: RegExp;
  group: string;
  /** Event details from the request; `null` / `undefined` entries are dropped. */
  attributes?: AttributesFn;
}

/** A scalar field of a JSON or multipart request body, if present. */
function bodyField(req: HttpRequest<unknown>, key: string): MetricsAttributeValue | undefined {
  const body = req.body;
  if (body instanceof FormData) {
    const value = body.get(key);
    return typeof value === 'string' ? value : undefined;
  }
  if (body && typeof body === 'object') {
    const value = (body as Record<string, unknown>)[key];
    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return value;
  }
  return undefined;
}

/** A row of {@link TRACKED_ROUTES}; a helper so each row stays one line. */
function route(pattern: RegExp, group: string, attributes?: AttributesFn, method: TrackedRoute['method'] = 'POST'): TrackedRoute {
  return { method, pattern, group, attributes };
}

/** The plugin name a route carries in its path (its first capture group). */
const pathName = (match: RegExpMatchArray) => decodeURIComponent(match[1]);

/** Every way of adding a dataset is a `datasetImported`, told apart by `importer`. */
const datasetImport =
  (importer?: string): AttributesFn =>
  (req, match) => ({ importer: importer ?? pathName(match), media_type: bodyField(req, 'media_type') });

/**
 * The action endpoints counted as usage events, one row per route. A row
 * fires once the request succeeds, so a rejected import or export is not
 * counted.
 */
const TRACKED_ROUTES: TrackedRoute[] = [
  route(/^dataset\/import\/([^/]+)$/, 'datasetImported', datasetImport()),
  route(/^dataset\/import-local-folder$/, 'datasetImported', datasetImport('local_folder')),
  route(/^dataset\/import-local-files$/, 'datasetImported', datasetImport('local_files')),
  route(/^dataset\/load-file$/, 'datasetImported', datasetImport('pickle_file')),
  route(/^dataset\/load-demo$/, 'datasetImported', (req) => ({ importer: 'demo', demo: bodyField(req, 'name') })),
  route(/^dataset\/combine$/, 'datasetsCombined'),
  route(/^dataset\/promote$/, 'datasetPromoted'),
  route(/^dataset\/export$/, 'datasetExported', undefined, 'GET'),
  route(/^exporters\/export$/, 'resultsExported', (req) => ({
    exporter: bodyField(req, 'exporter_name'),
    payload: bodyField(req, 'payload_kind'),
  })),
  route(/^label-importers\/import\/([^/]+)$/, 'labelsImported', (_req, match) => ({
    importer: pathName(match),
    into: 'new_detector',
  })),
  route(/^detectors\/[^/]+\/import-labels\/([^/]+)$/, 'labelsImported', (_req, match) => ({
    importer: pathName(match),
    into: 'existing_detector',
  })),
  route(/^detectors$/, 'detectorCreated', (req) => ({
    source: 'new_detector_form',
    media_type: bodyField(req, 'media_type'),
    embedder_type: bodyField(req, 'embedder_type'),
  })),
  route(/^detectors\/registry\/from-labelset\/([^/]+)$/, 'detectorCreated', (_req, match) => ({
    source: 'labelset',
    importer: pathName(match),
  })),
  route(/^detectors\/[^/]+\/portable-bundle$/, 'detectorExported'),
  route(/^find$/, 'find'),
  route(/^settings-importers\/import\/([^/]+)$/, 'settingsImported', (_req, match) => ({
    importer: pathName(match),
  })),
  route(/^settings-exporters\/export$/, 'settingsExported', (req) => ({
    exporter: bodyField(req, 'exporter_name'),
  })),
];

function compact(attributes: Attributes): Record<string, MetricsAttributeValue> {
  const out: Record<string, MetricsAttributeValue> = {};
  for (const [key, value] of Object.entries(attributes)) {
    if (value !== undefined && value !== null) out[key] = value;
  }
  return out;
}

/**
 * Logs a usage event (#5475) for each successful call to an action endpoint
 * in {@link TRACKED_ROUTES}: imports, exports, detector creation, Find. Doing
 * it here rather than in each `*-api.service.ts` keeps those wrappers thin and
 * counts every caller of a route, whichever component made the call.
 *
 * UI-only events (views, settings changes, clicks) are logged where they
 * happen; see `MetricsService`.
 */
export const metricsInterceptor: HttpInterceptorFn = (req, next) => {
  const url = req.url.split('?')[0];
  if (!url.startsWith('/api/')) {
    return next(req);
  }
  const path = url.slice('/api/'.length);
  let tracked: TrackedRoute | undefined;
  let match: RegExpMatchArray | null = null;
  for (const candidate of TRACKED_ROUTES) {
    if (candidate.method !== req.method) continue;
    match = path.match(candidate.pattern);
    if (match) {
      tracked = candidate;
      break;
    }
  }
  if (!tracked || !match) {
    return next(req);
  }

  const metrics = inject(MetricsService);
  const attributes = compact(tracked.attributes?.(req, match) ?? {});
  const group = tracked.group;
  return next(req).pipe(
    tap({
      next: (event) => {
        if (event instanceof HttpResponse && event.status >= 200 && event.status < 300) {
          metrics.logEvent({ group, attributes });
        }
      },
    }),
  );
};
