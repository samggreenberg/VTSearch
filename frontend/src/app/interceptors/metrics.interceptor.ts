import { HttpInterceptorFn, HttpRequest, HttpResponse } from '@angular/common/http';
import { inject } from '@angular/core';
import { tap } from 'rxjs/operators';
import { MetricsAttributeValue, MetricsService } from '../services/metrics.service';

type Attributes = Record<string, MetricsAttributeValue | undefined>;

/** One action endpoint whose success is a usage event (#5475). */
interface TrackedRoute {
  method: 'GET' | 'POST';
  /** Anchored, matched against the path without its query string. */
  pattern: RegExp;
  group: string;
  /** Event details from the request; `null` / `undefined` entries are dropped. */
  attributes?: (req: HttpRequest<unknown>, match: RegExpMatchArray) => Attributes;
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

/** The plugin name a route carries in its path. */
const pathName = (index: number) => (_req: HttpRequest<unknown>, match: RegExpMatchArray) =>
  decodeURIComponent(match[index]);

const datasetImport =
  (importer: (req: HttpRequest<unknown>, match: RegExpMatchArray) => string) =>
  (req: HttpRequest<unknown>, match: RegExpMatchArray): Attributes => ({
    importer: importer(req, match),
    media_type: bodyField(req, 'media_type'),
  });

/**
 * The action endpoints counted as usage events, one row per route. A row
 * fires once the request succeeds, so a rejected import or export is not
 * counted. Several routes can share a group (every way of adding a dataset is
 * a `datasetImported`, told apart by `importer`).
 */
const TRACKED_ROUTES: TrackedRoute[] = [
  {
    method: 'POST',
    pattern: /^\/api\/dataset\/import\/([^/]+)$/,
    group: 'datasetImported',
    attributes: datasetImport(pathName(1)),
  },
  {
    method: 'POST',
    pattern: /^\/api\/dataset\/import-local-folder$/,
    group: 'datasetImported',
    attributes: datasetImport(() => 'local_folder'),
  },
  {
    method: 'POST',
    pattern: /^\/api\/dataset\/import-local-files$/,
    group: 'datasetImported',
    attributes: datasetImport(() => 'local_files'),
  },
  {
    method: 'POST',
    pattern: /^\/api\/dataset\/load-file$/,
    group: 'datasetImported',
    attributes: datasetImport(() => 'pickle_file'),
  },
  {
    method: 'POST',
    pattern: /^\/api\/dataset\/load-demo$/,
    group: 'datasetImported',
    attributes: (req) => ({ importer: 'demo', demo: bodyField(req, 'name') }),
  },
  { method: 'POST', pattern: /^\/api\/dataset\/combine$/, group: 'datasetsCombined' },
  { method: 'POST', pattern: /^\/api\/dataset\/promote$/, group: 'datasetPromoted' },
  { method: 'GET', pattern: /^\/api\/dataset\/export$/, group: 'datasetExported' },
  {
    method: 'POST',
    pattern: /^\/api\/exporters\/export$/,
    group: 'resultsExported',
    attributes: (req) => ({ exporter: bodyField(req, 'exporter_name'), payload: bodyField(req, 'payload_kind') }),
  },
  {
    method: 'POST',
    pattern: /^\/api\/label-importers\/import\/([^/]+)$/,
    group: 'labelsImported',
    attributes: (req, match) => ({ importer: pathName(1)(req, match), into: 'new_detector' }),
  },
  {
    method: 'POST',
    pattern: /^\/api\/detectors\/[^/]+\/import-labels\/([^/]+)$/,
    group: 'labelsImported',
    attributes: (req, match) => ({ importer: pathName(1)(req, match), into: 'existing_detector' }),
  },
  {
    method: 'POST',
    pattern: /^\/api\/detectors$/,
    group: 'detectorCreated',
    attributes: (req) => ({
      source: 'new_detector_form',
      media_type: bodyField(req, 'media_type'),
      embedder_type: bodyField(req, 'embedder_type'),
    }),
  },
  {
    method: 'POST',
    pattern: /^\/api\/detectors\/registry\/from-labelset\/([^/]+)$/,
    group: 'detectorCreated',
    attributes: (req, match) => ({ source: 'labelset', importer: pathName(1)(req, match) }),
  },
  { method: 'POST', pattern: /^\/api\/detectors\/[^/]+\/portable-bundle$/, group: 'detectorExported' },
  { method: 'POST', pattern: /^\/api\/find$/, group: 'find' },
  {
    method: 'POST',
    pattern: /^\/api\/settings-importers\/import\/([^/]+)$/,
    group: 'settingsImported',
    attributes: (req, match) => ({ importer: pathName(1)(req, match) }),
  },
  {
    method: 'POST',
    pattern: /^\/api\/settings-exporters\/export$/,
    group: 'settingsExported',
    attributes: (req) => ({ exporter: bodyField(req, 'exporter_name') }),
  },
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
  const path = req.url.split('?')[0];
  let route: TrackedRoute | undefined;
  let match: RegExpMatchArray | null = null;
  for (const candidate of TRACKED_ROUTES) {
    if (candidate.method !== req.method) continue;
    match = path.match(candidate.pattern);
    if (match) {
      route = candidate;
      break;
    }
  }
  if (!route || !match) {
    return next(req);
  }

  const metrics = inject(MetricsService);
  const attributes = compact(route.attributes?.(req, match) ?? {});
  const group = route.group;
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
