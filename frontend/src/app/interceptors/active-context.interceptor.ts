import { HttpContextToken, HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { ActiveContextService } from '../services/active-context.service';

/**
 * Send one request to *this* dataset instead of the active one, with no
 * detector.
 *
 * For reading an item of a dataset that is not the active one: Show in photo
 * (#4750) fetches a face crop's photo out of the Image dataset its import
 * produced while the Face pair stays active. The detector header is dropped,
 * not kept: a request pairing the active detector with another dataset makes
 * the backend rehydrate that detector's votes against it
 * (`ensure_votes_match_active_dataset`), which would wipe the session the user
 * is in. `ActiveContextService.mediaUrlIn` is the `<img src>` counterpart.
 *
 *     http.post(url, body, {
 *       context: new HttpContext().set(DATASET_OVERRIDE, siblingId),
 *     });
 */
export const DATASET_OVERRIDE = new HttpContextToken<string>(() => '');

/**
 * Attaches `X-Dataset-Id` and `X-Detector-Id` headers to every outgoing
 * HTTP request so the backend resolves the correct dataset/model
 * context per-request.
 *
 * Headers are only added when the corresponding ID is non-empty. A request
 * carrying {@link DATASET_OVERRIDE} names that dataset and no detector.
 */
export const activeContextInterceptor: HttpInterceptorFn = (req, next) => {
  const override = req.context.get(DATASET_OVERRIDE);
  if (override) {
    return next(req.clone({ headers: req.headers.set('X-Dataset-Id', override).delete('X-Detector-Id') }));
  }

  const ctx = inject(ActiveContextService);

  const datasetId = ctx.datasetId;
  const modelId = ctx.modelId;

  if (!datasetId && !modelId) {
    return next(req);
  }

  let headers = req.headers;
  if (datasetId) {
    headers = headers.set('X-Dataset-Id', datasetId);
  }
  if (modelId) {
    headers = headers.set('X-Detector-Id', modelId);
  }

  return next(req.clone({ headers }));
};
