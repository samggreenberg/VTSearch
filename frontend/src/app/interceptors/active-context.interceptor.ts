import { HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { ActiveContextService } from '../services/active-context.service';

/**
 * Attaches `X-Dataset-Id` and `X-Model-Id` headers to every outgoing
 * HTTP request so the backend resolves the correct dataset/model
 * context per-request.
 *
 * Headers are only added when the corresponding ID is non-empty.
 */
export const activeContextInterceptor: HttpInterceptorFn = (req, next) => {
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
    headers = headers.set('X-Model-Id', modelId);
  }

  return next(req.clone({ headers }));
};
