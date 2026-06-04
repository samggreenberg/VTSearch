import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpContext } from '@angular/common/http';
import { Observable } from 'rxjs';
import { SKIP_ERROR_TOAST } from '../interceptors/error.interceptor';
import type {
  BinShape,
  ProjectionMeta,
  ProjectionBuildResponse,
  TilePayload,
} from '../models/projection.models';

@Injectable({ providedIn: 'root' })
export class ProjectionApiService {
  private http = inject(HttpClient);

  getMeta(shape: BinShape, subset = false): Observable<ProjectionMeta> {
    // The browse view polls this to discover whether a projection exists
    // yet for the requested bin shape. A not-loaded / not-built dataset
    // answers 404/409, which the caller handles inline (→ "empty"/Build
    // affordance). Suppress the global error toast so that expected state
    // doesn't alarm the user.
    //
    // `subset=true` targets the ephemeral UMAP fit over a subset of the
    // dataset's media (e.g. the positives of a Find run) rather than the
    // full-dataset projection.
    const params: Record<string, string> = { shape };
    if (subset) params['subset'] = '1';
    return this.http.get<ProjectionMeta>('/api/projection/meta', {
      params,
      context: new HttpContext().set(SKIP_ERROR_TOAST, true),
    });
  }

  build(shape: BinShape): Observable<ProjectionBuildResponse> {
    return this.http.post<ProjectionBuildResponse>('/api/projection/build', { shape });
  }

  /**
   * Build an ephemeral subset projection: UMAP the high-d vectors of just
   * `ids` (e.g. the positives of a Find run) to 2-D. Returns immediately with
   * a job id while the fit runs in the background; poll `getMeta(shape, true)`.
   */
  buildSubset(shape: BinShape, ids: number[]): Observable<ProjectionBuildResponse> {
    return this.http.post<ProjectionBuildResponse>('/api/projection/build', { shape, ids });
  }

  getTile(
    shape: BinShape,
    level: number,
    tx: number,
    ty: number,
    subset = false,
  ): Observable<TilePayload> {
    // The backend resolves the pyramid from the active dataset context, so the
    // tile URL is keyed by (shape, level, tx, ty); the projection id is tracked
    // client-side (TileCacheService) for cache invalidation, not in the path.
    const url = `/api/projection/tiles/${shape}/${level}/${tx}/${ty}`;
    if (subset) {
      return this.http.get<TilePayload>(url, { params: { subset: '1' } });
    }
    return this.http.get<TilePayload>(url);
  }
}
