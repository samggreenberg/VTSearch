import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpContext } from '@angular/common/http';
import { Observable } from 'rxjs';
import { SKIP_ERROR_TOAST } from '../interceptors/error.interceptor';
import type {
  ProjectionMeta,
  ProjectionBuildResponse,
  ProjectionLabelsResponse,
  TilePayload,
} from '../models/projection.models';

@Injectable({ providedIn: 'root' })
export class ProjectionApiService {
  private http = inject(HttpClient);

  getMeta(subset = false): Observable<ProjectionMeta> {
    // The browse view polls this to discover whether a projection exists
    // yet. A not-loaded / not-built dataset answers 404/409, which the caller
    // handles inline (→ "empty"/Build affordance). Suppress the global error
    // toast so that expected state doesn't alarm the user.
    //
    // `subset=true` targets the ephemeral UMAP fit over a subset of the
    // dataset's media (e.g. the positives of a Find run) rather than the
    // full-dataset projection. The bin shape (hex/square) is derived
    // server-side from the dataset's media type and reported back in
    // `bin_shape`; the client never selects it.
    const params: Record<string, string> = {};
    if (subset) params['subset'] = '1';
    return this.http.get<ProjectionMeta>('/api/projection/meta', {
      params,
      context: new HttpContext().set(SKIP_ERROR_TOAST, true),
    });
  }

  build(): Observable<ProjectionBuildResponse> {
    return this.http.post<ProjectionBuildResponse>('/api/projection/build', {});
  }

  /**
   * Build an ephemeral subset projection: UMAP the high-d vectors of just
   * `ids` (e.g. the positives of a Find run) to 2-D. Returns immediately with
   * a job id while the fit runs in the background; poll `getMeta(true)`.
   */
  buildSubset(ids: number[]): Observable<ProjectionBuildResponse> {
    return this.http.post<ProjectionBuildResponse>('/api/projection/build', { ids });
  }

  /**
   * Remove ids from the current subset projection in place — no UMAP re-fit.
   * The server re-bins the frozen layout minus those points and returns the
   * updated meta (same ``projection_id``, bumped ``content_version``).
   */
  subsetRemove(ids: number[]): Observable<ProjectionMeta> {
    return this.http.post<ProjectionMeta>('/api/projection/subset/remove', { ids });
  }

  /**
   * Re-fit UMAP over the whole dataset and replace the frozen layout with a
   * fresh arrangement (new ``projection_id``). Unlike {@link build}, ``force``
   * overrides the cached/persisted layout. Runs in the background; poll
   * `getMeta()`.
   */
  reproject(): Observable<ProjectionBuildResponse> {
    return this.http.post<ProjectionBuildResponse>('/api/projection/build', { force: true });
  }

  /**
   * Re-fit UMAP over just `ids` (the items currently in a subset browse) and
   * replace its layout with a fresh arrangement. Used to re-spread the
   * survivors after a cull, or to reshuffle. Runs in the background; poll
   * `getMeta(true)`.
   */
  reprojectSubset(ids: number[]): Observable<ProjectionBuildResponse> {
    return this.http.post<ProjectionBuildResponse>('/api/projection/build', { ids, force: true });
  }

  /**
   * Fetch the region signpost labels for the current projection — the named
   * regions the canvas renders as "street signs" over the map. Tiny payload
   * (one entry per named region), fetched once per projection id. A dataset
   * with no computed labels answers an empty list, not an error; the meta's
   * ``has_labels`` flag lets callers skip the request entirely.
   */
  getLabels(subset = false): Observable<ProjectionLabelsResponse> {
    const params: Record<string, string> = {};
    if (subset) params['subset'] = '1';
    // Signs are optional decoration on the map — a failure to fetch them must
    // never toast an error over a perfectly working browse view.
    return this.http.get<ProjectionLabelsResponse>('/api/projection/labels', {
      params,
      context: new HttpContext().set(SKIP_ERROR_TOAST, true),
    });
  }

  getTile(
    level: number,
    tx: number,
    ty: number,
    subset = false,
    cacheToken = '',
  ): Observable<TilePayload> {
    // The backend resolves the pyramid (and its bin shape) from the active
    // dataset context, so the tile URL is keyed only by (level, tx, ty). Tiles
    // are served ``immutable``, so ``cacheToken``
    // (``<projection_id>:<content_version>``) rides along as ``?v=`` to give
    // each projection — and each in-place edit of a subset — a distinct URL,
    // busting the HTTP cache when content changes.
    const url = `/api/projection/tiles/${level}/${tx}/${ty}`;
    const params: Record<string, string> = {};
    if (subset) params['subset'] = '1';
    if (cacheToken) params['v'] = cacheToken;
    // Suppress the global error toast: a tile can 404 transiently when it is
    // requested before the projection has finished building. The caller
    // (TileCacheService) already recovers by rendering an empty tile and
    // refetching once the build lands, so the failure is expected and
    // self-healing — surfacing it as a banner would just alarm the user.
    // Mirrors getMeta's SKIP_ERROR_TOAST rationale.
    const context = new HttpContext().set(SKIP_ERROR_TOAST, true);
    return this.http.get<TilePayload>(url, { params, context });
  }
}
