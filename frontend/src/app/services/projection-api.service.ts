import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpContext } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';
import { SKIP_ERROR_TOAST } from '../interceptors/error.interceptor';
import { ApiConfiguration } from '../generated/api-client/api-configuration';
import { buildProjection } from '../generated/api-client/fn/projection/build-projection';
import { getTile } from '../generated/api-client/fn/projection/get-tile';
import { projectionLabels } from '../generated/api-client/fn/projection/projection-labels';
import { projectionMeta } from '../generated/api-client/fn/projection/projection-meta';
import { removeFromSubset } from '../generated/api-client/fn/projection/remove-from-subset';
import type {
  ProjectionBuildResponse,
  ProjectionLabelsResponse,
  ProjectionMeta,
  TileResponse,
} from '../models/projection.models';

/**
 * The Browse projection endpoints.
 *
 * Every call takes its path from the generated client (`fn.PATH`) and returns
 * a generated type. Only `removeFromSubset` goes through its generated
 * function whole: the routes read the `?subset` / `?v` query flags and the
 * build's `{ids, force}` body straight off the request, so the spec declares
 * none of them and the generated functions cannot send them. Those calls are
 * built here, on the generated path.
 */
@Injectable({ providedIn: 'root' })
export class ProjectionApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

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
    return this.http.get<ProjectionMeta>(this.url(projectionMeta.PATH), {
      params: subsetParams(subset),
      context: new HttpContext().set(SKIP_ERROR_TOAST, true),
    });
  }

  build(): Observable<ProjectionBuildResponse> {
    return this.postBuild({});
  }

  /**
   * Build an ephemeral subset projection: UMAP the high-d vectors of just
   * `ids` (e.g. the positives of a Find run) to 2-D. Returns immediately with
   * a job id while the fit runs in the background; poll `getMeta(true)`.
   */
  buildSubset(ids: number[]): Observable<ProjectionBuildResponse> {
    return this.postBuild({ ids });
  }

  /**
   * Remove ids from the current subset projection in place — no UMAP re-fit.
   * The server re-bins the frozen layout minus those points and returns the
   * updated meta (same ``projection_id``, bumped ``content_version``).
   */
  subsetRemove(ids: number[]): Observable<ProjectionMeta> {
    return removeFromSubset(this.http, this.config.rootUrl, { body: { ids } }).pipe(map((r) => r.body));
  }

  /**
   * Re-fit UMAP over the whole dataset and replace the frozen layout with a
   * fresh arrangement (new ``projection_id``). Unlike {@link build}, ``force``
   * overrides the cached/persisted layout. Runs in the background; poll
   * `getMeta()`.
   */
  reproject(): Observable<ProjectionBuildResponse> {
    return this.postBuild({ force: true });
  }

  /**
   * Re-fit UMAP over just `ids` (the items currently in a subset browse) and
   * replace its layout with a fresh arrangement. Used to re-spread the
   * survivors after a cull, or to reshuffle. Runs in the background; poll
   * `getMeta(true)`.
   */
  reprojectSubset(ids: number[]): Observable<ProjectionBuildResponse> {
    return this.postBuild({ ids, force: true });
  }

  /**
   * Fetch the region signpost labels for the current projection — the named
   * regions the canvas renders as "street signs" over the map. Tiny payload
   * (one entry per named region), fetched once per projection id. A dataset
   * with no computed labels answers an empty list, not an error; the meta's
   * ``has_labels`` flag lets callers skip the request entirely.
   */
  getLabels(subset = false): Observable<ProjectionLabelsResponse> {
    // Signs are optional decoration on the map — a failure to fetch them must
    // never toast an error over a perfectly working browse view.
    return this.http.get<ProjectionLabelsResponse>(this.url(projectionLabels.PATH), {
      params: subsetParams(subset),
      context: new HttpContext().set(SKIP_ERROR_TOAST, true),
    });
  }

  getTile(
    level: number,
    tx: number,
    ty: number,
    subset = false,
    cacheToken = '',
  ): Observable<TileResponse> {
    // The backend resolves the pyramid (and its bin shape) from the active
    // dataset context, so the tile URL is keyed only by (level, tx, ty). Tiles
    // are served ``immutable``, so ``cacheToken``
    // (``<projection_id>:<content_version>``) rides along as ``?v=`` to give
    // each projection — and each in-place edit of a subset — a distinct URL,
    // busting the HTTP cache when content changes.
    const path = getTile.PATH.replace('{level}', String(level))
      .replace('{tx}', String(tx))
      .replace('{ty}', String(ty));
    const params = subsetParams(subset);
    if (cacheToken) params['v'] = cacheToken;
    // Suppress the global error toast: a tile can 404 transiently when it is
    // requested before the projection has finished building. The caller
    // (TileCacheService) already recovers by rendering an empty tile and
    // refetching once the build lands, so the failure is expected and
    // self-healing — surfacing it as a banner would just alarm the user.
    // Mirrors getMeta's SKIP_ERROR_TOAST rationale.
    const context = new HttpContext().set(SKIP_ERROR_TOAST, true);
    return this.http.get<TileResponse>(this.url(path), { params, context });
  }

  /** `POST /api/projection/build`: `ids` scopes it to a subset, `force` re-fits a cached layout. */
  private postBuild(body: { ids?: number[]; force?: boolean }): Observable<ProjectionBuildResponse> {
    return this.http.post<ProjectionBuildResponse>(this.url(buildProjection.PATH), body);
  }

  private url(path: string): string {
    return this.config.rootUrl + path;
  }
}

/** `?subset=1` for the ephemeral subset projection, nothing for the full-dataset one. */
function subsetParams(subset: boolean): Record<string, string> {
  return subset ? { subset: '1' } : {};
}
