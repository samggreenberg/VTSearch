import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpContext } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { MediaIdsListResponse } from '../generated/api-client/models/media-ids-list-response';
import type { MediaBatchResponse } from '../generated/api-client/models/media-batch-response';
import type { MediaParagraphResponse } from '../generated/api-client/models/media-paragraph-response';
import type { MediaVoteRequest } from '../generated/api-client/models/media-vote-request';
import type { MediaVoteResponse } from '../generated/api-client/models/media-vote-response';
import type { MediaAddToPileResponse } from '../generated/api-client/models/media-add-to-pile-response';
import type { MediaSourceResponse } from '../generated/api-client/models/media-source-response';
import { DATASET_OVERRIDE } from '../interceptors/active-context.interceptor';
import type { PayloadVariant } from '../models/api.models';
import type { VoteProvenance } from './vote-provenance.service';
import { listMediaIds } from '../generated/api-client/fn/medias/list-media-ids';
import { batchMedias } from '../generated/api-client/fn/medias/batch-medias';
import { mediaParagraphGet2 } from '../generated/api-client/fn/medias/media-paragraph-get-2';
import { voteMedia } from '../generated/api-client/fn/medias/vote-media';
import { mediaSource } from '../generated/api-client/fn/medias/media-source';

@Injectable({ providedIn: 'root' })
export class MediasApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  /**
   * Lightweight listing of every media in the loaded dataset.  Returns
   * only ``id``, ``type``, and (optionally) ``embedder``; the rest of the
   * metadata is fetched on demand via {@link getMediasBatch}.
   */
  getMediaIds(): Observable<MediaIdsListResponse[]> {
    return listMediaIds(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  getMediasBatch(ids: number[]): Observable<MediaBatchResponse[]> {
    return batchMedias(this.http, this.config.rootUrl, { body: { ids } }).pipe(map((r) => r.body));
  }

  /**
   * {@link getMediasBatch} out of *datasetId* rather than the active dataset,
   * with no detector (`DATASET_OVERRIDE`): Show in photo reads the photo's
   * metadata from the Face dataset's Image sibling (#4750).
   */
  getMediasBatchIn(datasetId: string, ids: number[]): Observable<MediaBatchResponse[]> {
    const context = new HttpContext().set(DATASET_OVERRIDE, datasetId);
    return batchMedias(this.http, this.config.rootUrl, { body: { ids } }, context).pipe(map((r) => r.body));
  }

  /**
   * The item a converter output was made from, in its sibling dataset, and the
   * output's box in it (#4749). Answers 404 when there is none and 409 when the
   * sibling is not loaded; `SourcePhotoService.lookup` reads both as answers.
   */
  getSource(id: number, context?: HttpContext): Observable<MediaSourceResponse> {
    return mediaSource(this.http, this.config.rootUrl, { media_id: id }, context).pipe(map((r) => r.body));
  }

  /** Text content of a text media.  `variant: 'original'` returns the
   *  pre-clean payload of an item a cleaner rewrote at load time. */
  getText(id: number, variant: PayloadVariant = ''): Observable<MediaParagraphResponse> {
    // Send the param only when it means something: an empty `variant=` in the
    // URL is what the server already does by default, and omitting it keeps the
    // canonical request byte-identical to what it was before cleaners existed.
    return mediaParagraphGet2(this.http, this.config.rootUrl, {
      media_id: id,
      ...(variant ? { variant } : {}),
    }).pipe(map((r) => r.body));
  }

  /**
   * Set the absolute vote state for a media item.
   *
   * ``target`` is the post-call state, not a "click direction"; the server
   * applies it idempotently (so concurrent stale-view tabs no longer race
   * the achievement counter, logical-bug-audit H1).  Callers that want the
   * old "click good toggles good off" behaviour should compute the toggle
   * locally (e.g. {@link VoteStateService.vote}) before invoking this method.
   *
   * Returns the server-confirmed new state and click-time so the optimistic
   * local view can be reconciled directly from the response without a
   * follow-up ``GET /api/votes``.
   *
   * ``provenance`` records how the item was surfaced. The server stores it
   * only when the call actually changes the vote state, so sending it on a
   * call that turns out to be idempotent cannot overwrite what the original
   * click recorded.
   */
  vote(
    id: number,
    target: 'good' | 'bad' | 'none',
    regionBox?: readonly number[] | null,
    provenance?: VoteProvenance | null,
  ): Observable<MediaVoteResponse> {
    const body: MediaVoteRequest = { target };
    if (regionBox && regionBox.length === 4) body.region_box = [...regionBox];
    if (provenance) body.provenance = provenance;
    return voteMedia(this.http, this.config.rootUrl, { media_id: id, body }).pipe(
      map((r) => r.body),
    );
  }

  /**
   * Apply one absolute vote target to many medias in a single request.
   *
   * Mirrors {@link vote} for a batch (idempotent, image-level — no region
   * boxes), persisting the detector labelset once server-side.  Used by the
   * Browser's "Verified Good" / "Verified Bad" actions.  Stays on plain HttpClient (like
   * {@link addToPile}) because the bulk route isn't modelled in the generated
   * client; the active-context interceptor still attaches the dataset /
   * detector headers the route requires.
   */
  voteBulk(
    ids: number[],
    target: 'good' | 'bad' | 'none',
    provenance?: VoteProvenance | null,
  ): Observable<{ ok: boolean; changed: number; missing: number[] }> {
    return this.http.post<{ ok: boolean; changed: number; missing: number[] }>(
      '/api/medias/vote-bulk',
      { ids, target, ...(provenance ? { provenance } : {}) },
    );
  }

  /** Multipart upload; stays on plain HttpClient because ng-openapi-gen
   *  doesn't model multipart bodies (the generated function's ``$Params``
   *  has no ``body`` field at all). */
  addToPile(file: File, label: 'good' | 'bad'): Observable<MediaAddToPileResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('label', label);
    return this.http.post<MediaAddToPileResponse>('/api/medias/add-to-pile', formData);
  }
}
