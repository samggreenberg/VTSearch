import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { MediaIdsListResponse } from '../generated/api-client/models/media-ids-list-response';
import type { MediaBatchResponse } from '../generated/api-client/models/media-batch-response';
import type { MediaParagraphResponse } from '../generated/api-client/models/media-paragraph-response';
import type { MediaVoteRequest } from '../generated/api-client/models/media-vote-request';
import type { MediaVoteResponse } from '../generated/api-client/models/media-vote-response';
import type { MediaAddToPileResponse } from '../generated/api-client/models/media-add-to-pile-response';
import { listMediaIds } from '../generated/api-client/fn/medias/list-media-ids';
import { batchMedias } from '../generated/api-client/fn/medias/batch-medias';
import { mediaParagraphGet2 } from '../generated/api-client/fn/medias/media-paragraph-get-2';
import { voteMedia } from '../generated/api-client/fn/medias/vote-media';

@Injectable({ providedIn: 'root' })
export class MediasApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  /**
   * Lightweight listing of every media in the loaded dataset.  Returns
   * only ``id``, ``type``, and (optionally) ``embedder`` - the rest of the
   * metadata is fetched on demand via {@link getMediasBatch}.
   */
  getMediaIds(): Observable<MediaIdsListResponse[]> {
    return listMediaIds(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  getMediasBatch(ids: number[]): Observable<MediaBatchResponse[]> {
    return batchMedias(this.http, this.config.rootUrl, { body: { ids } }).pipe(map((r) => r.body));
  }

  getText(id: number): Observable<MediaParagraphResponse> {
    return mediaParagraphGet2(this.http, this.config.rootUrl, { media_id: id }).pipe(map((r) => r.body));
  }

  /**
   * Set the absolute vote state for a media item.
   *
   * ``target`` is the post-call state, not a "click direction" - the server
   * applies it idempotently (so concurrent stale-view tabs no longer race
   * the achievement counter, logical-bug-audit H1).  Callers that want the
   * old "click good toggles good off" behaviour should compute the toggle
   * locally (e.g. {@link VoteStateService.vote}) before invoking this method.
   *
   * Returns the server-confirmed new state and click-time so the optimistic
   * local view can be reconciled directly from the response without a
   * follow-up ``GET /api/votes``.
   */
  vote(
    id: number,
    target: 'good' | 'bad' | 'none',
    regionBox?: readonly number[] | null,
  ): Observable<MediaVoteResponse> {
    const body: MediaVoteRequest = { target };
    if (regionBox && regionBox.length === 4) body.region_box = [...regionBox];
    return voteMedia(this.http, this.config.rootUrl, { media_id: id, body }).pipe(
      map((r) => r.body),
    );
  }

  /** Multipart upload - stays on plain HttpClient because ng-openapi-gen
   *  doesn't model multipart bodies (the generated function's ``$Params``
   *  has no ``body`` field at all). */
  addToPile(file: File, label: 'good' | 'bad'): Observable<MediaAddToPileResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('label', label);
    return this.http.post<MediaAddToPileResponse>('/api/medias/add-to-pile', formData);
  }
}
