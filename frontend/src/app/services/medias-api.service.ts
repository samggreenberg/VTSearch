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
import { apiMediasIdsGet } from '../generated/api-client/fn/medias/api-medias-ids-get';
import { apiMediasBatchPost } from '../generated/api-client/fn/medias/api-medias-batch-post';
import { apiMediasMediaIdTextGet } from '../generated/api-client/fn/medias/api-medias-media-id-text-get';
import { apiMediasMediaIdVotePost } from '../generated/api-client/fn/medias/api-medias-media-id-vote-post';

@Injectable({ providedIn: 'root' })
export class MediasApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  /**
   * Lightweight listing of every media in the loaded dataset.  Returns
   * only ``id``, ``type``, and (optionally) ``embedder`` — the rest of the
   * metadata is fetched on demand via {@link getMediasBatch}.
   */
  getMediaIds(): Observable<MediaIdsListResponse[]> {
    return apiMediasIdsGet(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  getMediasBatch(ids: number[]): Observable<MediaBatchResponse[]> {
    return apiMediasBatchPost(this.http, this.config.rootUrl, { body: { ids } }).pipe(map((r) => r.body));
  }

  /** Binary stream — stays on plain HttpClient because ng-openapi-gen
   *  doesn't model binary response bodies usefully (the generated function
   *  declares the success body as ``Error`` because the spec only carries
   *  error responses for these routes). */
  getAudio(id: number): Observable<Blob> {
    return this.http.get(`/api/medias/${id}/audio`, { responseType: 'blob' });
  }

  /** Binary stream — see {@link getAudio}. */
  getVideo(id: number): Observable<Blob> {
    return this.http.get(`/api/medias/${id}/video`, { responseType: 'blob' });
  }

  /** Binary stream — see {@link getAudio}. */
  getImage(id: number): Observable<Blob> {
    return this.http.get(`/api/medias/${id}/image`, { responseType: 'blob' });
  }

  getText(id: number): Observable<MediaParagraphResponse> {
    return apiMediasMediaIdTextGet(this.http, this.config.rootUrl, { media_id: id }).pipe(map((r) => r.body));
  }

  /** Binary stream — see {@link getAudio}. */
  getMedia(id: number): Observable<Blob> {
    return this.http.get(`/api/medias/${id}/media`, { responseType: 'blob' });
  }

  vote(
    id: number,
    label: 'good' | 'bad',
    regionBox?: readonly number[] | null,
  ): Observable<MediaVoteResponse> {
    const body: MediaVoteRequest = { vote: label };
    if (regionBox && regionBox.length === 4) body.region_box = [...regionBox];
    return apiMediasMediaIdVotePost(this.http, this.config.rootUrl, { media_id: id, body }).pipe(
      map((r) => r.body),
    );
  }

  /** Multipart upload — stays on plain HttpClient because ng-openapi-gen
   *  doesn't model multipart bodies (the generated function's ``$Params``
   *  has no ``body`` field at all). */
  addToPile(file: File, label: 'good' | 'bad'): Observable<MediaAddToPileResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('label', label);
    return this.http.post<MediaAddToPileResponse>('/api/medias/add-to-pile', formData);
  }
}
