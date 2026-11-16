import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpContext } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import { listBlurredMedia } from '../generated/api-client/fn/medias/list-blurred-media';
import { setMediaBlur } from '../generated/api-client/fn/medias/set-media-blur';
import type { MediaBlurResponse } from '../generated/api-client/models/media-blur-response';
import { DATASET_OVERRIDE } from '../interceptors/active-context.interceptor';
import { SKIP_ERROR_TOAST } from '../interceptors/error.interceptor';

/**
 * The two blur routes (#5689), always aimed at a named dataset.
 *
 * Both calls carry {@link DATASET_OVERRIDE}, the active dataset included: blur
 * state belongs to a dataset and not to a detector, several image sites show a
 * sibling dataset's items (a face's source photo), and a request that paired
 * the active detector with another dataset would rehydrate that detector's
 * votes against it. With the override there is one path for every dataset.
 */
@Injectable({ providedIn: 'root' })
export class BlurApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  /**
   * Ascending ids of *datasetId*'s items that show blurred. A background read,
   * polled for every dataset on screen, so a failure is not toasted:
   * `BlurStateService` keeps what it had and asks again on its next tick.
   */
  getBlurredIn(datasetId: string): Observable<number[]> {
    const context = new HttpContext().set(DATASET_OVERRIDE, datasetId).set(SKIP_ERROR_TOAST, true);
    return listBlurredMedia(this.http, this.config.rootUrl, undefined, context).pipe(map((r) => r.body.ids));
  }

  /** Record *blurred* as item *id*'s Blur / Unblur override in *datasetId*. */
  setBlurredIn(datasetId: string, id: number, blurred: boolean): Observable<MediaBlurResponse> {
    const context = new HttpContext().set(DATASET_OVERRIDE, datasetId);
    return setMediaBlur(this.http, this.config.rootUrl, { media_id: id, body: { blurred } }, context).pipe(
      map((r) => r.body),
    );
  }
}
