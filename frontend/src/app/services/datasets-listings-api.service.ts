import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { DemoCategoriesResponse } from '../generated/api-client/models/demo-categories-response';
import type { DemoDatasetListResponse } from '../generated/api-client/models/demo-dataset-list-response';
import type {
  ClipperInfo,
  ConverterInfo,
  EmbedderInfo,
  MediaTypeInfo,
} from '../models/api.models';
import { clippersList } from '../generated/api-client/fn/datasets-listings/clippers-list';
import { convertersList } from '../generated/api-client/fn/datasets-listings/converters-list';
import { demoDatasetCategories } from '../generated/api-client/fn/datasets-ui/demo-dataset-categories';
import { demoDatasetList } from '../generated/api-client/fn/datasets-ui/demo-dataset-list';
import { embeddersList } from '../generated/api-client/fn/datasets-listings/embedders-list';
import { mediaTypesList } from '../generated/api-client/fn/datasets-listings/media-types-list';

/** Plugin / media-type listings used by import-config widgets,
 *  settings panels, and the demo dataset picker.  Each listing returns
 *  plugin ``to_dict()`` payloads cast at the boundary to the richer
 *  ``ClipperInfo`` / ``EmbedderInfo`` / ``ConverterInfo`` /
 *  ``MediaTypeInfo`` interfaces in ``frontend/src/app/models/api.models.ts``. */
@Injectable({ providedIn: 'root' })
export class DatasetsListingsApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  getMediaTypes(): Observable<{ media_types: MediaTypeInfo[] }> {
    return mediaTypesList(this.http, this.config.rootUrl).pipe(
      map((r) => r.body as unknown as { media_types: MediaTypeInfo[] }),
    );
  }

  getClippers(mediaType?: string): Observable<ClipperInfo[]> {
    return clippersList(this.http, this.config.rootUrl, { media_type: mediaType }).pipe(
      map((r) => r.body.clippers as unknown as ClipperInfo[]),
    );
  }

  getEmbedders(mediaType?: string): Observable<EmbedderInfo[]> {
    return embeddersList(this.http, this.config.rootUrl, { media_type: mediaType }).pipe(
      map((r) => r.body.embedders as unknown as EmbedderInfo[]),
    );
  }

  getConverters(target?: string): Observable<ConverterInfo[]> {
    return convertersList(this.http, this.config.rootUrl, { target }).pipe(
      map((r) => r.body.converters as unknown as ConverterInfo[]),
    );
  }

  getDemoList(embedder?: string, clipper?: string): Observable<DemoDatasetListResponse> {
    return demoDatasetList(this.http, this.config.rootUrl, { embedder, clipper }).pipe(
      map((r) => r.body),
    );
  }

  getDemoCategories(name: string): Observable<DemoCategoriesResponse> {
    return demoDatasetCategories(this.http, this.config.rootUrl, { name }).pipe(
      map((r) => r.body),
    );
  }
}
