import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { DemoCategoriesResponse } from '../generated/api-client/models/demo-categories-response';
import type { DemoDatasetListResponse } from '../generated/api-client/models/demo-dataset-list-response';
import type {
  CleanerInfo,
  ClipperInfo,
  ConverterInfo,
  EmbedderInfo,
  MediaTypeInfo,
} from '../models/api.models';
import { cleanersList } from '../generated/api-client/fn/datasets-listings/cleaners-list';
import { clippersList } from '../generated/api-client/fn/datasets-listings/clippers-list';
import { convertersList } from '../generated/api-client/fn/datasets-listings/converters-list';
import { demoDatasetCategories } from '../generated/api-client/fn/datasets-ui/demo-dataset-categories';
import { demoDatasetList } from '../generated/api-client/fn/datasets-ui/demo-dataset-list';
import { embeddersList } from '../generated/api-client/fn/datasets-listings/embedders-list';
import { mediaTypesList } from '../generated/api-client/fn/datasets-listings/media-types-list';

/** Plugin / media-type listings used by import-config widgets, settings
 *  panels, and the demo dataset picker.  Every payload here is described by a
 *  nested Marshmallow schema in ``vtsearch/schemas/datasets.py``, so the
 *  generated models are the real types and nothing is cast at the boundary:
 *  a backend field rename fails this build. */
@Injectable({ providedIn: 'root' })
export class DatasetsListingsApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  getMediaTypes(): Observable<{ media_types: MediaTypeInfo[] }> {
    return mediaTypesList(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  getClippers(mediaType?: string): Observable<ClipperInfo[]> {
    return clippersList(this.http, this.config.rootUrl, { media_type: mediaType }).pipe(
      map((r) => r.body.clippers),
    );
  }

  /** Cleanup gates registered for `mediaType` (all of them when omitted).
   *  Unlike clippers these are not a radio choice: the import form renders
   *  one checkbox per entry, pre-checked when `default_enabled` is set. */
  getCleaners(mediaType?: string): Observable<CleanerInfo[]> {
    return cleanersList(this.http, this.config.rootUrl, { media_type: mediaType }).pipe(
      map((r) => r.body.cleaners),
    );
  }

  getEmbedders(mediaType?: string): Observable<EmbedderInfo[]> {
    return embeddersList(this.http, this.config.rootUrl, { media_type: mediaType }).pipe(
      map((r) => r.body.embedders),
    );
  }

  getConverters(target?: string): Observable<ConverterInfo[]> {
    return convertersList(this.http, this.config.rootUrl, { target }).pipe(
      map((r) => r.body.converters),
    );
  }

  getConvertersForSource(source: string): Observable<ConverterInfo[]> {
    return convertersList(this.http, this.config.rootUrl, { source }).pipe(
      map((r) => r.body.converters),
    );
  }

  getDemoList(embedder?: string, clipper?: string, converter?: string): Observable<DemoDatasetListResponse> {
    return demoDatasetList(this.http, this.config.rootUrl, { embedder, clipper, converter }).pipe(
      map((r) => r.body),
    );
  }

  getDemoCategories(name: string): Observable<DemoCategoriesResponse> {
    return demoDatasetCategories(this.http, this.config.rootUrl, { name }).pipe(
      map((r) => r.body),
    );
  }
}
