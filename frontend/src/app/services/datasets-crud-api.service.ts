import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { ClearStagingResponse } from '../generated/api-client/models/clear-staging-response';
import type { DatasetAllImportersListResponse } from '../generated/api-client/models/dataset-all-importers-list-response';
import type { DatasetAvailableFilesResponse } from '../generated/api-client/models/dataset-available-files-response';
import type { DatasetClearResponse } from '../generated/api-client/models/dataset-clear-response';
import type { DatasetCombineRequest } from '../generated/api-client/models/dataset-combine-request';
import type { DatasetImportersListResponse } from '../generated/api-client/models/dataset-importers-list-response';
import type { DatasetLoadDemoRequest } from '../generated/api-client/models/dataset-load-demo-request';
import type { DatasetLoadSourceRequest } from '../generated/api-client/models/dataset-load-source-request';
import type { DatasetLoadStartedResponse } from '../generated/api-client/models/dataset-load-started-response';
import type { DatasetStageFileResponse } from '../generated/api-client/models/dataset-stage-file-response';
import type { DatasetStagingStartedResponse } from '../generated/api-client/models/dataset-staging-started-response';
import type { DetectMediaTypeResponse } from '../generated/api-client/models/detect-media-type-response';
import type { ImporterFieldOptionsResponse } from '../generated/api-client/models/importer-field-options-response';
import type {
  ImporterInfo,
  ImporterPickerTab,
} from '../models/api.models';
import { availableDatasetFiles } from '../generated/api-client/fn/datasets-staging/available-dataset-files';
import { clearDatasetRoute } from '../generated/api-client/fn/datasets-load/clear-dataset-route';
import { clearStaging } from '../generated/api-client/fn/datasets-staging/clear-staging';
import { combineDatasetsRoute } from '../generated/api-client/fn/datasets-staging/combine-datasets-route';
import { datasetAllImporters } from '../generated/api-client/fn/datasets-listings/dataset-all-importers';
import { datasetImporters } from '../generated/api-client/fn/datasets-listings/dataset-importers';
import { detectMediaType } from '../generated/api-client/fn/datasets-ui/detect-media-type';
import { importerFieldOptions } from '../generated/api-client/fn/datasets-staging/importer-field-options';
import { loadDatasetFromSource } from '../generated/api-client/fn/datasets-load/load-dataset-from-source';
import { loadDemoDatasetRoute } from '../generated/api-client/fn/datasets-load/load-demo-dataset-route';
import { stageDemo } from '../generated/api-client/fn/datasets-staging/stage-demo';

/** The importer/clipper/embedder/converter listings return plugin
 *  ``to_dict()`` payloads; the generated types describe them as
 *  ``Array<{[key: string]: any}>`` because the inner shapes are
 *  plugin-dependent.  The richer ``ImporterInfo`` / ``ClipperInfo`` /
 *  ``EmbedderInfo`` / ``ConverterInfo`` interfaces in
 *  ``frontend/src/app/models/api.models.ts`` describe the actual fields
 *  consumers read off these payloads; this service casts at the boundary
 *  so callers don't have to. */
type ImportersResponse = { importers: ImporterInfo[]; tabs?: ImporterPickerTab[] };

/** Dataset CRUD: importer listings, import, stage, clear, export,
 *  detect-media-type, plus the load-from-source / load-demo entry points
 *  that kick off an ingest. */
@Injectable({ providedIn: 'root' })
export class DatasetsCrudApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  getImporters(): Observable<ImportersResponse> {
    return datasetImporters(this.http, this.config.rootUrl).pipe(
      map((r) => r.body as unknown as ImportersResponse),
    );
  }

  getAllImporters(): Observable<ImportersResponse> {
    return datasetAllImporters(this.http, this.config.rootUrl).pipe(
      map((r) => r.body as unknown as ImportersResponse),
    );
  }

  detectMediaType(
    source: string,
    path: string,
    recursive: boolean,
    limit = 50,
  ): Observable<DetectMediaTypeResponse> {
    return detectMediaType(this.http, this.config.rootUrl, {
      source,
      path,
      recursive,
      limit,
    }).pipe(map((r) => r.body));
  }

  getAvailableFiles(): Observable<DatasetAvailableFilesResponse> {
    return availableDatasetFiles(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  /** Plugin-field route: body shape is plugin-dependent, not described in
   *  the OpenAPI spec.  Stays on plain ``HttpClient``. */
  runImporter(importerName: string, params: Record<string, unknown>): Observable<unknown> {
    return this.http.post(`/api/dataset/import/${encodeURIComponent(importerName)}`, params);
  }

  /**
   * Fetch dropdown options for an importer field whose options are computed
   * at runtime by the importer (``dynamic_options=true``).
   */
  getImporterFieldOptions(
    importerName: string,
    fieldKey: string,
    values: Record<string, unknown>,
  ): Observable<ImporterFieldOptionsResponse> {
    return importerFieldOptions(this.http, this.config.rootUrl, {
      importer_name: importerName,
      body: { field_key: fieldKey, values },
    }).pipe(map((r) => r.body));
  }

  /** Multipart upload: the caller builds the FormData with the files,
   *  ``media_type``, optional ``embedder`` / ``clipper`` / ``clipper_params``.
   *  Stays on plain ``HttpClient`` because ng-openapi-gen doesn't model
   *  multipart bodies (the generated function's ``$Params`` has no ``body``
   *  field). */
  importLocalFolder(formData: FormData): Observable<DatasetLoadStartedResponse> {
    return this.http.post<DatasetLoadStartedResponse>('/api/dataset/import-local-folder', formData);
  }

  /** Multipart upload: the caller builds the FormData with a single
   *  ``paths_file`` (txt list or npz archive), ``media_type``, optional
   *  ``embedder`` / ``clipper`` / ``clipper_params`` / ``source_specs``.
   *  See {@link importLocalFolder} for why this stays on plain HttpClient. */
  importLocalFiles(formData: FormData): Observable<DatasetLoadStartedResponse> {
    return this.http.post<DatasetLoadStartedResponse>('/api/dataset/import-local-files', formData);
  }

  loadDemo(name: string, params?: Record<string, string>): Observable<DatasetLoadStartedResponse> {
    const body: DatasetLoadDemoRequest = { name, ...params };
    return loadDemoDatasetRoute(this.http, this.config.rootUrl, { body }).pipe(map((r) => r.body));
  }

  /** Multipart upload; see {@link importLocalFolder}.  ``buildProjection``
   *  opts into computing the 2-D Browse projection at ingest. */
  loadFile(file: File, buildProjection = false): Observable<DatasetLoadStartedResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('build_projection', buildProjection ? 'true' : 'false');
    return this.http.post<DatasetLoadStartedResponse>('/api/dataset/load-file', formData);
  }

  /** Multipart upload; see {@link importLocalFolder}. */
  stageFile(file: File): Observable<DatasetStageFileResponse> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post<DatasetStageFileResponse>('/api/dataset/stage-file', formData);
  }

  /** Plugin-field route: body shape is plugin-dependent, not described in
   *  the OpenAPI spec.  Stays on plain ``HttpClient``. */
  stageImport(importerName: string, params: Record<string, unknown>): Observable<unknown> {
    return this.http.post(`/api/dataset/stage-import/${encodeURIComponent(importerName)}`, params);
  }

  stageDemo(name: string): Observable<DatasetStagingStartedResponse> {
    return stageDemo(this.http, this.config.rootUrl, {
      name,
      body: {},
    }).pipe(map((r) => r.body));
  }

  clearStaging(): Observable<ClearStagingResponse> {
    return clearStaging(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  combineDatasets(params: DatasetCombineRequest): Observable<DatasetLoadStartedResponse> {
    return combineDatasetsRoute(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  clearDataset(): Observable<DatasetClearResponse> {
    return clearDatasetRoute(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  /** Binary stream; stays on plain ``HttpClient`` so ``responseType: 'blob'``
   *  produces a ``Blob`` (the generated function declares the success body as
   *  ``Error`` because the spec only carries error responses for this route). */
  exportDataset(): Observable<Blob> {
    return this.http.get('/api/dataset/export', { responseType: 'blob' });
  }

  loadSource(params: DatasetLoadSourceRequest): Observable<DatasetLoadStartedResponse> {
    return loadDatasetFromSource(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }
}
