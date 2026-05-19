import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { BrowseMediaFilesResponse } from '../generated/api-client/models/browse-media-files-response';
import type { BrowseMediaFilesSelectResponse } from '../generated/api-client/models/browse-media-files-select-response';
import type { CancelDatasetLoadResponse } from '../generated/api-client/models/cancel-dataset-load-response';
import type { ClearStagingResponse } from '../generated/api-client/models/clear-staging-response';
import type { DashboardDiskUsageResponse } from '../generated/api-client/models/dashboard-disk-usage-response';
import type { DatasetAllImportersListResponse } from '../generated/api-client/models/dataset-all-importers-list-response';
import type { DatasetAvailableFilesResponse } from '../generated/api-client/models/dataset-available-files-response';
import type { DatasetClearResponse } from '../generated/api-client/models/dataset-clear-response';
import type { DatasetCombineRequest } from '../generated/api-client/models/dataset-combine-request';
import type { DatasetImportersListResponse } from '../generated/api-client/models/dataset-importers-list-response';
import type { DatasetLoadDemoRequest } from '../generated/api-client/models/dataset-load-demo-request';
import type { DatasetLoadFolderRequest } from '../generated/api-client/models/dataset-load-folder-request';
import type { DatasetLoadSourceRequest } from '../generated/api-client/models/dataset-load-source-request';
import type { DatasetLoadStartedResponse } from '../generated/api-client/models/dataset-load-started-response';
import type { DatasetRegistryLoadResponse } from '../generated/api-client/models/dataset-registry-load-response';
import type { DatasetRegistryOkResponse } from '../generated/api-client/models/dataset-registry-ok-response';
import type { DatasetRegistryReadersResponse } from '../generated/api-client/models/dataset-registry-readers-response';
import type { DatasetRegistryRenameResponse } from '../generated/api-client/models/dataset-registry-rename-response';
import type { DatasetRegistryStatsResponse } from '../generated/api-client/models/dataset-registry-stats-response';
import type { DatasetsRegistryListResponse } from '../generated/api-client/models/datasets-registry-list-response';
import type { DatasetStageFileResponse } from '../generated/api-client/models/dataset-stage-file-response';
import type { DatasetStagingStartedResponse } from '../generated/api-client/models/dataset-staging-started-response';
import type { DatasetStatusResponse } from '../generated/api-client/models/dataset-status-response';
import type { DemoCategoriesResponse } from '../generated/api-client/models/demo-categories-response';
import type { DemoDatasetListResponse } from '../generated/api-client/models/demo-dataset-list-response';
import type { DetectMediaTypeResponse } from '../generated/api-client/models/detect-media-type-response';
import type { ImporterFieldOptionsResponse } from '../generated/api-client/models/importer-field-options-response';
import type {
  ClipperInfo,
  ConverterInfo,
  EmbedderInfo,
  ImporterInfo,
  ImporterPickerTab,
} from '../models/api.models';
import { apiBrowseMediaFilesGet } from '../generated/api-client/fn/datasets-ui/api-browse-media-files-get';
import { apiBrowseMediaFilesSelectPost } from '../generated/api-client/fn/datasets-ui/api-browse-media-files-select-post';
import { apiClippersGet } from '../generated/api-client/fn/datasets-listings/api-clippers-get';
import { apiConvertersGet } from '../generated/api-client/fn/datasets-listings/api-converters-get';
import { apiDashboardDiskUsageGet } from '../generated/api-client/fn/datasets-ui/api-dashboard-disk-usage-get';
import { apiDatasetAllImportersGet } from '../generated/api-client/fn/datasets-listings/api-dataset-all-importers-get';
import { apiDatasetAvailableFilesGet } from '../generated/api-client/fn/datasets-staging/api-dataset-available-files-get';
import { apiDatasetCancelPost } from '../generated/api-client/fn/datasets-status/api-dataset-cancel-post';
import { apiDatasetCancelTaskIdPost } from '../generated/api-client/fn/datasets-status/api-dataset-cancel-task-id-post';
import { apiDatasetClearPost } from '../generated/api-client/fn/datasets-load/api-dataset-clear-post';
import { apiDatasetCombinePost } from '../generated/api-client/fn/datasets-staging/api-dataset-combine-post';
import { apiDatasetDemoCategoriesNameGet } from '../generated/api-client/fn/datasets-ui/api-dataset-demo-categories-name-get';
import { apiDatasetDemoListGet } from '../generated/api-client/fn/datasets-ui/api-dataset-demo-list-get';
import { apiDatasetDetectMediaTypeGet } from '../generated/api-client/fn/datasets-ui/api-dataset-detect-media-type-get';
import { apiDatasetImportImporterNameOptionsPost } from '../generated/api-client/fn/datasets-staging/api-dataset-import-importer-name-options-post';
import { apiDatasetImportersGet } from '../generated/api-client/fn/datasets-listings/api-dataset-importers-get';
import { apiDatasetLoadDemoPost } from '../generated/api-client/fn/datasets-load/api-dataset-load-demo-post';
import { apiDatasetLoadFolderPost } from '../generated/api-client/fn/datasets-load/api-dataset-load-folder-post';
import { apiDatasetLoadSourcePost } from '../generated/api-client/fn/datasets-load/api-dataset-load-source-post';
import { apiDatasetStageDemoNamePost } from '../generated/api-client/fn/datasets-staging/api-dataset-stage-demo-name-post';
import { apiDatasetStagingDelete } from '../generated/api-client/fn/datasets-staging/api-dataset-staging-delete';
import { apiDatasetStatusGet } from '../generated/api-client/fn/datasets-status/api-dataset-status-get';
import { apiDatasetsRegistryDatasetIdDelete } from '../generated/api-client/fn/datasets-registry/api-datasets-registry-dataset-id-delete';
import { apiDatasetsRegistryDatasetIdLoadPost } from '../generated/api-client/fn/datasets-registry/api-datasets-registry-dataset-id-load-post';
import { apiDatasetsRegistryDatasetIdReadersPut } from '../generated/api-client/fn/datasets-registry/api-datasets-registry-dataset-id-readers-put';
import { apiDatasetsRegistryDatasetIdRenamePut } from '../generated/api-client/fn/datasets-registry/api-datasets-registry-dataset-id-rename-put';
import { apiDatasetsRegistryDatasetIdStatsGet } from '../generated/api-client/fn/datasets-registry/api-datasets-registry-dataset-id-stats-get';
import { apiDatasetsRegistryDatasetIdUnloadPost } from '../generated/api-client/fn/datasets-registry/api-datasets-registry-dataset-id-unload-post';
import { apiDatasetsRegistryGet } from '../generated/api-client/fn/datasets-registry/api-datasets-registry-get';
import { apiEmbeddersGet } from '../generated/api-client/fn/datasets-listings/api-embedders-get';
import { apiMediaTypesGet } from '../generated/api-client/fn/datasets-listings/api-media-types-get';

/** The importer/clipper/embedder/converter listings return plugin
 *  ``to_dict()`` payloads — the generated types describe them as
 *  ``Array<{[key: string]: any}>`` because the inner shapes are
 *  plugin-dependent.  The richer ``ImporterInfo`` / ``ClipperInfo`` /
 *  ``EmbedderInfo`` / ``ConverterInfo`` interfaces in
 *  ``frontend/src/app/models/api.models.ts`` describe the actual fields
 *  consumers read off these payloads; this service casts at the boundary
 *  so callers don't have to. */
type ImportersResponse = { importers: ImporterInfo[]; tabs?: ImporterPickerTab[] };

@Injectable({ providedIn: 'root' })
export class DatasetsApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  getStatus(): Observable<DatasetStatusResponse> {
    return apiDatasetStatusGet(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  getImporters(): Observable<ImportersResponse> {
    return apiDatasetImportersGet(this.http, this.config.rootUrl).pipe(
      map((r) => r.body as unknown as ImportersResponse),
    );
  }

  getAllImporters(): Observable<ImportersResponse> {
    return apiDatasetAllImportersGet(this.http, this.config.rootUrl).pipe(
      map((r) => r.body as unknown as ImportersResponse),
    );
  }

  getDemoList(embedder?: string, clipper?: string): Observable<DemoDatasetListResponse> {
    return apiDatasetDemoListGet(this.http, this.config.rootUrl, { embedder, clipper }).pipe(
      map((r) => r.body),
    );
  }

  getDemoCategories(name: string): Observable<DemoCategoriesResponse> {
    return apiDatasetDemoCategoriesNameGet(this.http, this.config.rootUrl, { name }).pipe(
      map((r) => r.body),
    );
  }

  browseMediaFiles(source: string, path: string): Observable<BrowseMediaFilesResponse> {
    return apiBrowseMediaFilesGet(this.http, this.config.rootUrl, { source, path }).pipe(
      map((r) => r.body),
    );
  }

  selectBrowsedFile(source: string, path: string): Observable<BrowseMediaFilesSelectResponse> {
    return apiBrowseMediaFilesSelectPost(this.http, this.config.rootUrl, {
      body: { source, path },
    }).pipe(map((r) => r.body));
  }

  getMediaTypes(): Observable<{ media_types: import('../models/api.models').MediaTypeInfo[] }> {
    return apiMediaTypesGet(this.http, this.config.rootUrl).pipe(
      map((r) => r.body as unknown as { media_types: import('../models/api.models').MediaTypeInfo[] }),
    );
  }

  detectMediaType(
    source: string,
    path: string,
    recursive: boolean,
    limit = 50,
  ): Observable<DetectMediaTypeResponse> {
    return apiDatasetDetectMediaTypeGet(this.http, this.config.rootUrl, {
      source,
      path,
      recursive,
      limit,
    }).pipe(map((r) => r.body));
  }

  getClippers(mediaType?: string): Observable<ClipperInfo[]> {
    return apiClippersGet(this.http, this.config.rootUrl, { media_type: mediaType }).pipe(
      map((r) => r.body.clippers as unknown as ClipperInfo[]),
    );
  }

  getEmbedders(mediaType?: string): Observable<EmbedderInfo[]> {
    return apiEmbeddersGet(this.http, this.config.rootUrl, { media_type: mediaType }).pipe(
      map((r) => r.body.embedders as unknown as EmbedderInfo[]),
    );
  }

  getConverters(target?: string): Observable<ConverterInfo[]> {
    return apiConvertersGet(this.http, this.config.rootUrl, { target }).pipe(
      map((r) => r.body.converters as unknown as ConverterInfo[]),
    );
  }

  getAvailableFiles(): Observable<DatasetAvailableFilesResponse> {
    return apiDatasetAvailableFilesGet(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  /** Plugin-field route — body shape is plugin-dependent, not described in
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
    return apiDatasetImportImporterNameOptionsPost(this.http, this.config.rootUrl, {
      importer_name: importerName,
      body: { field_key: fieldKey, values },
    }).pipe(map((r) => r.body));
  }

  /** Multipart upload — the caller builds the FormData with the files,
   *  ``media_type``, optional ``embedder`` / ``clipper`` / ``clipper_params``.
   *  Stays on plain ``HttpClient`` because ng-openapi-gen doesn't model
   *  multipart bodies (the generated function's ``$Params`` has no ``body``
   *  field). */
  importLocalFolder(formData: FormData): Observable<DatasetLoadStartedResponse> {
    return this.http.post<DatasetLoadStartedResponse>('/api/dataset/import-local-folder', formData);
  }

  loadDemo(name: string, params?: Record<string, string>): Observable<DatasetLoadStartedResponse> {
    const body: DatasetLoadDemoRequest = { name, ...params };
    return apiDatasetLoadDemoPost(this.http, this.config.rootUrl, { body }).pipe(map((r) => r.body));
  }

  /** Multipart upload — see {@link importLocalFolder}. */
  loadFile(file: File): Observable<DatasetLoadStartedResponse> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post<DatasetLoadStartedResponse>('/api/dataset/load-file', formData);
  }

  /** Multipart upload — see {@link importLocalFolder}. */
  stageFile(file: File): Observable<DatasetStageFileResponse> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post<DatasetStageFileResponse>('/api/dataset/stage-file', formData);
  }

  /** Plugin-field route — body shape is plugin-dependent, not described in
   *  the OpenAPI spec.  Stays on plain ``HttpClient``. */
  stageImport(importerName: string, params: Record<string, unknown>): Observable<unknown> {
    return this.http.post(`/api/dataset/stage-import/${encodeURIComponent(importerName)}`, params);
  }

  stageDemo(name: string): Observable<DatasetStagingStartedResponse> {
    return apiDatasetStageDemoNamePost(this.http, this.config.rootUrl, {
      name,
      body: {},
    }).pipe(map((r) => r.body));
  }

  clearStaging(): Observable<ClearStagingResponse> {
    return apiDatasetStagingDelete(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  combineDatasets(params: DatasetCombineRequest): Observable<DatasetLoadStartedResponse> {
    return apiDatasetCombinePost(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  cancelIngest(): Observable<CancelDatasetLoadResponse> {
    return apiDatasetCancelPost(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  cancelTask(taskId: string): Observable<CancelDatasetLoadResponse> {
    return apiDatasetCancelTaskIdPost(this.http, this.config.rootUrl, { task_id: taskId }).pipe(
      map((r) => r.body),
    );
  }

  clearDataset(): Observable<DatasetClearResponse> {
    return apiDatasetClearPost(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  /** Binary stream — stays on plain ``HttpClient`` so ``responseType: 'blob'``
   *  produces a ``Blob`` (the generated function declares the success body as
   *  ``Error`` because the spec only carries error responses for this route). */
  exportDataset(): Observable<Blob> {
    return this.http.get('/api/dataset/export', { responseType: 'blob' });
  }

  getRegistry(): Observable<DatasetsRegistryListResponse> {
    return apiDatasetsRegistryGet(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  loadRegistered(datasetId: string): Observable<DatasetRegistryLoadResponse> {
    return apiDatasetsRegistryDatasetIdLoadPost(this.http, this.config.rootUrl, {
      dataset_id: datasetId,
    }).pipe(map((r) => r.body));
  }

  unloadRegistered(datasetId: string): Observable<DatasetRegistryOkResponse> {
    return apiDatasetsRegistryDatasetIdUnloadPost(this.http, this.config.rootUrl, {
      dataset_id: datasetId,
    }).pipe(map((r) => r.body));
  }

  deleteRegistered(datasetId: string): Observable<DatasetRegistryOkResponse> {
    return apiDatasetsRegistryDatasetIdDelete(this.http, this.config.rootUrl, {
      dataset_id: datasetId,
    }).pipe(map((r) => r.body));
  }

  renameRegistered(
    datasetId: string,
    newName: string,
  ): Observable<DatasetRegistryRenameResponse> {
    return apiDatasetsRegistryDatasetIdRenamePut(this.http, this.config.rootUrl, {
      dataset_id: datasetId,
      body: { name: newName },
    }).pipe(map((r) => r.body));
  }

  updateReaders(
    datasetId: string,
    readers: string[],
  ): Observable<DatasetRegistryReadersResponse> {
    return apiDatasetsRegistryDatasetIdReadersPut(this.http, this.config.rootUrl, {
      dataset_id: datasetId,
      body: { readers },
    }).pipe(map((r) => r.body));
  }

  loadSource(params: DatasetLoadSourceRequest): Observable<DatasetLoadStartedResponse> {
    return apiDatasetLoadSourcePost(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  /** Hint the backend to warm this dataset's embedder in a background
   *  thread, so a subsequent Train click is instant. Fire-and-forget:
   *  the backend dedupes against already-loaded embedders. */
  preloadEmbedder(datasetId: string): Observable<{ ok: boolean; embedder: string }> {
    return this.http.post<{ ok: boolean; embedder: string }>(
      `/api/datasets/registry/${encodeURIComponent(datasetId)}/preload-embedder`,
      {},
    );
  }

  getDatasetStats(datasetId: string): Observable<DatasetRegistryStatsResponse> {
    return apiDatasetsRegistryDatasetIdStatsGet(this.http, this.config.rootUrl, {
      dataset_id: datasetId,
    }).pipe(map((r) => r.body));
  }

  getDiskUsage(): Observable<DashboardDiskUsageResponse> {
    return apiDashboardDiskUsageGet(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }
}
