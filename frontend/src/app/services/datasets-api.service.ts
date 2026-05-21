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
import { browseMediaFiles } from '../generated/api-client/fn/datasets-ui/browse-media-files';
import { selectBrowsedFile } from '../generated/api-client/fn/datasets-ui/select-browsed-file';
import { clippersList } from '../generated/api-client/fn/datasets-listings/clippers-list';
import { convertersList } from '../generated/api-client/fn/datasets-listings/converters-list';
import { dashboardDiskUsage } from '../generated/api-client/fn/datasets-ui/dashboard-disk-usage';
import { datasetAllImporters } from '../generated/api-client/fn/datasets-listings/dataset-all-importers';
import { availableDatasetFiles } from '../generated/api-client/fn/datasets-staging/available-dataset-files';
import { cancelDatasetLoad } from '../generated/api-client/fn/datasets-status/cancel-dataset-load';
import { cancelDatasetLoadTask } from '../generated/api-client/fn/datasets-status/cancel-dataset-load-task';
import { clearDatasetRoute } from '../generated/api-client/fn/datasets-load/clear-dataset-route';
import { combineDatasetsRoute } from '../generated/api-client/fn/datasets-staging/combine-datasets-route';
import { demoDatasetCategories } from '../generated/api-client/fn/datasets-ui/demo-dataset-categories';
import { demoDatasetList } from '../generated/api-client/fn/datasets-ui/demo-dataset-list';
import { detectMediaType } from '../generated/api-client/fn/datasets-ui/detect-media-type';
import { importerFieldOptions } from '../generated/api-client/fn/datasets-staging/importer-field-options';
import { datasetImporters } from '../generated/api-client/fn/datasets-listings/dataset-importers';
import { loadDemoDatasetRoute } from '../generated/api-client/fn/datasets-load/load-demo-dataset-route';
import { loadDatasetFolder } from '../generated/api-client/fn/datasets-load/load-dataset-folder';
import { loadDatasetFromSource } from '../generated/api-client/fn/datasets-load/load-dataset-from-source';
import { stageDemo } from '../generated/api-client/fn/datasets-staging/stage-demo';
import { clearStaging } from '../generated/api-client/fn/datasets-staging/clear-staging';
import { datasetStatus } from '../generated/api-client/fn/datasets-status/dataset-status';
import { deleteRegisteredDataset } from '../generated/api-client/fn/datasets-registry/delete-registered-dataset';
import { loadRegisteredDataset } from '../generated/api-client/fn/datasets-registry/load-registered-dataset';
import { updateDatasetReaders } from '../generated/api-client/fn/datasets-registry/update-dataset-readers';
import { renameRegisteredDataset } from '../generated/api-client/fn/datasets-registry/rename-registered-dataset';
import { getDatasetStats } from '../generated/api-client/fn/datasets-registry/get-dataset-stats';
import { unloadRegisteredDataset } from '../generated/api-client/fn/datasets-registry/unload-registered-dataset';
import { listRegisteredDatasets } from '../generated/api-client/fn/datasets-registry/list-registered-datasets';
import { embeddersList } from '../generated/api-client/fn/datasets-listings/embedders-list';
import { mediaTypesList } from '../generated/api-client/fn/datasets-listings/media-types-list';

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
    return datasetStatus(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

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

  browseMediaFiles(source: string, path: string): Observable<BrowseMediaFilesResponse> {
    return browseMediaFiles(this.http, this.config.rootUrl, { source, path }).pipe(
      map((r) => r.body),
    );
  }

  selectBrowsedFile(source: string, path: string): Observable<BrowseMediaFilesSelectResponse> {
    return selectBrowsedFile(this.http, this.config.rootUrl, {
      body: { source, path },
    }).pipe(map((r) => r.body));
  }

  getMediaTypes(): Observable<{ media_types: import('../models/api.models').MediaTypeInfo[] }> {
    return mediaTypesList(this.http, this.config.rootUrl).pipe(
      map((r) => r.body as unknown as { media_types: import('../models/api.models').MediaTypeInfo[] }),
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

  getAvailableFiles(): Observable<DatasetAvailableFilesResponse> {
    return availableDatasetFiles(this.http, this.config.rootUrl).pipe(map((r) => r.body));
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
    return importerFieldOptions(this.http, this.config.rootUrl, {
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
    return loadDemoDatasetRoute(this.http, this.config.rootUrl, { body }).pipe(map((r) => r.body));
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

  cancelIngest(): Observable<CancelDatasetLoadResponse> {
    return cancelDatasetLoad(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  cancelTask(taskId: string): Observable<CancelDatasetLoadResponse> {
    return cancelDatasetLoadTask(this.http, this.config.rootUrl, { task_id: taskId }).pipe(
      map((r) => r.body),
    );
  }

  clearDataset(): Observable<DatasetClearResponse> {
    return clearDatasetRoute(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  /** Binary stream — stays on plain ``HttpClient`` so ``responseType: 'blob'``
   *  produces a ``Blob`` (the generated function declares the success body as
   *  ``Error`` because the spec only carries error responses for this route). */
  exportDataset(): Observable<Blob> {
    return this.http.get('/api/dataset/export', { responseType: 'blob' });
  }

  getRegistry(): Observable<DatasetsRegistryListResponse> {
    return listRegisteredDatasets(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  loadRegistered(datasetId: string): Observable<DatasetRegistryLoadResponse> {
    return loadRegisteredDataset(this.http, this.config.rootUrl, {
      dataset_id: datasetId,
    }).pipe(map((r) => r.body));
  }

  unloadRegistered(datasetId: string): Observable<DatasetRegistryOkResponse> {
    return unloadRegisteredDataset(this.http, this.config.rootUrl, {
      dataset_id: datasetId,
    }).pipe(map((r) => r.body));
  }

  deleteRegistered(datasetId: string): Observable<DatasetRegistryOkResponse> {
    return deleteRegisteredDataset(this.http, this.config.rootUrl, {
      dataset_id: datasetId,
    }).pipe(map((r) => r.body));
  }

  renameRegistered(
    datasetId: string,
    newName: string,
  ): Observable<DatasetRegistryRenameResponse> {
    return renameRegisteredDataset(this.http, this.config.rootUrl, {
      dataset_id: datasetId,
      body: { name: newName },
    }).pipe(map((r) => r.body));
  }

  updateReaders(
    datasetId: string,
    readers: string[],
  ): Observable<DatasetRegistryReadersResponse> {
    return updateDatasetReaders(this.http, this.config.rootUrl, {
      dataset_id: datasetId,
      body: { readers },
    }).pipe(map((r) => r.body));
  }

  loadSource(params: DatasetLoadSourceRequest): Observable<DatasetLoadStartedResponse> {
    return loadDatasetFromSource(this.http, this.config.rootUrl, { body: params }).pipe(
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
    return getDatasetStats(this.http, this.config.rootUrl, {
      dataset_id: datasetId,
    }).pipe(map((r) => r.body));
  }

  getDiskUsage(): Observable<DashboardDiskUsageResponse> {
    return dashboardDiskUsage(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }
}
