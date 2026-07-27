import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { CancelDatasetLoadResponse } from '../generated/api-client/models/cancel-dataset-load-response';
import type { DatasetDomainShiftResponse } from '../generated/api-client/models/dataset-domain-shift-response';
import type { DatasetRegistryLoadResponse } from '../generated/api-client/models/dataset-registry-load-response';
import type { DatasetRegistryOkResponse } from '../generated/api-client/models/dataset-registry-ok-response';
import type { DatasetRegistryReadersResponse } from '../generated/api-client/models/dataset-registry-readers-response';
import type { DatasetRegistryRenameResponse } from '../generated/api-client/models/dataset-registry-rename-response';
import type { DatasetRegistryDuplicatesResponse } from '../generated/api-client/models/dataset-registry-duplicates-response';
import type { DatasetRegistryStatsResponse } from '../generated/api-client/models/dataset-registry-stats-response';
import type { DatasetsRegistryListResponse } from '../generated/api-client/models/datasets-registry-list-response';
import type { DatasetStatusResponse } from '../generated/api-client/models/dataset-status-response';
import { cancelDatasetLoad } from '../generated/api-client/fn/datasets-status/cancel-dataset-load';
import { cancelDatasetLoadTask } from '../generated/api-client/fn/datasets-status/cancel-dataset-load-task';
import { datasetStatus } from '../generated/api-client/fn/datasets-status/dataset-status';
import { datasetDomainShift } from '../generated/api-client/fn/datasets-registry/dataset-domain-shift';
import { deleteRegisteredDataset } from '../generated/api-client/fn/datasets-registry/delete-registered-dataset';
import { getDatasetDuplicates } from '../generated/api-client/fn/datasets-registry/get-dataset-duplicates';
import { getDatasetStats } from '../generated/api-client/fn/datasets-registry/get-dataset-stats';
import { listRegisteredDatasets } from '../generated/api-client/fn/datasets-registry/list-registered-datasets';
import { loadRegisteredDataset } from '../generated/api-client/fn/datasets-registry/load-registered-dataset';
import { renameRegisteredDataset } from '../generated/api-client/fn/datasets-registry/rename-registered-dataset';
import { unloadRegisteredDataset } from '../generated/api-client/fn/datasets-registry/unload-registered-dataset';
import { updateDatasetReaders } from '../generated/api-client/fn/datasets-registry/update-dataset-readers';

/** Dataset registry: list / load / unload / rename / readers / stats /
 *  preload-embedder, plus the dataset-status / cancel-ingest / cancel-task
 *  helpers that drive the loading lifecycle. */
@Injectable({ providedIn: 'root' })
export class DatasetsRegistryApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  getStatus(): Observable<DatasetStatusResponse> {
    return datasetStatus(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  cancelIngest(): Observable<CancelDatasetLoadResponse> {
    return cancelDatasetLoad(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  cancelTask(taskId: string): Observable<CancelDatasetLoadResponse> {
    return cancelDatasetLoadTask(this.http, this.config.rootUrl, { task_id: taskId }).pipe(
      map((r) => r.body),
    );
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

  /** The collapsed duplicate sets of a *loaded* dataset, expanded to their
   *  full membership. The backend refuses (400) when the dataset isn't
   *  loaded — duplicate provenance lives only in memory. */
  getDatasetDuplicates(datasetId: string): Observable<DatasetRegistryDuplicatesResponse> {
    return getDatasetDuplicates(this.http, this.config.rootUrl, {
      dataset_id: datasetId,
    }).pipe(map((r) => r.body));
  }

  /** Typicality of the *active* dataset (the ``X-Dataset-Id`` header, set by
   *  the active-context interceptor) under *referenceDatasetId*'s coverage
   *  atlas. ``frac_atypical`` is roughly the share of the active dataset that
   *  lies outside the reference (training) domain. The backend refuses (400)
   *  when the reference isn't loaded, has no atlas, or uses a different
   *  embedder. */
  domainShift(referenceDatasetId: string): Observable<DatasetDomainShiftResponse> {
    return datasetDomainShift(this.http, this.config.rootUrl, {
      dataset_id: referenceDatasetId,
    }).pipe(map((r) => r.body));
  }
}
