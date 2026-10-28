import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpContext } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { DetectorCancelResponse } from '../generated/api-client/models/detector-cancel-response';
import type { DetectorRegistryAutofindResponse } from '../generated/api-client/models/detector-registry-autofind-response';
import type { DetectorRegistryCreateRequest } from '../generated/api-client/models/detector-registry-create-request';
import type { DetectorRegistryCreateResponse } from '../generated/api-client/models/detector-registry-create-response';
import type { DetectorRegistryDeleteResponse } from '../generated/api-client/models/detector-registry-delete-response';
import type { DetectorRegistryListResponse } from '../generated/api-client/models/detector-registry-list-response';
import type { DetectorRegistryLoadResponse } from '../generated/api-client/models/detector-registry-load-response';
import type { DetectorRegistryRenameResponse } from '../generated/api-client/models/detector-registry-rename-response';
import type { DetectorRegistryReadersResponse } from '../generated/api-client/models/detector-registry-readers-response';
import type { DetectorRegistryStatsResponse } from '../generated/api-client/models/detector-registry-stats-response';
import type { DetectorRegistryUnloadResponse } from '../generated/api-client/models/detector-registry-unload-response';
import type { DetectorBrowsePositivesResponse } from '../generated/api-client/models/detector-browse-positives-response';
import type { DetectorBrowsePositivesReleaseResponse } from '../generated/api-client/models/detector-browse-positives-release-response';
import { browseDetectorPositives } from '../generated/api-client/fn/detectors-registry/browse-detector-positives';
import { cancelDetectorLoadingTask } from '../generated/api-client/fn/detectors-registry/cancel-detector-loading-task';
import { deleteRegisteredDetector } from '../generated/api-client/fn/detectors-registry/delete-registered-detector';
import { getDetectorStats } from '../generated/api-client/fn/detectors-registry/get-detector-stats';
import { releaseDetectorPositivesBrowse } from '../generated/api-client/fn/detectors-registry/release-detector-positives-browse';
import { listRegisteredDetectors } from '../generated/api-client/fn/detectors-registry/list-registered-detectors';
import { loadDetectorRoute } from '../generated/api-client/fn/detectors-registry/load-detector-route';
import { registerDetectorRoute } from '../generated/api-client/fn/detectors-registry/register-detector-route';
import { renameRegisteredDetector } from '../generated/api-client/fn/detectors-registry/rename-registered-detector';
import { setDetectorAutofind } from '../generated/api-client/fn/detectors-registry/set-detector-autofind';
import { updateDetectorReaders } from '../generated/api-client/fn/detectors-registry/update-detector-readers';
import { unloadDetectorRoute } from '../generated/api-client/fn/detectors-registry/unload-detector-route';
import { DATASET_OVERRIDE } from '../interceptors/active-context.interceptor';

/** Detector registry: list / load / unload / rename / cancel / autofind /
 *  labelset-move / from-labelset. */
@Injectable({ providedIn: 'root' })
export class DetectorsRegistryApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);

  getRegistry(): Observable<DetectorRegistryListResponse> {
    return listRegisteredDetectors(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  registerDetector(
    params: DetectorRegistryCreateRequest,
  ): Observable<DetectorRegistryCreateResponse> {
    return registerDetectorRoute(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  /** Plugin-field route: stays on plain ``HttpClient`` because the body
   *  shape is plugin-dependent and not described in the OpenAPI spec.
   *  Same pattern as ``LabelImportersApiService.runImport``. */
  registerDetectorFromLabelset(
    importerName: string,
    params: Record<string, unknown>,
    file?: File,
    fileFieldKey?: string,
  ): Observable<unknown> {
    const url = `/api/detectors/registry/from-labelset/${encodeURIComponent(importerName)}`;
    if (file && fileFieldKey) {
      const formData = new FormData();
      formData.append(fileFieldKey, file, file.name);
      for (const [key, value] of Object.entries(params)) {
        if (key !== fileFieldKey) {
          formData.append(key, String(value ?? ''));
        }
      }
      return this.http.post(url, formData);
    }
    return this.http.post(url, params);
  }

  /** Pass *context* to tag the request, e.g. with {@link SKIP_ERROR_TOAST}
   *  when the caller reports a failure itself (the Dashboard's bulk delete
   *  sums a batch's failures into one toast). */
  deleteFromRegistry(detectorId: string, context?: HttpContext): Observable<DetectorRegistryDeleteResponse> {
    return deleteRegisteredDetector(
      this.http,
      this.config.rootUrl,
      { detector_id: detectorId },
      context,
    ).pipe(map((r) => r.body));
  }

  renameInRegistry(
    detectorId: string,
    newName: string,
  ): Observable<DetectorRegistryRenameResponse> {
    return renameRegisteredDetector(this.http, this.config.rootUrl, {
      detector_id: detectorId,
      body: { name: newName },
    }).pipe(map((r) => r.body));
  }

  /** Load a detector. ``datasetId`` names the dataset it is loaded for, with
   *  no detector (`DATASET_OVERRIDE`): the load restores the detector's labels
   *  and seeds its examples against the request's dataset, so a pair switch
   *  names the dataset it is switching to rather than the one still active
   *  (#5358). Without it the request carries the active pair. */
  loadDetector(detectorId: string | null, datasetId = ''): Observable<DetectorRegistryLoadResponse> {
    const context = datasetId ? new HttpContext().set(DATASET_OVERRIDE, datasetId) : undefined;
    return loadDetectorRoute(this.http, this.config.rootUrl, {
      body: { detector_id: detectorId },
    }, context).pipe(map((r) => r.body));
  }

  unloadDetector(detectorId: string): Observable<DetectorRegistryUnloadResponse> {
    return unloadDetectorRoute(this.http, this.config.rootUrl, {
      detector_id: detectorId,
    }).pipe(map((r) => r.body));
  }

  cancelDetectorLoadingTask(taskId: string): Observable<DetectorCancelResponse> {
    return cancelDetectorLoadingTask(this.http, this.config.rootUrl, {
      task_id: taskId,
    }).pipe(map((r) => r.body));
  }

  setAutofind(detectorId: string, autofind: boolean): Observable<DetectorRegistryAutofindResponse> {
    return setDetectorAutofind(this.http, this.config.rootUrl, {
      detector_id: detectorId,
      body: { autofind },
    }).pipe(map((r) => r.body));
  }

  /** Update a detector's access list (creator-only on the backend).
   *  ``["*"]`` makes it visible to all users. */
  updateReaders(detectorId: string, readers: string[]): Observable<DetectorRegistryReadersResponse> {
    return updateDetectorReaders(this.http, this.config.rootUrl, {
      detector_id: detectorId,
      body: { readers },
    }).pipe(map((r) => r.body));
  }

  getDetectorStats(detectorId: string): Observable<DetectorRegistryStatsResponse> {
    return getDetectorStats(this.http, this.config.rootUrl, {
      detector_id: detectorId,
    }).pipe(map((r) => r.body));
  }

  /** Prepare an in-memory VTSBrowse map of this detector's positives,
   *  embedded with the detector's own embedder. Returns the synthetic
   *  ``dataset_id`` to navigate to plus the progress ``task_id``. */
  browsePositives(detectorId: string): Observable<DetectorBrowsePositivesResponse> {
    return browseDetectorPositives(this.http, this.config.rootUrl, {
      detector_id: detectorId,
    }).pipe(map((r) => r.body));
  }

  /** Free the ephemeral positives-browse context when leaving the view. */
  releasePositivesBrowse(detectorId: string): Observable<DetectorBrowsePositivesReleaseResponse> {
    return releaseDetectorPositivesBrowse(this.http, this.config.rootUrl, {
      detector_id: detectorId,
    }).pipe(map((r) => r.body));
  }
}
