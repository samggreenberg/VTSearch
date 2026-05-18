import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { AutoDetectRequest } from '../generated/api-client/models/auto-detect-request';
import type { AutoDetectResponse } from '../generated/api-client/models/auto-detect-response';
import type { AutoExtractResponse } from '../generated/api-client/models/auto-extract-response';
import type { AutoLocalizeResponse } from '../generated/api-client/models/auto-localize-response';
import type { AutorunExtractorCreateRequest } from '../generated/api-client/models/autorun-extractor-create-request';
import type { AutorunExtractorsListResponse } from '../generated/api-client/models/autorun-extractors-list-response';
import type { AutorunLocalizerCreateRequest } from '../generated/api-client/models/autorun-localizer-create-request';
import type { AutorunLocalizersListResponse } from '../generated/api-client/models/autorun-localizers-list-response';
import type { AutorunProcessorCreateResponse } from '../generated/api-client/models/autorun-processor-create-response';
import type { AutorunProcessorDeleteResponse } from '../generated/api-client/models/autorun-processor-delete-response';
import type { AutorunProcessorRenameResponse } from '../generated/api-client/models/autorun-processor-rename-response';
import type { DetectorCancelResponse } from '../generated/api-client/models/detector-cancel-response';
import type { DetectorCombineRequest } from '../generated/api-client/models/detector-combine-request';
import type { DetectorCombineResponse } from '../generated/api-client/models/detector-combine-response';
import type { DetectorCreateRequest } from '../generated/api-client/models/detector-create-request';
import type { DetectorCreateResponse } from '../generated/api-client/models/detector-create-response';
import type { DetectorDeleteResponse } from '../generated/api-client/models/detector-delete-response';
import type { DetectorDetail } from '../generated/api-client/models/detector-detail';
import type { DetectorExamplesRequest } from '../generated/api-client/models/detector-examples-request';
import type { DetectorExamplesResponse } from '../generated/api-client/models/detector-examples-response';
import type { DetectorLabelVoteResponse } from '../generated/api-client/models/detector-label-vote-response';
import type { DetectorLabelsDetailResponse } from '../generated/api-client/models/detector-labels-detail-response';
import type { DetectorLabelsetMoveResponse } from '../generated/api-client/models/detector-labelset-move-response';
import type { DetectorRegistryAutorunResponse } from '../generated/api-client/models/detector-registry-autorun-response';
import type { DetectorRegistryCreateRequest } from '../generated/api-client/models/detector-registry-create-request';
import type { DetectorRegistryCreateResponse } from '../generated/api-client/models/detector-registry-create-response';
import type { DetectorRegistryDeleteResponse } from '../generated/api-client/models/detector-registry-delete-response';
import type { DetectorRegistryListResponse } from '../generated/api-client/models/detector-registry-list-response';
import type { DetectorRegistryLoadResponse } from '../generated/api-client/models/detector-registry-load-response';
import type { DetectorRegistryRenameResponse } from '../generated/api-client/models/detector-registry-rename-response';
import type { DetectorRegistryUnloadResponse } from '../generated/api-client/models/detector-registry-unload-response';
import type { DetectorRenameResponse } from '../generated/api-client/models/detector-rename-response';
import type { DetectorSaveLabelsResponse } from '../generated/api-client/models/detector-save-labels-response';
import type { DetectorsListResponse } from '../generated/api-client/models/detectors-list-response';
import type { ExtractRequest } from '../generated/api-client/models/extract-request';
import type { ExtractResponse } from '../generated/api-client/models/extract-response';
import type { FindCheckLabelsRequest } from '../generated/api-client/models/find-check-labels-request';
import type { FindCheckLabelsResponse } from '../generated/api-client/models/find-check-labels-response';
import type { FindCheckLabelsWarning } from '../generated/api-client/models/find-check-labels-warning';
import type { FindLabelRequest } from '../generated/api-client/models/find-label-request';
import type { FindLabelResponse } from '../generated/api-client/models/find-label-response';
import type { FindRequest } from '../generated/api-client/models/find-request';
import type { FindResponse } from '../generated/api-client/models/find-response';
import type { LocalizeRequest } from '../generated/api-client/models/localize-request';
import type { LocalizeResponse } from '../generated/api-client/models/localize-response';
import type { PregenProcessorsAddResponse } from '../generated/api-client/models/pregen-processors-add-response';
import type { PregenProcessorsListResponse } from '../generated/api-client/models/pregen-processors-list-response';
import { apiAutoDetectPost } from '../generated/api-client/fn/detector-scoring/api-auto-detect-post';
import { apiAutoExtractPost } from '../generated/api-client/fn/processors-scoring/api-auto-extract-post';
import { apiAutoLocalizePost } from '../generated/api-client/fn/processors-scoring/api-auto-localize-post';
import { apiAutorunExtractorsGet } from '../generated/api-client/fn/processors-crud/api-autorun-extractors-get';
import { apiAutorunExtractorsNameDelete } from '../generated/api-client/fn/processors-crud/api-autorun-extractors-name-delete';
import { apiAutorunExtractorsNameRenamePut } from '../generated/api-client/fn/processors-crud/api-autorun-extractors-name-rename-put';
import { apiAutorunExtractorsPost } from '../generated/api-client/fn/processors-crud/api-autorun-extractors-post';
import { apiAutorunLocalizersGet } from '../generated/api-client/fn/processors-crud/api-autorun-localizers-get';
import { apiAutorunLocalizersNameDelete } from '../generated/api-client/fn/processors-crud/api-autorun-localizers-name-delete';
import { apiAutorunLocalizersNameRenamePut } from '../generated/api-client/fn/processors-crud/api-autorun-localizers-name-rename-put';
import { apiAutorunLocalizersPost } from '../generated/api-client/fn/processors-crud/api-autorun-localizers-post';
import { apiDetectorsCancelTaskIdPost } from '../generated/api-client/fn/detectors-registry/api-detectors-cancel-task-id-post';
import { apiDetectorsCombinePost } from '../generated/api-client/fn/detectors-crud/api-detectors-combine-post';
import { apiDetectorsGet } from '../generated/api-client/fn/detectors-crud/api-detectors-get';
import { apiDetectorsNameDelete } from '../generated/api-client/fn/detectors-crud/api-detectors-name-delete';
import { apiDetectorsNameExamplesPut } from '../generated/api-client/fn/detectors-crud/api-detectors-name-examples-put';
import { apiDetectorsNameGet } from '../generated/api-client/fn/detectors-crud/api-detectors-name-get';
import { apiDetectorsNameLabelsDetailGet } from '../generated/api-client/fn/detectors-labels/api-detectors-name-labels-detail-get';
import { apiDetectorsNameLabelsElementIdVotePost } from '../generated/api-client/fn/detectors-labels/api-detectors-name-labels-element-id-vote-post';
import { apiDetectorsNameLabelsPost } from '../generated/api-client/fn/detectors-labels/api-detectors-name-labels-post';
import { apiDetectorsNameRenamePut } from '../generated/api-client/fn/detectors-crud/api-detectors-name-rename-put';
import { apiDetectorsPost } from '../generated/api-client/fn/detectors-crud/api-detectors-post';
import { apiDetectorsRegistryDetectorIdAutorunPut } from '../generated/api-client/fn/detectors-registry/api-detectors-registry-detector-id-autorun-put';
import { apiDetectorsRegistryDetectorIdDelete } from '../generated/api-client/fn/detectors-registry/api-detectors-registry-detector-id-delete';
import { apiDetectorsRegistryDetectorIdLabelsetSourceMoveFilePost } from '../generated/api-client/fn/detectors-registry/api-detectors-registry-detector-id-labelset-source-move-file-post';
import { apiDetectorsRegistryDetectorIdRenamePut } from '../generated/api-client/fn/detectors-registry/api-detectors-registry-detector-id-rename-put';
import { apiDetectorsRegistryDetectorIdUnloadPost } from '../generated/api-client/fn/detectors-registry/api-detectors-registry-detector-id-unload-post';
import { apiDetectorsRegistryGet } from '../generated/api-client/fn/detectors-registry/api-detectors-registry-get';
import { apiDetectorsRegistryLoadPost } from '../generated/api-client/fn/detectors-registry/api-detectors-registry-load-post';
import { apiDetectorsRegistryPost } from '../generated/api-client/fn/detectors-registry/api-detectors-registry-post';
import { apiExtractPost } from '../generated/api-client/fn/processors-scoring/api-extract-post';
import { apiFindCheckLabelsPost } from '../generated/api-client/fn/detector-find/api-find-check-labels-post';
import { apiFindLabelPost } from '../generated/api-client/fn/detector-scoring/api-find-label-post';
import { apiFindPost } from '../generated/api-client/fn/detector-find/api-find-post';
import { apiLocalizePost } from '../generated/api-client/fn/processors-scoring/api-localize-post';
import { apiPregenProcessorsAddPost } from '../generated/api-client/fn/processors-crud/api-pregen-processors-add-post';
import { apiPregenProcessorsGet } from '../generated/api-client/fn/processors-crud/api-pregen-processors-get';
import { ActiveContextService } from './active-context.service';

/** Backwards-compatible alias for the generated ``FindCheckLabelsWarning`` —
 *  consumers of this service still import this name. */
export type FindLabelWarning = FindCheckLabelsWarning;

/**
 * API surface for detector CRUD, the detector registry, and detector-driven
 * scoring (auto-detect, find, find-label).  Also covers the autorun
 * extractor / localizer / pregen-processor endpoints, which share the
 * detector lifecycle from the dashboard's perspective.
 */
@Injectable({ providedIn: 'root' })
export class DetectorsApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);
  private activeContext = inject(ActiveContextService);

  // --- Detector CRUD ---

  list(): Observable<DetectorsListResponse> {
    return apiDetectorsGet(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  create(params: DetectorCreateRequest): Observable<DetectorCreateResponse> {
    return apiDetectorsPost(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  get(name: string): Observable<DetectorDetail> {
    return apiDetectorsNameGet(this.http, this.config.rootUrl, { name }).pipe(map((r) => r.body));
  }

  delete(name: string): Observable<DetectorDeleteResponse> {
    return apiDetectorsNameDelete(this.http, this.config.rootUrl, { name }).pipe(
      map((r) => r.body),
    );
  }

  rename(name: string, newName: string): Observable<DetectorRenameResponse> {
    return apiDetectorsNameRenamePut(this.http, this.config.rootUrl, {
      name,
      body: { new_name: newName },
    }).pipe(map((r) => r.body));
  }

  setExamples(
    name: string,
    examples: DetectorExamplesRequest['examples'],
  ): Observable<DetectorExamplesResponse> {
    return apiDetectorsNameExamplesPut(this.http, this.config.rootUrl, {
      name,
      body: { examples },
    }).pipe(map((r) => r.body));
  }

  saveLabels(name: string): Observable<DetectorSaveLabelsResponse> {
    return apiDetectorsNameLabelsPost(this.http, this.config.rootUrl, { name }).pipe(
      map((r) => r.body),
    );
  }

  getLabelsDetail(name: string): Observable<DetectorLabelsDetailResponse> {
    return apiDetectorsNameLabelsDetailGet(this.http, this.config.rootUrl, { name }).pipe(
      map((r) => r.body),
    );
  }

  voteLabelElement(
    name: string,
    elementId: string,
    vote: 'good' | 'bad',
  ): Observable<DetectorLabelVoteResponse> {
    return apiDetectorsNameLabelsElementIdVotePost(this.http, this.config.rootUrl, {
      name,
      element_id: elementId,
      body: { vote },
    }).pipe(map((r) => r.body));
  }

  /** Preview/thumbnail URLs are binary streams — kept as direct URLs so the
   *  active-context query params reach the backend (the generated function
   *  for these declares the success body as ``Error`` because the spec only
   *  carries error responses). */
  labelPreviewUrl(name: string, elementId: string): string {
    return this.activeContext.mediaUrl(
      `/api/detectors/${encodeURIComponent(name)}/labels/${encodeURIComponent(elementId)}/preview`,
    );
  }

  labelThumbnailUrl(name: string, elementId: string): string {
    return this.activeContext.mediaUrl(
      `/api/detectors/${encodeURIComponent(name)}/labels/${encodeURIComponent(elementId)}/thumbnail`,
    );
  }

  combine(
    names: string[],
    newName: string,
    conflictPolicy: DetectorCombineRequest['conflict_policy'] = 'drop',
  ): Observable<DetectorCombineResponse> {
    return apiDetectorsCombinePost(this.http, this.config.rootUrl, {
      body: { names, new_name: newName, conflict_policy: conflictPolicy },
    }).pipe(map((r) => r.body));
  }

  // --- Detector Registry ---

  getRegistry(): Observable<DetectorRegistryListResponse> {
    return apiDetectorsRegistryGet(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  registerDetector(
    params: DetectorRegistryCreateRequest,
  ): Observable<DetectorRegistryCreateResponse> {
    return apiDetectorsRegistryPost(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  /** Plugin-field route — stays on plain ``HttpClient`` because the body
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

  deleteFromRegistry(detectorId: string): Observable<DetectorRegistryDeleteResponse> {
    return apiDetectorsRegistryDetectorIdDelete(this.http, this.config.rootUrl, {
      detector_id: detectorId,
    }).pipe(map((r) => r.body));
  }

  renameInRegistry(
    detectorId: string,
    newName: string,
  ): Observable<DetectorRegistryRenameResponse> {
    return apiDetectorsRegistryDetectorIdRenamePut(this.http, this.config.rootUrl, {
      detector_id: detectorId,
      body: { name: newName },
    }).pipe(map((r) => r.body));
  }

  moveLabelsetSourceFile(
    detectorId: string,
    oldPath: string,
    newPath: string,
  ): Observable<DetectorLabelsetMoveResponse> {
    return apiDetectorsRegistryDetectorIdLabelsetSourceMoveFilePost(
      this.http,
      this.config.rootUrl,
      {
        detector_id: detectorId,
        body: { old_path: oldPath, new_path: newPath },
      },
    ).pipe(map((r) => r.body));
  }

  loadDetector(detectorId: string | null): Observable<DetectorRegistryLoadResponse> {
    return apiDetectorsRegistryLoadPost(this.http, this.config.rootUrl, {
      body: { detector_id: detectorId },
    }).pipe(map((r) => r.body));
  }

  unloadDetector(detectorId: string): Observable<DetectorRegistryUnloadResponse> {
    return apiDetectorsRegistryDetectorIdUnloadPost(this.http, this.config.rootUrl, {
      detector_id: detectorId,
    }).pipe(map((r) => r.body));
  }

  cancelDetectorLoadingTask(taskId: string): Observable<DetectorCancelResponse> {
    return apiDetectorsCancelTaskIdPost(this.http, this.config.rootUrl, {
      task_id: taskId,
    }).pipe(map((r) => r.body));
  }

  setAutorun(detectorId: string, autorun: boolean): Observable<DetectorRegistryAutorunResponse> {
    return apiDetectorsRegistryDetectorIdAutorunPut(this.http, this.config.rootUrl, {
      detector_id: detectorId,
      body: { autorun },
    }).pipe(map((r) => r.body));
  }

  // --- Extractors ---

  getAutorunExtractors(): Observable<AutorunExtractorsListResponse> {
    return apiAutorunExtractorsGet(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  createExtractor(
    params: AutorunExtractorCreateRequest,
  ): Observable<AutorunProcessorCreateResponse> {
    return apiAutorunExtractorsPost(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  deleteExtractor(name: string): Observable<AutorunProcessorDeleteResponse> {
    return apiAutorunExtractorsNameDelete(this.http, this.config.rootUrl, { name }).pipe(
      map((r) => r.body),
    );
  }

  renameExtractor(name: string, newName: string): Observable<AutorunProcessorRenameResponse> {
    return apiAutorunExtractorsNameRenamePut(this.http, this.config.rootUrl, {
      name,
      body: { new_name: newName },
    }).pipe(map((r) => r.body));
  }

  // --- Localizers ---

  getAutorunLocalizers(): Observable<AutorunLocalizersListResponse> {
    return apiAutorunLocalizersGet(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  createLocalizer(
    params: AutorunLocalizerCreateRequest,
  ): Observable<AutorunProcessorCreateResponse> {
    return apiAutorunLocalizersPost(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  deleteLocalizer(name: string): Observable<AutorunProcessorDeleteResponse> {
    return apiAutorunLocalizersNameDelete(this.http, this.config.rootUrl, { name }).pipe(
      map((r) => r.body),
    );
  }

  renameLocalizer(name: string, newName: string): Observable<AutorunProcessorRenameResponse> {
    return apiAutorunLocalizersNameRenamePut(this.http, this.config.rootUrl, {
      name,
      body: { new_name: newName },
    }).pipe(map((r) => r.body));
  }

  // --- Scoring ---

  autoDetect(params: AutoDetectRequest): Observable<AutoDetectResponse> {
    return apiAutoDetectPost(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  extract(params: ExtractRequest): Observable<ExtractResponse> {
    return apiExtractPost(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  autoExtract(): Observable<AutoExtractResponse> {
    return apiAutoExtractPost(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  localize(params: LocalizeRequest): Observable<LocalizeResponse> {
    return apiLocalizePost(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  autoLocalize(): Observable<AutoLocalizeResponse> {
    return apiAutoLocalizePost(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  // --- Find ---

  findCheckLabels(params: FindCheckLabelsRequest): Observable<FindCheckLabelsResponse> {
    return apiFindCheckLabelsPost(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  find(params: FindRequest): Observable<FindResponse> {
    return apiFindPost(this.http, this.config.rootUrl, { body: params }).pipe(map((r) => r.body));
  }

  findLabel(params: FindLabelRequest): Observable<FindLabelResponse> {
    return apiFindLabelPost(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  // --- Pregen processors ---

  getPregenProcessors(): Observable<PregenProcessorsListResponse> {
    return apiPregenProcessorsGet(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  addPregenProcessors(): Observable<PregenProcessorsAddResponse> {
    return apiPregenProcessorsAddPost(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }
}
