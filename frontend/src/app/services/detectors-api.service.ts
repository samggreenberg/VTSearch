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
import type { FindCancelResponse } from '../generated/api-client/models/find-cancel-response';
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
import { autoDetect } from '../generated/api-client/fn/detector-scoring/auto-detect';
import { autoExtract } from '../generated/api-client/fn/processors-scoring/auto-extract';
import { autoLocalize } from '../generated/api-client/fn/processors-scoring/auto-localize';
import { getAutorunExtractorsRoute } from '../generated/api-client/fn/processors-crud/get-autorun-extractors-route';
import { deleteAutorunExtractorRoute } from '../generated/api-client/fn/processors-crud/delete-autorun-extractor-route';
import { renameAutorunExtractorRoute } from '../generated/api-client/fn/processors-crud/rename-autorun-extractor-route';
import { addAutorunExtractorRoute } from '../generated/api-client/fn/processors-crud/add-autorun-extractor-route';
import { getAutorunLocalizersRoute } from '../generated/api-client/fn/processors-crud/get-autorun-localizers-route';
import { deleteAutorunLocalizerRoute } from '../generated/api-client/fn/processors-crud/delete-autorun-localizer-route';
import { renameAutorunLocalizerRoute } from '../generated/api-client/fn/processors-crud/rename-autorun-localizer-route';
import { addAutorunLocalizerRoute } from '../generated/api-client/fn/processors-crud/add-autorun-localizer-route';
import { cancelDetectorLoadingTask } from '../generated/api-client/fn/detectors-registry/cancel-detector-loading-task';
import { combineDetectors } from '../generated/api-client/fn/detectors-crud/combine-detectors';
import { listDetectors } from '../generated/api-client/fn/detectors-crud/list-detectors';
import { deleteDetector } from '../generated/api-client/fn/detectors-crud/delete-detector';
import { setDetectorExamples } from '../generated/api-client/fn/detectors-crud/set-detector-examples';
import { getDetector } from '../generated/api-client/fn/detectors-crud/get-detector';
import { getDetectorLabelsDetail } from '../generated/api-client/fn/detectors-labels/get-detector-labels-detail';
import { voteDetectorLabel } from '../generated/api-client/fn/detectors-labels/vote-detector-label';
import { saveDetectorLabels } from '../generated/api-client/fn/detectors-labels/save-detector-labels';
import { renameDetector } from '../generated/api-client/fn/detectors-crud/rename-detector';
import { createDetector } from '../generated/api-client/fn/detectors-crud/create-detector';
import { setDetectorAutorun } from '../generated/api-client/fn/detectors-registry/set-detector-autorun';
import { deleteRegisteredDetector } from '../generated/api-client/fn/detectors-registry/delete-registered-detector';
import { moveLabelsetSourceFile } from '../generated/api-client/fn/detectors-registry/move-labelset-source-file';
import { renameRegisteredDetector } from '../generated/api-client/fn/detectors-registry/rename-registered-detector';
import { unloadDetectorRoute } from '../generated/api-client/fn/detectors-registry/unload-detector-route';
import { listRegisteredDetectors } from '../generated/api-client/fn/detectors-registry/list-registered-detectors';
import { loadDetectorRoute } from '../generated/api-client/fn/detectors-registry/load-detector-route';
import { registerDetectorRoute } from '../generated/api-client/fn/detectors-registry/register-detector-route';
import { runExtract } from '../generated/api-client/fn/processors-scoring/run-extract';
import { cancelFind } from '../generated/api-client/fn/detector-find/cancel-find';
import { findCheckLabels } from '../generated/api-client/fn/detector-find/find-check-labels';
import { findLabel } from '../generated/api-client/fn/detector-scoring/find-label';
import { multiFind } from '../generated/api-client/fn/detector-find/multi-find';
import { runLocalize } from '../generated/api-client/fn/processors-scoring/run-localize';
import { addPregenProcessors } from '../generated/api-client/fn/processors-crud/add-pregen-processors';
import { listPregenProcessors } from '../generated/api-client/fn/processors-crud/list-pregen-processors';
import { ActiveContextService } from './active-context.service';
import { VtDialogService } from './dialog.service';

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
    return listDetectors(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  create(params: DetectorCreateRequest): Observable<DetectorCreateResponse> {
    return createDetector(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  get(name: string): Observable<DetectorDetail> {
    return getDetector(this.http, this.config.rootUrl, { name }).pipe(map((r) => r.body));
  }

  delete(name: string): Observable<DetectorDeleteResponse> {
    return deleteDetector(this.http, this.config.rootUrl, { name }).pipe(
      map((r) => r.body),
    );
  }

  rename(name: string, newName: string): Observable<DetectorRenameResponse> {
    return renameDetector(this.http, this.config.rootUrl, {
      name,
      body: { new_name: newName },
    }).pipe(map((r) => r.body));
  }

  setExamples(
    name: string,
    examples: DetectorExamplesRequest['examples'],
  ): Observable<DetectorExamplesResponse> {
    return setDetectorExamples(this.http, this.config.rootUrl, {
      name,
      body: { examples },
    }).pipe(map((r) => r.body));
  }

  saveLabels(name: string): Observable<DetectorSaveLabelsResponse> {
    return saveDetectorLabels(this.http, this.config.rootUrl, { name }).pipe(
      map((r) => r.body),
    );
  }

  getLabelsDetail(name: string): Observable<DetectorLabelsDetailResponse> {
    return getDetectorLabelsDetail(this.http, this.config.rootUrl, { name }).pipe(
      map((r) => r.body),
    );
  }

  voteLabelElement(
    name: string,
    elementId: string,
    vote: 'good' | 'bad',
  ): Observable<DetectorLabelVoteResponse> {
    return voteDetectorLabel(this.http, this.config.rootUrl, {
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
    return combineDetectors(this.http, this.config.rootUrl, {
      body: { names, new_name: newName, conflict_policy: conflictPolicy },
    }).pipe(map((r) => r.body));
  }

  // --- Detector Registry ---

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
    return deleteRegisteredDetector(this.http, this.config.rootUrl, {
      detector_id: detectorId,
    }).pipe(map((r) => r.body));
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

  moveLabelsetSourceFile(
    detectorId: string,
    oldPath: string,
    newPath: string,
  ): Observable<DetectorLabelsetMoveResponse> {
    return moveLabelsetSourceFile(
      this.http,
      this.config.rootUrl,
      {
        detector_id: detectorId,
        body: { old_path: oldPath, new_path: newPath },
      },
    ).pipe(map((r) => r.body));
  }

  /** Prompt the user to move an orphaned labelset file after a rename.
   *  Generated ``PendingLabelsetMove`` arrives as ``{old_path,new_path} | {} | null``
   *  (a marshmallow ``allow_none`` artifact); narrow it here so callers
   *  pass the rename response through unchanged. */
  async promptMoveOrphanedLabelsetFile(
    dialog: VtDialogService,
    detectorId: string,
    pending: unknown,
  ): Promise<void> {
    if (
      !pending ||
      typeof pending !== 'object' ||
      !('old_path' in pending) ||
      !('new_path' in pending) ||
      typeof (pending as { old_path: unknown }).old_path !== 'string' ||
      typeof (pending as { new_path: unknown }).new_path !== 'string'
    ) {
      return;
    }
    const { old_path: oldPath, new_path: newPath } = pending as { old_path: string; new_path: string };
    const ok = await dialog.confirm(
      `Move existing labelset file "${oldPath}" to "${newPath}"?`,
    );
    if (!ok) return;
    this.moveLabelsetSourceFile(detectorId, oldPath, newPath).subscribe();
  }

  loadDetector(detectorId: string | null): Observable<DetectorRegistryLoadResponse> {
    return loadDetectorRoute(this.http, this.config.rootUrl, {
      body: { detector_id: detectorId },
    }).pipe(map((r) => r.body));
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

  setAutorun(detectorId: string, autorun: boolean): Observable<DetectorRegistryAutorunResponse> {
    return setDetectorAutorun(this.http, this.config.rootUrl, {
      detector_id: detectorId,
      body: { autorun },
    }).pipe(map((r) => r.body));
  }

  // --- Extractors ---

  getAutorunExtractors(): Observable<AutorunExtractorsListResponse> {
    return getAutorunExtractorsRoute(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  createExtractor(
    params: AutorunExtractorCreateRequest,
  ): Observable<AutorunProcessorCreateResponse> {
    return addAutorunExtractorRoute(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  deleteExtractor(name: string): Observable<AutorunProcessorDeleteResponse> {
    return deleteAutorunExtractorRoute(this.http, this.config.rootUrl, { name }).pipe(
      map((r) => r.body),
    );
  }

  renameExtractor(name: string, newName: string): Observable<AutorunProcessorRenameResponse> {
    return renameAutorunExtractorRoute(this.http, this.config.rootUrl, {
      name,
      body: { new_name: newName },
    }).pipe(map((r) => r.body));
  }

  // --- Localizers ---

  getAutorunLocalizers(): Observable<AutorunLocalizersListResponse> {
    return getAutorunLocalizersRoute(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  createLocalizer(
    params: AutorunLocalizerCreateRequest,
  ): Observable<AutorunProcessorCreateResponse> {
    return addAutorunLocalizerRoute(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  deleteLocalizer(name: string): Observable<AutorunProcessorDeleteResponse> {
    return deleteAutorunLocalizerRoute(this.http, this.config.rootUrl, { name }).pipe(
      map((r) => r.body),
    );
  }

  renameLocalizer(name: string, newName: string): Observable<AutorunProcessorRenameResponse> {
    return renameAutorunLocalizerRoute(this.http, this.config.rootUrl, {
      name,
      body: { new_name: newName },
    }).pipe(map((r) => r.body));
  }

  // --- Scoring ---

  autoDetect(params: AutoDetectRequest): Observable<AutoDetectResponse> {
    return autoDetect(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  extract(params: ExtractRequest): Observable<ExtractResponse> {
    return runExtract(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  autoExtract(): Observable<AutoExtractResponse> {
    return autoExtract(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  localize(params: LocalizeRequest): Observable<LocalizeResponse> {
    return runLocalize(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  autoLocalize(): Observable<AutoLocalizeResponse> {
    return autoLocalize(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  // --- Find ---

  findCheckLabels(params: FindCheckLabelsRequest): Observable<FindCheckLabelsResponse> {
    return findCheckLabels(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  find(params: FindRequest): Observable<FindResponse> {
    return multiFind(this.http, this.config.rootUrl, { body: params }).pipe(map((r) => r.body));
  }

  findLabel(params: FindLabelRequest): Observable<FindLabelResponse> {
    return findLabel(this.http, this.config.rootUrl, { body: params }).pipe(
      map((r) => r.body),
    );
  }

  /** Cancel any in-flight find / find-label / auto-detect scoring. */
  cancelFind(): Observable<FindCancelResponse> {
    return cancelFind(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  // --- Pregen processors ---

  getPregenProcessors(): Observable<PregenProcessorsListResponse> {
    return listPregenProcessors(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }

  addPregenProcessors(): Observable<PregenProcessorsAddResponse> {
    return addPregenProcessors(this.http, this.config.rootUrl).pipe(map((r) => r.body));
  }
}
