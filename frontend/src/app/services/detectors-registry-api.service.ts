import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
import type { DetectorCancelResponse } from '../generated/api-client/models/detector-cancel-response';
import type { DetectorLabelsetMoveResponse } from '../generated/api-client/models/detector-labelset-move-response';
import type { DetectorRegistryAutorunResponse } from '../generated/api-client/models/detector-registry-autorun-response';
import type { DetectorRegistryCreateRequest } from '../generated/api-client/models/detector-registry-create-request';
import type { DetectorRegistryCreateResponse } from '../generated/api-client/models/detector-registry-create-response';
import type { DetectorRegistryDeleteResponse } from '../generated/api-client/models/detector-registry-delete-response';
import type { DetectorRegistryListResponse } from '../generated/api-client/models/detector-registry-list-response';
import type { DetectorRegistryLoadResponse } from '../generated/api-client/models/detector-registry-load-response';
import type { DetectorRegistryRenameResponse } from '../generated/api-client/models/detector-registry-rename-response';
import type { DetectorRegistryUnloadResponse } from '../generated/api-client/models/detector-registry-unload-response';
import { cancelDetectorLoadingTask } from '../generated/api-client/fn/detectors-registry/cancel-detector-loading-task';
import { deleteRegisteredDetector } from '../generated/api-client/fn/detectors-registry/delete-registered-detector';
import { listRegisteredDetectors } from '../generated/api-client/fn/detectors-registry/list-registered-detectors';
import { loadDetectorRoute } from '../generated/api-client/fn/detectors-registry/load-detector-route';
import { moveLabelsetSourceFile } from '../generated/api-client/fn/detectors-registry/move-labelset-source-file';
import { registerDetectorRoute } from '../generated/api-client/fn/detectors-registry/register-detector-route';
import { renameRegisteredDetector } from '../generated/api-client/fn/detectors-registry/rename-registered-detector';
import { setDetectorAutorun } from '../generated/api-client/fn/detectors-registry/set-detector-autorun';
import { unloadDetectorRoute } from '../generated/api-client/fn/detectors-registry/unload-detector-route';
import { VtDialogService } from './dialog.service';

/** Detector registry: list / load / unload / rename / cancel / autorun /
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
}
