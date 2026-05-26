import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { ApiConfiguration } from '../generated/api-client/api-configuration';
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
import type { DetectorRenameResponse } from '../generated/api-client/models/detector-rename-response';
import type { DetectorSaveLabelsResponse } from '../generated/api-client/models/detector-save-labels-response';
import type { DetectorsListResponse } from '../generated/api-client/models/detectors-list-response';
import { combineDetectors } from '../generated/api-client/fn/detectors-crud/combine-detectors';
import { createDetector } from '../generated/api-client/fn/detectors-crud/create-detector';
import { deleteDetector } from '../generated/api-client/fn/detectors-crud/delete-detector';
import { getDetector } from '../generated/api-client/fn/detectors-crud/get-detector';
import { listDetectors } from '../generated/api-client/fn/detectors-crud/list-detectors';
import { renameDetector } from '../generated/api-client/fn/detectors-crud/rename-detector';
import { setDetectorExamples } from '../generated/api-client/fn/detectors-crud/set-detector-examples';
import { getDetectorLabelsDetail } from '../generated/api-client/fn/detectors-labels/get-detector-labels-detail';
import { saveDetectorLabels } from '../generated/api-client/fn/detectors-labels/save-detector-labels';
import { voteDetectorLabel } from '../generated/api-client/fn/detectors-labels/vote-detector-label';
import { ActiveContextService } from './active-context.service';

/** Detector CRUD: list/create/get/delete/rename/labels/combine/examples. */
@Injectable({ providedIn: 'root' })
export class DetectorsCrudApiService {
  private http = inject(HttpClient);
  private config = inject(ApiConfiguration);
  private activeContext = inject(ActiveContextService);

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
}
