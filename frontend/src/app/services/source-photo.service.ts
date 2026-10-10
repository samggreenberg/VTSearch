import { HttpContext, HttpErrorResponse } from '@angular/common/http';
import { Injectable, computed, inject, signal } from '@angular/core';
import { Observable, catchError, map, of } from 'rxjs';

import type { RegionBox } from '../components/center-panel/image-viewer/image-viewer.component';
import { SKIP_ERROR_TOAST } from '../interceptors/error.interceptor';
import { importSiblingsOfType } from '../utils/import-siblings';
import { ActiveDatasetService } from './active-dataset.service';
import { DatasetStateService } from './dataset-state.service';
import { MediasApiService } from './medias-api.service';

/**
 * Where a converter output's source item is, as `GET /api/medias/<id>/source`
 * answered (#4749).
 *
 * - `found`: the item in the sibling dataset, with the output's box in it.
 * - `not-loaded`: the sibling that holds it is registered but not loaded (409).
 * - `none`: there is no source item to show (404: not a converter output, no
 *   readable sibling, or no matching item); `message` says which.
 * - `error`: the lookup itself failed.
 */
export type SourceLookup =
  | { kind: 'found'; datasetId: string; mediaId: number; box: RegionBox | null }
  | { kind: 'not-loaded'; datasetId: string }
  | { kind: 'none'; message: string }
  | { kind: 'error'; message: string };

/** The Show in photo overlay the user asked for. */
export interface SourcePhotoRequest {
  /** The item (a face crop) whose photo to show, in the active dataset. */
  mediaId: number;
  /** The active dataset's name when it was asked for: where Back returns to. */
  fromName: string;
}

/**
 * Show in photo (#4750): from a face crop, its source photo with the face
 * outlined.
 *
 * The photo is an item of the Image dataset the same import produced, and it
 * is shown in an overlay over the Train / Test view rather than by switching
 * the active pair: the face detector is no detector for an Image dataset, so a
 * switch would land on the incompatible-pair explainer, and the pair reset
 * would lose the face ranking. The overlay reads the photo with the sibling's
 * id alone (`DATASET_OVERRIDE`, `ActiveContextService.mediaUrlIn`), never with
 * the active detector's.
 *
 * Holds which overlay is open, so the centre pane's button and the list's
 * right-click menu open the same one; `vt-center-panel` renders it.
 */
@Injectable({ providedIn: 'root' })
export class SourcePhotoService {
  private readonly mediasApi = inject(MediasApiService);
  private readonly activeDataset = inject(ActiveDatasetService);
  private readonly datasetState = inject(DatasetStateService);

  /**
   * Whether to offer Show in photo on the active dataset: its import also
   * produced an Image dataset, where the photos are. An item-level answer is
   * {@link lookup}'s; this keeps every other dataset from asking per item.
   */
  readonly offered = computed(
    () => importSiblingsOfType(this.activeDataset.dataset(), this.datasetState.datasets, 'image').length > 0,
  );

  private readonly _request = signal<SourcePhotoRequest | null>(null);
  /** The open overlay, or null. */
  readonly request = this._request.asReadonly();

  /** Open the overlay for *mediaId*, an item of the active dataset. */
  open(mediaId: number): void {
    this._request.set({ mediaId, fromName: this.activeDataset.datasetName() });
  }

  close(): void {
    this._request.set(null);
  }

  /**
   * Ask where *mediaId*'s source item is. Never errors and never toasts: a 404
   * or a 409 is an answer here, not a failure.
   */
  lookup(mediaId: number): Observable<SourceLookup> {
    return this.mediasApi.getSource(mediaId, new HttpContext().set(SKIP_ERROR_TOAST, true)).pipe(
      map((r): SourceLookup => ({
        kind: 'found',
        datasetId: r.dataset_id,
        mediaId: r.media_id,
        box: toBox(r.box),
      })),
      catchError((err: unknown) => of(lookupFailure(err))),
    );
  }
}

function toBox(box: readonly number[] | null | undefined): RegionBox | null {
  return box && box.length === 4 ? [box[0], box[1], box[2], box[3]] : null;
}

function lookupFailure(err: unknown): SourceLookup {
  if (!(err instanceof HttpErrorResponse)) return { kind: 'error', message: String(err) };
  const body = (err.error && typeof err.error === 'object' ? err.error : {}) as Record<string, unknown>;
  const message = typeof body['message'] === 'string' ? body['message'] : err.message;
  if (err.status === 409 && body['error_code'] === 'source_not_loaded' && typeof body['dataset_id'] === 'string') {
    return { kind: 'not-loaded', datasetId: body['dataset_id'] };
  }
  if (err.status === 404) return { kind: 'none', message };
  return { kind: 'error', message };
}
