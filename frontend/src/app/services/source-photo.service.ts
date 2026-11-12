import { HttpContext, HttpErrorResponse } from '@angular/common/http';
import { Injectable, computed, inject, signal } from '@angular/core';
import { Observable, catchError, map, of } from 'rxjs';

import type { OutlineBox, RegionBox } from '../components/center-panel/image-viewer/image-viewer.component';
import { SKIP_ERROR_TOAST } from '../interceptors/error.interceptor';
import { Media } from '../models/api.models';
import { importSiblingsOfType } from '../utils/import-siblings';
import { ActiveDatasetService } from './active-dataset.service';
import { DatasetStateService } from './dataset-state.service';
import { MediaTypeCapabilityService } from './media-type-capability.service';
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

/**
 * How much wider than its crop the Face compare layout's **Wider** view shows
 * a face (#5645): this many times the crop's width and height, about the same
 * centre. With the converter's default padding of 0.25 a crop is 1.5 times the
 * face box, so this is three times the face box: the whole head, ears and hair
 * included, and the top of the shoulders.
 */
export const WIDER_SCALE = 2;

/** *box* scaled by *scale* about its own centre. Not clamped to the photo: a
 *  face at the photo's edge is shown at the same size as one in the middle,
 *  and the viewer's pan keeps the photo's edge at the pane's. */
export function widerBox(box: RegionBox, scale = WIDER_SCALE): RegionBox {
  const [x0, y0, x1, y1] = box;
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;
  const hw = ((x1 - x0) * scale) / 2;
  const hh = ((y1 - y0) * scale) / 2;
  return [cx - hw, cy - hh, cx + hw, cy + hh];
}

/**
 * What one pane of the Face compare layout shows while **Wider** is on
 * (#5645).
 *
 * - `crop`: the face as stored. Wider is off, or this face's photo cannot be
 *   shown: an example picture, a face with no photo or no recorded box, a
 *   sibling not loaded, a failed lookup.
 * - `pending`: the lookup is out. The pane shows nothing yet, rather than the
 *   crop for a moment and then the photo.
 * - `photo`: the photo, framed on {@link widerBox} of the face's box in it.
 */
export type WiderView =
  | { readonly kind: 'crop' }
  | { readonly kind: 'pending' }
  | { readonly kind: 'photo'; readonly datasetId: string; readonly media: Media; readonly fitBox: RegionBox };

const CROP: WiderView = { kind: 'crop' };

/**
 * Resolve one pane's {@link WiderView} from its source lookup: *loading* while
 * the lookup is out, *lookup* once it is back (undefined before it was asked).
 * *name* is the face's file name, which the photo's viewer is given as its
 * alt text.
 */
export function widerView(lookup: SourceLookup | undefined, loading: boolean, name?: string): WiderView {
  if (loading) return { kind: 'pending' };
  if (lookup?.kind !== 'found' || !lookup.box) return CROP;
  return {
    kind: 'photo',
    datasetId: lookup.datasetId,
    media: { id: lookup.mediaId, media_type: 'image', ...(name ? { filename: name } : {}) },
    fitBox: widerBox(lookup.box),
  };
}

/** The Show in photo overlay the user asked for. */
export interface SourcePhotoRequest {
  /** The item (a face crop) whose photo to show, in the active dataset. */
  mediaId: number;
  /** The active dataset's name when it was asked for: where Back returns to. */
  fromName: string;
  /**
   * Several faces to outline on the photo, each by whose it is, in place of
   * the one box the lookup returns for *mediaId*: a photo two people share,
   * from the People view. Boxes are normalised to that photo. Absent or
   * empty, the overlay outlines *mediaId*'s own face, as Show in photo
   * always has.
   */
  boxes?: readonly OutlineBox[];
}

/**
 * Show in photo (#4750): from a face crop, its source photo with the face
 * outlined.
 *
 * The photo is an item of the Image dataset the same import produced, and it
 * is shown in an overlay over the Train / Tune view rather than by switching
 * the active pair: the face detector is no detector for an Image dataset, so a
 * switch would land on the incompatible-pair explainer, and the pair reset
 * would lose the face ranking. The overlay reads the photo with the sibling's
 * id alone (`DATASET_OVERRIDE`, `ActiveContextService.mediaUrlIn`), never with
 * the active detector's.
 *
 * Holds which overlay is open, so the centre pane's button and the list's
 * right-click menu open the same one; `vt-center-panel` renders it. Browse
 * offers it too, from the bin details and the selection panel, and
 * `vt-browse-view` renders it there.
 */
@Injectable({ providedIn: 'root' })
export class SourcePhotoService {
  private readonly mediasApi = inject(MediasApiService);
  private readonly activeDataset = inject(ActiveDatasetService);
  private readonly datasetState = inject(DatasetStateService);
  private readonly mediaTypes = inject(MediaTypeCapabilityService);

  /**
   * Whether to offer Show in photo on the active dataset: its items all came
   * out of a converter (a convert-in type such as `face`), and its import also
   * produced an Image dataset, where the photos are. That leaves out a
   * Document dataset beside an Image one, whose pages have no source item.
   * An item-level answer is {@link lookup}'s; this keeps every other dataset
   * from asking per item. False until the media-type registry has loaded
   * (`MediaTypeCapabilityService.ensureLoaded`, which `vt-center-panel` calls).
   */
  readonly offered = computed(() => {
    const dataset = this.activeDataset.dataset();
    if (!dataset || !this.mediaTypes.isConvertIn(dataset.media_type)) return false;
    return importSiblingsOfType(dataset, this.datasetState.datasets, 'image').length > 0;
  });

  /**
   * {@link offered}, and the Image dataset with the photos is in memory, so a
   * lookup can find a photo rather than answer 409. The Face compare layout's
   * Wider toggle (#5645) shows only then: it is the dataset's state, so the
   * toggle does not come and go with each face's lookup. The registry is
   * re-read when a load finishes (`DashboardLoadingTasksService`), which turns
   * this on after Show in photo's Load button.
   */
  readonly photosLoaded = computed(() => {
    if (!this.offered()) return false;
    const siblings = importSiblingsOfType(this.activeDataset.dataset(), this.datasetState.datasets, 'image');
    return siblings.some((d) => d.loaded === true);
  });

  private readonly _request = signal<SourcePhotoRequest | null>(null);
  /** The open overlay, or null. */
  readonly request = this._request.asReadonly();

  /**
   * Open the overlay for *mediaId*, an item of the active dataset. *boxes*
   * outline several faces by whose they are, in place of *mediaId*'s own
   * ({@link SourcePhotoRequest.boxes}); *fromName* names where Back returns
   * to, the active dataset by default.
   */
  open(mediaId: number, options: { boxes?: readonly OutlineBox[]; fromName?: string } = {}): void {
    this._request.set({
      mediaId,
      fromName: options.fromName || this.activeDataset.datasetName(),
      ...(options.boxes?.length ? { boxes: options.boxes } : {}),
    });
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
