import { Injectable, Signal, computed, inject, signal } from '@angular/core';
import { Observable, concat, from, of } from 'rxjs';
import {
  EvenOutChoice,
  EvenOutChoices,
  PixelMeasure,
  evenOutChoices,
  innerFaceBox,
  measurePixels,
} from '../components/center-panel/face-even-out/even-out';
import { FaceLandmarks } from '../components/center-panel/image-viewer/face-link';
import { SettingsStateService } from './settings-state.service';

/** What a picture is measured at: its long side drawn this many pixels. Enough
 *  for an inner face of a few thousand pixels, and quick to read. */
const MEASURE_SIZE = 192;

/** How many pictures' measures are kept, so stepping back to a face or a
 *  reference reads nothing again. */
const CACHE_SIZE = 256;

/** A picture to measure: its URL, and the landmarks that place its inner face. */
export interface MeasureRequest {
  readonly src: string;
  readonly landmarks: FaceLandmarks | null | undefined;
}

/** A string naming *request*, for comparing requests by value. */
export function measureKey(request: MeasureRequest): string {
  return `${request.src}\n${JSON.stringify(request.landmarks ?? null)}`;
}

/**
 * Evening out how the Face compare layout's two faces look (#5674, #5675):
 * the user's choices, the held key that shows both originals, and the reading
 * of each picture's pixels that the map is fitted to (`even-out.ts`).
 *
 * The choices are the per-user `face_even_out` setting. `SettingsStateService`
 * writes the store only once the PUT returns, so a click is held here until
 * the server echoes it, as `PanelHideStateService` holds a panel's fold.
 *
 * A picture is measured from its own URL, drawn small on a canvas: every
 * picture is same-origin (`/api/medias/<id>/image`, or an example's own
 * route), so its pixels can be read. A face is always measured on its crop,
 * even while Wider shows its photo: the map is global, so what the crop says
 * applies unchanged to the photo it was cut from.
 */
@Injectable({ providedIn: 'root' })
export class FaceEvenOutService {
  private readonly settingsState = inject(SettingsStateService);

  private readonly pending = signal<EvenOutChoices | null>(null);
  private seq = 0;

  /** The choices in force: a click still on its way to the server, or the setting over the defaults. */
  readonly choices: Signal<EvenOutChoices> = computed(
    () => this.pending() ?? evenOutChoices(this.settingsState.settingsSignal()?.face_even_out),
  );

  /** The held key is down: both faces are shown as taken (ENFSI 5.2.3, "refer back to the original"). */
  readonly showOriginals = signal(false);

  /** Each picture's read, and its answer once it has one. */
  private readonly measures = new Map<string, MeasureEntry>();

  /** Tick or untick *choice*, now, and remember it. */
  set(choice: EvenOutChoice, on: boolean): void {
    const next = { ...this.choices(), [choice]: on };
    const n = ++this.seq;
    this.pending.set(next);
    const settle = () => {
      if (this.seq === n) this.pending.set(null);
    };
    this.settingsState.update({ face_even_out: next }).subscribe({ next: settle, error: settle });
  }

  /**
   * What *request*'s picture's pixels say: null at once, then the measure
   * once the picture has loaded, or null again if it cannot be read. A
   * picture already read answers at once, so stepping back to a face shows
   * it evened out from its first frame. No request is null.
   *
   * A plain observable rather than a resource, so a picture that never loads
   * (jsdom loads none) holds up nothing that waits for the app to settle.
   */
  measured(request: MeasureRequest | undefined): Observable<PixelMeasure | null> {
    if (!request) return of(null);
    const key = measureKey(request);
    let entry = this.measures.get(key);
    if (entry) {
      // Most recently used last, so the oldest is the one let go.
      this.measures.delete(key);
    } else {
      const promise = readPixels(request);
      const created: MeasureEntry = { promise, settled: false, value: null };
      entry = created;
      void promise.then((value) => {
        created.settled = true;
        created.value = value;
        // A failed read is not kept: the next ask tries again.
        if (value === null && this.measures.get(key) === created) this.measures.delete(key);
      });
    }
    this.measures.set(key, entry);
    while (this.measures.size > CACHE_SIZE) this.measures.delete(this.measures.keys().next().value!);
    return entry.settled ? of(entry.value) : concat(of(null), from(entry.promise));
  }
}

interface MeasureEntry {
  readonly promise: Promise<PixelMeasure | null>;
  settled: boolean;
  value: PixelMeasure | null;
}

/** Load *request*'s picture, draw it small and measure it. */
function readPixels(request: MeasureRequest): Promise<PixelMeasure | null> {
  if (typeof Image === 'undefined' || typeof document === 'undefined') return Promise.resolve(null);
  return new Promise((resolve) => {
    const img = new Image();
    img.decoding = 'async';
    img.onload = () => resolve(measureImage(img, request.landmarks));
    img.onerror = () => resolve(null);
    img.src = request.src;
  });
}

function measureImage(img: HTMLImageElement, landmarks: FaceLandmarks | null | undefined): PixelMeasure | null {
  const w = img.naturalWidth;
  const h = img.naturalHeight;
  if (!(w > 0 && h > 0)) return null;
  const scale = Math.min(1, MEASURE_SIZE / Math.max(w, h));
  const width = Math.max(1, Math.round(w * scale));
  const height = Math.max(1, Math.round(h * scale));
  try {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return null;
    ctx.drawImage(img, 0, 0, width, height);
    const { data } = ctx.getImageData(0, 0, width, height);
    return measurePixels(data, width, height, innerFaceBox(landmarks, width, height));
  } catch {
    // A canvas the browser will not let be read: show the picture as taken.
    return null;
  }
}

