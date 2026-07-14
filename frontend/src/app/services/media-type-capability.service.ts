import { Injectable, Injector, computed, inject, signal } from '@angular/core';

import { DatasetsListingsApiService } from './datasets-listings-api.service';
import type { MediaTypeInfo } from '../models/api.models';

/**
 * The thumbnail-backed media types, used as the default answer until the served
 * ``GET /api/media-types`` registry resolves (and as the answer if it fails to
 * load). This is the **only** hardcoded copy of the set left in the frontend;
 * every per-item helper reads {@link MediaTypeCapabilityService.usesThumbnails}
 * instead, so once the registry loads a new thumbnail type flips on purely from
 * the backend ``MediaType.has_thumbnail`` capability — no frontend edit needed.
 * Keeping the fallback in sync with the backend only matters for the sub-second
 * window before the registry arrives (it avoids a flash of placeholder icons).
 */
export const FALLBACK_THUMBNAIL_TYPE_IDS: ReadonlySet<string> = new Set([
  'image',
  'video',
  'document',
  'audio',
]);

/**
 * Process-wide cache of media-type capability metadata (the
 * ``GET /api/media-types`` registry), so callers can answer "does this media
 * type have a browsable thumbnail?" from the served ``has_thumbnail`` field
 * instead of each reimplementing the ``{image, video, document, audio}`` set.
 *
 * A feature component that renders thumbnails calls {@link ensureLoaded} once
 * (mirrors {@link EmbedderCapabilityService}); until the registry resolves (and
 * if it fails to load) {@link usesThumbnails} answers from
 * {@link FALLBACK_THUMBNAIL_TYPE_IDS} so behaviour is unchanged during the load
 * window. The API service is resolved lazily inside {@link ensureLoaded} rather
 * than injected in the constructor, so leaf components can read
 * {@link usesThumbnails} without pulling ``HttpClient`` into their injector.
 */
@Injectable({ providedIn: 'root' })
export class MediaTypeCapabilityService {
  private readonly injector = inject(Injector);

  /** Loaded media-type metadata, or ``null`` until the registry resolves. */
  readonly infos = signal<MediaTypeInfo[] | null>(null);
  private loading = false;

  /** Type ids that have a browsable thumbnail: the served set once the registry
   *  has loaded, or the fallback until then. */
  private readonly thumbnailTypeIds = computed<ReadonlySet<string>>(() => {
    const infos = this.infos();
    if (!infos) return FALLBACK_THUMBNAIL_TYPE_IDS;
    return new Set(infos.filter((m) => m.has_thumbnail === true).map((m) => m.type_id));
  });

  /** Kick off a one-time load of the media-type registry. Idempotent. */
  ensureLoaded(): void {
    if (this.infos() !== null || this.loading) return;
    this.loading = true;
    this.injector.get(DatasetsListingsApiService).getMediaTypes().subscribe({
      next: (r) => {
        this.infos.set(r.media_types ?? []);
        this.loading = false;
      },
      error: () => {
        this.infos.set([]);
        this.loading = false;
      },
    });
  }

  /** True once the registry has resolved (success or failure). */
  get loaded(): boolean {
    return this.infos() !== null;
  }

  /**
   * Whether items of *mediaType* have a browsable thumbnail — image/video/
   * document tiles, or audio's waveform PNG. Data-driven from the served
   * ``has_thumbnail`` field; falls back to {@link FALLBACK_THUMBNAIL_TYPE_IDS}
   * until the registry loads so no thumbnail flashes to a placeholder. This is a
   * pure read (no side effects); a feature component must call
   * {@link ensureLoaded} for the served set to override the fallback.
   */
  usesThumbnails(mediaType: string): boolean {
    return this.thumbnailTypeIds().has(mediaType);
  }
}
