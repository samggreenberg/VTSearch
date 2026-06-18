import { Injectable, computed, inject, signal } from '@angular/core';
import { rxResource } from '@angular/core/rxjs-interop';
import type { Media } from '../models/api.models';
import type { MediaIdsListResponse } from '../generated/api-client/models/media-ids-list-response';
import { MediasApiService } from './medias-api.service';
import { MediaMetadataCacheService } from './media-metadata-cache.service';

/**
 * Tracks the dataset-wide list of media stubs (``{id, type, embedder?}``)
 * and the current selection.
 *
 * The stub list is a read path on Angular's reactive resource primitives (see
 * `docs/plans/httpresource-migration.md`): an `rxResource` wraps the existing
 * generated-client method (`MediasApiService.getMediaIds()`), so the typed
 * client and interceptor chain are untouched while the hand-rolled
 * subscribe/`BehaviorSubject` bookkeeping is gone. The public surface is
 * signal-based: read `mediasSignal()` / `isLoading()` / `selectedId()`; call
 * `loadMedias()` / `selectMedia()` / `clear()` to drive it.
 *
 * Each stub is the minimum the UI needs to build the virtual scroller, the
 * stripe overview, and the media-type / embedder gates.  Full per-item
 * metadata (``filename``, ``md5``, ``custom_metadata``, …) is fetched on
 * demand for whatever is currently in the viewport via
 * {@link MediaMetadataCacheService}.
 */
@Injectable({ providedIn: 'root' })
export class MediaStateService {
  private readonly mediasApi = inject(MediasApiService);
  private readonly metadataCache = inject(MediaMetadataCacheService);

  // A monotonic load counter doubles as the resource request. `0` means
  // "not requested yet" -> `params()` returns `undefined` -> the resource stays
  // idle (no fetch). Each `loadMedias()` bumps the counter, which changes the
  // request and so re-runs the loader.
  private readonly loadCount = signal(0);

  private readonly resource = rxResource({
    params: () => (this.loadCount() === 0 ? undefined : this.loadCount()),
    stream: () => this.mediasApi.getMediaIds(),
  });

  /** The dataset-wide media stubs; empty until the first successful load (or after `clear()`). */
  readonly mediasSignal = computed<MediaIdsListResponse[]>(() => this.resource.value() ?? []);

  /** True while ``/api/medias/ids`` is in flight. Drives skeleton loaders in
   *  the media list/grid. */
  readonly isLoading = this.resource.isLoading;

  private readonly selectedIdSignal = signal<number | null>(null);

  /** The currently-selected media id, or `null`. */
  readonly selectedId = this.selectedIdSignal.asReadonly();

  get selectedMedia(): Media | null {
    const id = this.selectedIdSignal();
    if (id === null) return null;
    return this.getMedia(id);
  }

  /**
   * Return the best-available representation of a media item: the cached
   * batch entry (with ``filename`` / ``md5`` / metadata) if loaded, else
   * the stub from the dataset listing, else ``null``.
   */
  getMedia(id: number): Media | null {
    const cached = this.metadataCache.get(id);
    if (cached) return cached;
    return this.mediasSignal().find((m) => m.id === id) ?? null;
  }

  selectMedia(id: number): void {
    this.selectedIdSignal.set(id);
    this.metadataCache.ensureLoaded([id]);
  }

  /** Fetch (or refetch) the dataset media stubs. No-op while a fetch is in flight. */
  loadMedias(): void {
    if (this.resource.isLoading()) return;
    this.loadCount.update((n) => n + 1);
  }

  clear(): void {
    this.resource.set([]);
    this.selectedIdSignal.set(null);
    this.metadataCache.clear();
  }
}
