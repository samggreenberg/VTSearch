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

  /**
   * Fetch (or refetch) the dataset media stubs.
   *
   * Always restarts the loader, even mid-flight. An in-flight request carries
   * the `X-Dataset-Id` header of the pair that was active when it was
   * dispatched, so on a rapid A->B->C pair switch its response is the *wrong*
   * dataset's id list; skipping the refetch would leave that stale list
   * rendered against the new pair forever (nothing else re-fetches). Bumping
   * `loadCount` changes the resource request, which makes `rxResource`
   * unsubscribe from the stale stream (cancelling the request) and re-run the
   * loader with the current headers.
   */
  loadMedias(): void {
    this.loadCount.update((n) => n + 1);
  }

  /**
   * Drop the selection without touching the stub list or the metadata cache.
   *
   * Media ids are *per-dataset*, so a selection is pair-scoped state: it is
   * only meaningful against the pair that installed it. Carrying it across a
   * pair change strands the centre viewer on an item from the pair we left —
   * silently, because `ActiveContextService.mediaUrl` stamps the dataset id
   * into the `<img src>` at build time and the viewers only rebuild that src
   * when the media *id* changes (see `ImageViewerComponent.lastMediaId` and
   * its four siblings). The stale item therefore keeps resolving against its
   * own dataset and renders happily instead of 404-ing. See #3489.
   *
   * Separate from {@link clear} because the pair-change reset wants only this
   * half: it re-fetches the stub list (rather than blanking it, which would
   * flash the grid empty) and the metadata cache is already dataset-keyed.
   */
  clearSelection(): void {
    this.selectedIdSignal.set(null);
  }

  clear(): void {
    this.resource.set([]);
    this.selectedIdSignal.set(null);
    this.metadataCache.clear();
  }
}
