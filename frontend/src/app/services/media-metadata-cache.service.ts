import { Injectable, OnDestroy } from '@angular/core';
import { BehaviorSubject, Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { MediaItem } from '../models/api.models';
import { MediasApiService } from './medias-api.service';

/**
 * Default batch size for metadata requests.  Larger batches reduce HTTP round
 * trips; smaller batches reduce per-request latency.
 */
const BATCH_SIZE = 200;

/**
 * Caches media metadata (id, type, filename, md5, etc.) and loads it lazily in
 * batches via `POST /api/medias/batch`.
 *
 * The full `/api/medias` response can be hundreds of megabytes for large
 * datasets.  This cache fetches only the metadata the UI actually needs —
 * typically the items currently visible in the virtual-scrolling viewport.
 *
 * Usage:
 *   1. Call `ensureLoaded(ids)` with the IDs the viewport needs.
 *   2. Subscribe to `medias$` or call `get(id)` for cached items.
 *   3. `toMediaItems(ids)` returns MediaItem[] for already-cached IDs (skips
 *      unknown ones so the template can render immediately).
 */
@Injectable({ providedIn: 'root' })
export class MediaMetadataCacheService implements OnDestroy {
  private readonly cache = new Map<number, MediaItem>();
  private readonly pendingIds = new Set<number>();
  private batchTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly destroy$ = new Subject<void>();

  /** Emits the current cache version (increments on every batch arrival). */
  private readonly versionSubject = new BehaviorSubject<number>(0);
  readonly version$ = this.versionSubject.asObservable();

  constructor(private mediasApi: MediasApiService) {}

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
    if (this.batchTimer) clearTimeout(this.batchTimer);
  }

  /** Return a cached MediaItem or undefined if not yet fetched. */
  get(id: number): MediaItem | undefined {
    return this.cache.get(id);
  }

  /** Whether the cache has metadata for this id. */
  has(id: number): boolean {
    return this.cache.has(id);
  }

  /** Current cache size. */
  get size(): number {
    return this.cache.size;
  }

  /** Bulk-insert items (e.g. from the initial full `/api/medias` load for small datasets). */
  populate(items: MediaItem[]): void {
    for (const item of items) {
      this.cache.set(item.id, item);
    }
    this.versionSubject.next(this.versionSubject.value + 1);
  }

  /** Clear all cached metadata. */
  clear(): void {
    this.cache.clear();
    this.pendingIds.clear();
    this.versionSubject.next(0);
  }

  /**
   * Ensure metadata is loaded for the given IDs.  Already-cached and
   * already-pending IDs are skipped.  New IDs are coalesced into a batch
   * request that fires on a microtask (so rapid consecutive calls are merged).
   */
  ensureLoaded(ids: number[]): void {
    const needed: number[] = [];
    for (const id of ids) {
      if (!this.cache.has(id) && !this.pendingIds.has(id)) {
        needed.push(id);
        this.pendingIds.add(id);
      }
    }
    if (needed.length === 0) return;
    this.scheduleBatchFetch();
  }

  /**
   * Build a MediaItem[] for the given IDs using cached data.
   * IDs that are not yet cached are omitted.
   */
  toMediaItems(ids: number[]): MediaItem[] {
    const result: MediaItem[] = [];
    for (const id of ids) {
      const item = this.cache.get(id);
      if (item) result.push(item);
    }
    return result;
  }

  // ---------------------------------------------------------------------------
  // Internal
  // ---------------------------------------------------------------------------

  private scheduleBatchFetch(): void {
    if (this.batchTimer) return; // already scheduled
    this.batchTimer = setTimeout(() => {
      this.batchTimer = null;
      this.flushPending();
    }, 0);
  }

  private flushPending(): void {
    const ids = Array.from(this.pendingIds);
    this.pendingIds.clear();
    if (ids.length === 0) return;

    // Split into chunks to keep individual requests bounded.
    for (let i = 0; i < ids.length; i += BATCH_SIZE) {
      const chunk = ids.slice(i, i + BATCH_SIZE);
      this.mediasApi
        .getMediasBatch(chunk)
        .pipe(takeUntil(this.destroy$))
        .subscribe((items) => {
          for (const item of items) {
            this.cache.set(item.id, item);
          }
          this.versionSubject.next(this.versionSubject.value + 1);
        });
    }
  }
}
