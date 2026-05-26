import { Injectable, OnDestroy } from '@angular/core';
import { BehaviorSubject, Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import type { MediaBatchResponse } from '../generated/api-client/models/media-batch-response';
import { ActiveContextService } from './active-context.service';
import { MediasApiService } from './medias-api.service';

/**
 * Default batch size for metadata requests.  Larger batches reduce HTTP round
 * trips; smaller batches reduce per-request latency.
 */
const BATCH_SIZE = 200;

/**
 * Caches full media metadata (`filename`, `md5`, `custom_metadata`, …)
 * loaded lazily in batches via `POST /api/medias/batch`.
 *
 * The dataset-wide listing comes from `GET /api/medias/ids` and only
 * carries `id`, `type`, and (optionally) `embedder`.  This cache fills in
 * the rest on demand - typically for the items currently in a
 * virtual-scrolling viewport.
 *
 * Usage:
 *   1. Call `ensureLoaded(ids)` with the IDs the viewport needs.
 *   2. Subscribe to `version$` to re-render when new items arrive, or call
 *      `get(id)` to read a single cached item.
 *   3. `toMediaItems(ids)` returns MediaBatchResponse[] for already-cached
 *      IDs (skips unknown ones so the template can render immediately).
 *
 * Media IDs are per-dataset, so every cache entry is keyed by
 * `${datasetId}:${mediaId}` and pending IDs are bucketed by the dataset
 * they were queued under - otherwise a switch from dataset A to dataset B
 * returns A's filename / md5 / metadata for B's id.
 */
@Injectable({ providedIn: 'root' })
export class MediaMetadataCacheService implements OnDestroy {
  private readonly cache = new Map<string, MediaBatchResponse>();
  private readonly pendingByDataset = new Map<string, Set<number>>();
  private batchTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly destroy$ = new Subject<void>();

  /** Emits the current cache version (increments on every batch arrival). */
  private readonly versionSubject = new BehaviorSubject<number>(0);
  readonly version$ = this.versionSubject.asObservable();

  constructor(
    private mediasApi: MediasApiService,
    private activeContext: ActiveContextService,
  ) {}

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
    if (this.batchTimer) clearTimeout(this.batchTimer);
  }

  /** Return a cached metadata entry or undefined if not yet fetched. */
  get(id: number): MediaBatchResponse | undefined {
    return this.cache.get(this.activeKey(id));
  }

  /** Whether the cache has metadata for this id. */
  has(id: number): boolean {
    return this.cache.has(this.activeKey(id));
  }

  /** Current cache size. */
  get size(): number {
    return this.cache.size;
  }

  /** Clear all cached metadata. */
  clear(): void {
    this.cache.clear();
    this.pendingByDataset.clear();
    this.versionSubject.next(0);
  }

  /**
   * Ensure metadata is loaded for the given IDs.  Already-cached and
   * already-pending IDs are skipped.  New IDs are coalesced into a batch
   * request that fires on a microtask (so rapid consecutive calls are merged).
   */
  ensureLoaded(ids: number[]): void {
    const datasetId = this.activeContext.datasetId;
    const pending = this.getOrCreatePending(datasetId);
    for (const id of ids) {
      if (!this.cache.has(this.keyFor(datasetId, id))) {
        pending.add(id);
      }
    }
    if (pending.size === 0) return;
    this.scheduleBatchFetch();
  }

  /**
   * Build a MediaBatchResponse[] for the given IDs using cached data.
   * IDs that are not yet cached are omitted.
   */
  toMediaItems(ids: number[]): MediaBatchResponse[] {
    const result: MediaBatchResponse[] = [];
    const datasetId = this.activeContext.datasetId;
    for (const id of ids) {
      const item = this.cache.get(this.keyFor(datasetId, id));
      if (item) result.push(item);
    }
    return result;
  }

  // ---------------------------------------------------------------------------
  // Internal
  // ---------------------------------------------------------------------------

  private keyFor(datasetId: string, id: number): string {
    return `${datasetId}:${id}`;
  }

  private activeKey(id: number): string {
    return this.keyFor(this.activeContext.datasetId, id);
  }

  private getOrCreatePending(datasetId: string): Set<number> {
    let pending = this.pendingByDataset.get(datasetId);
    if (!pending) {
      pending = new Set<number>();
      this.pendingByDataset.set(datasetId, pending);
    }
    return pending;
  }

  private scheduleBatchFetch(): void {
    if (this.batchTimer) return; // already scheduled
    this.batchTimer = setTimeout(() => {
      this.batchTimer = null;
      this.flushPending();
    }, 0);
  }

  private flushPending(): void {
    // The batch request's X-Dataset-Id header is set from
    // `ActiveContextService` at dispatch time, so we can only drain
    // entries queued under the currently-active dataset. Entries for
    // inactive datasets stay queued; the next `ensureLoaded()` after a
    // switch re-schedules a flush for that dataset's bucket.
    const datasetId = this.activeContext.datasetId;
    const pending = this.pendingByDataset.get(datasetId);
    if (!pending || pending.size === 0) return;
    const ids = Array.from(pending);
    pending.clear();

    // Split into chunks to keep individual requests bounded.
    for (let i = 0; i < ids.length; i += BATCH_SIZE) {
      const chunk = ids.slice(i, i + BATCH_SIZE);
      this.mediasApi
        .getMediasBatch(chunk)
        .pipe(takeUntil(this.destroy$))
        .subscribe({
          next: (items) => {
            for (const item of items) {
              this.cache.set(this.keyFor(datasetId, item.id), item);
            }
            this.versionSubject.next(this.versionSubject.value + 1);
          },
          error: () => {
            // Re-queue failed IDs under the dataset that issued the
            // request so they can be retried on the next request.
            const bucket = this.getOrCreatePending(datasetId);
            for (const id of chunk) {
              if (!this.cache.has(this.keyFor(datasetId, id))) {
                bucket.add(id);
              }
            }
          },
        });
    }
  }
}
