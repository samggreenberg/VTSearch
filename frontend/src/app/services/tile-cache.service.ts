import { Injectable, inject } from '@angular/core';
import { Observable, of, ReplaySubject, Subject } from 'rxjs';
import { ProjectionApiService } from './projection-api.service';
import type { TileResponse } from '../models/projection.models';

interface CacheEntry {
  tile: TileResponse;
  lastAccess: number;
}

/** A tile asked for and not yet sent, with what its batch needs to send it. */
interface QueuedTile {
  level: number;
  tx: number;
  ty: number;
  /** Cache key, under the view the tile was asked for in. */
  key: string;
  /** {@link TileCacheService.viewToken} when asked for: the batch's view. */
  token: string;
  subset: boolean;
  cacheToken: string;
  /** Answers every caller waiting on this tile, once its batch lands. */
  done: ReplaySubject<TileResponse>;
  /** What {@link TileCacheService.getTile} handed out, held in `inflight`. */
  tile$: Observable<TileResponse>;
}

/**
 * Most tiles one request carries. A view asks for a few dozen tiles at once,
 * and the minimap's overview for over a hundred; the server takes up to 128.
 */
export const TILE_BATCH_SIZE = 64;

@Injectable({ providedIn: 'root' })
export class TileCacheService {
  private projectionApi = inject(ProjectionApiService);

  private cache = new Map<string, CacheEntry>();
  private inflight = new Map<string, Observable<TileResponse>>();
  /** Tiles asked for in this task, sent together when it ends. */
  private queue: QueuedTile[] = [];
  private flushQueued = false;
  // Deeper datasets carve more pyramid levels, so a small flat LRU thrashes:
  // panning back over seen ground misses and the hex grid blanks. Hold more.
  private readonly MAX_ENTRIES = 2048;
  // Levels 0..PINNED_LEVELS-1 are the coarse top of the pyramid: a handful of
  // tiles that cover the whole projection and are the cheapest thing to keep
  // warm. Exempt them from eviction so a zoom/pan back to the overview is never
  // a cache miss (and so they're available as a fallback layer later).
  private readonly PINNED_LEVELS = 3;
  private projectionId = '';
  // Membership version of the current (subset) projection. Bumped server-side
  // when items are removed from a subset browse in place; the layout identity
  // (projectionId) is kept stable so the canvas doesn't re-frame, so this is
  // what distinguishes "same layout, different contents" for the tile cache.
  private contentVersion = 0;
  // Whether tiles are fetched from the ephemeral subset projection (the
  // positives of a Find run) rather than the full-dataset projection. The two
  // have distinct projection ids, so the cache invalidates on switch.
  private subset = false;

  readonly tileLoaded$ = new Subject<TileResponse>();

  setProjectionId(id: string): void {
    if (id !== this.projectionId) {
      this.cache.clear();
      this.inflight.clear();
      this.projectionId = id;
      this.contentVersion = 0;
    }
  }

  /**
   * Update the membership version. Entries are keyed by it, so a change makes
   * the prior version's tiles unreachable (refetched on demand) without
   * clearing the whole cache — and rides along on the tile URL so the HTTP
   * cache refreshes too.
   */
  setContentVersion(version: number): void {
    this.contentVersion = version;
  }

  setSubset(subset: boolean): void {
    this.subset = subset;
  }

  /**
   * A tile, from the cache when it's there, else fetched.
   *
   * Fetches are batched (#5637). Every tile asked for in one task — a canvas
   * frame asks for all of its visible tiles in one draw — goes out together,
   * one request per level per {@link TILE_BATCH_SIZE} tiles, instead of one
   * request each. Over HTTP/1.1 a browser runs only about six requests at a
   * time per host, so one request per tile made a remote server's round trip
   * time the cost of every five or six tiles. Each batch lists its tiles
   * sorted, so the same view asks for the same URLs on every visit and the
   * browser cache keeps answering them.
   *
   * The returned observable emits the tile once and completes; on a failed
   * fetch it emits an empty tile, which is not cached, so a later frame asks
   * again.
   */
  getTile(level: number, tx: number, ty: number): Observable<TileResponse> | null {
    const key = this.key(level, tx, ty);

    const cached = this.cache.get(key);
    if (cached) {
      cached.lastAccess = Date.now();
      return of(cached.tile);
    }

    const existing = this.inflight.get(key);
    if (existing) return existing;

    if (!this.projectionId) return null;

    const done = new ReplaySubject<TileResponse>(1);
    const tile$ = done.asObservable();
    this.inflight.set(key, tile$);
    // Identity of the view this request belongs to, captured now. Subscribers
    // are fire-and-forget (the canvas and minimap never unsubscribe), so a
    // response can land long after a projection switch cleared the cache;
    // `land` drops it instead of caching another projection's cells.
    this.queue.push({
      level,
      tx,
      ty,
      key,
      token: this.viewToken(),
      subset: this.subset,
      cacheToken: this.cacheToken(),
      done,
      tile$,
    });
    if (!this.flushQueued) {
      this.flushQueued = true;
      queueMicrotask(() => this.flush());
    }
    return tile$;
  }

  getCached(level: number, tx: number, ty: number): TileResponse | null {
    const entry = this.cache.get(this.key(level, tx, ty));
    if (entry) {
      entry.lastAccess = Date.now();
      return entry.tile;
    }
    return null;
  }

  prefetch(level: number, tx: number, ty: number): void {
    const key = this.key(level, tx, ty);
    if (this.cache.has(key) || this.inflight.has(key)) return;
    this.getTile(level, tx, ty)?.subscribe();
  }

  clear(): void {
    this.cache.clear();
    this.inflight.clear();
    this.projectionId = '';
    this.contentVersion = 0;
  }

  /**
   * Send this task's queued tiles: one request per (view, level), in
   * {@link TILE_BATCH_SIZE} chunks of row-major sorted tiles. A tile whose view
   * was switched away from before it went out is not sent at all.
   */
  private flush(): void {
    this.flushQueued = false;
    const queued = this.queue;
    this.queue = [];
    const groups = new Map<string, QueuedTile[]>();
    for (const q of queued) {
      if (q.token !== this.viewToken()) {
        this.land(q, null);
        continue;
      }
      const group = `${q.token}|${q.level}`;
      const list = groups.get(group);
      if (list) list.push(q);
      else groups.set(group, [q]);
    }
    for (const group of groups.values()) {
      group.sort((a, b) => a.ty - b.ty || a.tx - b.tx);
      for (let i = 0; i < group.length; i += TILE_BATCH_SIZE) {
        this.send(group.slice(i, i + TILE_BATCH_SIZE));
      }
    }
  }

  private send(batch: QueuedTile[]): void {
    const { level, subset, cacheToken } = batch[0];
    const coords = batch.map((q) => [q.tx, q.ty] as const);
    this.projectionApi.getTiles(level, coords, subset, cacheToken).subscribe({
      next: (tiles) => {
        const byCoord = new Map(tiles.map((t) => [`${t.tx}:${t.ty}`, t]));
        for (const q of batch) this.land(q, byCoord.get(`${q.tx}:${q.ty}`) ?? null);
      },
      error: () => {
        for (const q of batch) this.land(q, null);
      },
    });
  }

  /** Settle one queued tile: cache it (if its view is still current) and
   *  answer its callers; ``null`` answers an uncached empty tile. */
  private land(q: QueuedTile, tile: TileResponse | null): void {
    // Only our own entry: a clear() and a re-ask may have put a newer one there.
    if (this.inflight.get(q.key) === q.tile$) this.inflight.delete(q.key);
    if (tile && this.viewToken() === q.token) {
      this.put(q.key, tile);
      this.tileLoaded$.next(tile);
    }
    q.done.next(tile ?? { level: q.level, tx: q.tx, ty: q.ty, cells: [] });
    q.done.complete();
  }

  private key(level: number, tx: number, ty: number): string {
    return `${this.viewToken()}:${level}:${tx}:${ty}`;
  }

  /**
   * Full identity of what a tile's cells depict: layout (``projectionId``),
   * membership (``contentVersion``) and which projection the tiles come from
   * (``subset``). Keying on all three means a tile fetched under one view can
   * never be read back under another, even if a stale ``put`` slips through —
   * ``setProjectionId`` resets ``contentVersion`` to 0, so the version alone
   * collides across projections.
   */
  private viewToken(): string {
    return `${this.projectionId}:${this.contentVersion}:${this.subset ? 's' : 'f'}`;
  }

  /** Cache-bust token for the tile URL: ``<projection_id>:<content_version>``. */
  private cacheToken(): string {
    return `${this.projectionId}:${this.contentVersion}`;
  }

  private put(key: string, tile: TileResponse): void {
    if (this.cache.size >= this.MAX_ENTRIES) {
      this.evict();
    }
    this.cache.set(key, { tile, lastAccess: Date.now() });
  }

  private evict(): void {
    const target = Math.floor(this.MAX_ENTRIES * 0.75);
    // Only coarse-but-not-pinned tiles are eviction candidates; pinned coarse
    // levels stay resident. They're few, so they can't crowd out the budget.
    const candidates = [...this.cache.entries()]
      .filter(([, e]) => e.tile.level >= this.PINNED_LEVELS)
      .sort((a, b) => a[1].lastAccess - b[1].lastAccess);
    const toRemove = this.cache.size - target;
    for (let i = 0; i < toRemove && i < candidates.length; i++) {
      this.cache.delete(candidates[i][0]);
    }
  }
}
