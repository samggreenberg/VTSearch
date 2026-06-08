import { Injectable, inject } from '@angular/core';
import { Observable, of, Subject } from 'rxjs';
import { tap, shareReplay, catchError } from 'rxjs/operators';
import { ProjectionApiService } from './projection-api.service';
import type { BinShape, TilePayload } from '../models/projection.models';

interface CacheEntry {
  tile: TilePayload;
  lastAccess: number;
}

@Injectable({ providedIn: 'root' })
export class TileCacheService {
  private projectionApi = inject(ProjectionApiService);

  private cache = new Map<string, CacheEntry>();
  private inflight = new Map<string, Observable<TilePayload>>();
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
  // The bin shape (hex/square) tiles are currently fetched for. It is part of
  // the cache key, so switching shapes keeps both binnings cached side by side
  // (they share one projection id, so the id alone can't tell them apart).
  private binShape: BinShape = 'hex';
  // Whether tiles are fetched from the ephemeral subset projection (the
  // positives of a Find run) rather than the full-dataset projection. The two
  // have distinct projection ids, so the cache invalidates on switch.
  private subset = false;

  readonly tileLoaded$ = new Subject<TilePayload>();

  setProjectionId(id: string): void {
    if (id !== this.projectionId) {
      this.cache.clear();
      this.inflight.clear();
      this.projectionId = id;
      this.contentVersion = 0;
    }
  }

  setBinShape(shape: BinShape): void {
    this.binShape = shape;
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

  getTile(level: number, tx: number, ty: number): Observable<TilePayload> | null {
    const key = this.key(level, tx, ty);

    const cached = this.cache.get(key);
    if (cached) {
      cached.lastAccess = Date.now();
      return of(cached.tile);
    }

    const existing = this.inflight.get(key);
    if (existing) return existing;

    if (!this.projectionId) return null;

    const req$ = this.projectionApi.getTile(this.binShape, level, tx, ty, this.subset, this.cacheToken()).pipe(
      tap((tile) => {
        this.inflight.delete(key);
        this.put(key, tile);
        this.tileLoaded$.next(tile);
      }),
      catchError(() => {
        this.inflight.delete(key);
        const empty: TilePayload = { level, tx, ty, cells: [] };
        return of(empty);
      }),
      shareReplay(1),
    );
    this.inflight.set(key, req$);
    return req$;
  }

  getCached(level: number, tx: number, ty: number): TilePayload | null {
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

  private key(level: number, tx: number, ty: number): string {
    return `${this.binShape}:${this.contentVersion}:${level}:${tx}:${ty}`;
  }

  /** Cache-bust token for the tile URL: ``<projection_id>:<content_version>``. */
  private cacheToken(): string {
    return `${this.projectionId}:${this.contentVersion}`;
  }

  private put(key: string, tile: TilePayload): void {
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
