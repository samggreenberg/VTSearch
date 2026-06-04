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
  private readonly MAX_ENTRIES = 512;
  private projectionId = '';
  // The bin shape (hex/square) tiles are currently fetched for. It is part of
  // the cache key, so switching shapes keeps both binnings cached side by side
  // (they share one projection id, so the id alone can't tell them apart).
  private binShape: BinShape = 'hex';

  readonly tileLoaded$ = new Subject<TilePayload>();

  setProjectionId(id: string): void {
    if (id !== this.projectionId) {
      this.cache.clear();
      this.inflight.clear();
      this.projectionId = id;
    }
  }

  setBinShape(shape: BinShape): void {
    this.binShape = shape;
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

    const req$ = this.projectionApi.getTile(this.binShape, level, tx, ty).pipe(
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
  }

  private key(level: number, tx: number, ty: number): string {
    return `${this.binShape}:${level}:${tx}:${ty}`;
  }

  private put(key: string, tile: TilePayload): void {
    if (this.cache.size >= this.MAX_ENTRIES) {
      this.evict();
    }
    this.cache.set(key, { tile, lastAccess: Date.now() });
  }

  private evict(): void {
    const target = Math.floor(this.MAX_ENTRIES * 0.75);
    const entries = [...this.cache.entries()].sort((a, b) => a[1].lastAccess - b[1].lastAccess);
    const toRemove = entries.length - target;
    for (let i = 0; i < toRemove; i++) {
      this.cache.delete(entries[i][0]);
    }
  }
}
