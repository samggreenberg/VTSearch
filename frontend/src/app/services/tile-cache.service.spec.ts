import { TestBed } from '@angular/core/testing';
import { Observable, Subject, of } from 'rxjs';
import { TileCacheService } from './tile-cache.service';
import { ProjectionApiService } from './projection-api.service';
import type { TilePayload } from '../models/projection.models';

describe('TileCacheService', () => {
  let service: TileCacheService;
  let getTile: ReturnType<typeof vi.fn>;

  function payload(level: number, tx: number, ty: number): TilePayload {
    return { level, tx, ty, cells: [] };
  }

  beforeEach(() => {
    getTile = vi.fn((level: number, tx: number, ty: number): Observable<TilePayload> =>
      of(payload(level, tx, ty)),
    );
    TestBed.configureTestingModule({
      providers: [TileCacheService, { provide: ProjectionApiService, useValue: { getTile } }],
    });
    service = TestBed.inject(TileCacheService);
  });

  it('getTile returns null before a projection id is set', () => {
    expect(service.getTile(0, 0, 0)).toBeNull();
  });

  it('fetches a tile once, then serves it from cache without re-hitting the API', () => {
    service.setProjectionId('p1');

    let first: TilePayload | undefined;
    service.getTile(2, 1, 1)?.subscribe((t) => (first = t));
    expect(first).toEqual(payload(2, 1, 1));
    expect(getTile).toHaveBeenCalledTimes(1);

    let second: TilePayload | undefined;
    service.getTile(2, 1, 1)?.subscribe((t) => (second = t));
    expect(second).toEqual(payload(2, 1, 1));
    // Still one call: the second read hit the cache.
    expect(getTile).toHaveBeenCalledTimes(1);
  });

  it('passes the subset flag and cache token through to the API', () => {
    service.setProjectionId('proj-9');
    service.setContentVersion(4);
    service.setSubset(true);
    service.getTile(1, 0, 0)?.subscribe();
    expect(getTile).toHaveBeenCalledWith(1, 0, 0, true, 'proj-9:4');
  });

  it('dedupes concurrent in-flight requests for the same tile', () => {
    const pending = new Subject<TilePayload>();
    getTile.mockReturnValueOnce(pending.asObservable());
    service.setProjectionId('p1');

    const a = service.getTile(5, 2, 2);
    const b = service.getTile(5, 2, 2);
    expect(a).toBe(b);
    expect(getTile).toHaveBeenCalledTimes(1);
  });

  it('recovers from a tile fetch error by emitting an empty tile', () => {
    getTile.mockReturnValueOnce(new Observable<TilePayload>((o) => o.error(new Error('404'))));
    service.setProjectionId('p1');

    let got: TilePayload | undefined;
    service.getTile(3, 7, 8)?.subscribe((t) => (got = t));
    expect(got).toEqual({ level: 3, tx: 7, ty: 8, cells: [] });
  });

  it('changing the projection id clears the cache', () => {
    service.setProjectionId('p1');
    service.getTile(2, 1, 1)?.subscribe();
    expect(service.getCached(2, 1, 1)).not.toBeNull();

    service.setProjectionId('p2');
    expect(service.getCached(2, 1, 1)).toBeNull();
  });

  it('setContentVersion re-keys entries so a stale version misses', () => {
    service.setProjectionId('p1');
    service.getTile(2, 1, 1)?.subscribe();
    expect(service.getCached(2, 1, 1)).not.toBeNull();

    service.setContentVersion(9);
    // Same layout, different membership version → prior tile is unreachable.
    expect(service.getCached(2, 1, 1)).toBeNull();
  });

  it('getCached returns null for an uncached tile and the tile once cached', () => {
    service.setProjectionId('p1');
    expect(service.getCached(4, 0, 0)).toBeNull();
    service.getTile(4, 0, 0)?.subscribe();
    expect(service.getCached(4, 0, 0)).toEqual(payload(4, 0, 0));
  });

  it('prefetch warms the cache and is a no-op when already cached', () => {
    service.setProjectionId('p1');
    service.prefetch(1, 0, 0);
    expect(getTile).toHaveBeenCalledTimes(1);
    service.prefetch(1, 0, 0);
    expect(getTile).toHaveBeenCalledTimes(1);
  });

  it('clear resets the cache and requires a new projection id', () => {
    service.setProjectionId('p1');
    service.getTile(2, 1, 1)?.subscribe();
    service.clear();
    expect(service.getCached(2, 1, 1)).toBeNull();
    expect(service.getTile(2, 1, 1)).toBeNull();
  });

  it('emits loaded tiles on tileLoaded$', () => {
    const loaded: TilePayload[] = [];
    service.tileLoaded$.subscribe((t) => loaded.push(t));
    service.setProjectionId('p1');
    service.getTile(2, 3, 4)?.subscribe();
    expect(loaded).toEqual([payload(2, 3, 4)]);
  });

  it('evicts coarse-but-unpinned tiles under memory pressure while keeping pinned levels', () => {
    service.setProjectionId('p1');
    // Pinned coarse tiles (levels 0..2) must survive eviction.
    for (let tx = 0; tx < 4; tx++) service.getTile(0, tx, 0)?.subscribe();
    // Overfill with evictable level-4 tiles to trip the LRU sweep.
    for (let tx = 0; tx < 2200; tx++) service.getTile(4, tx, 0)?.subscribe();

    // Every pinned tile is still resident.
    for (let tx = 0; tx < 4; tx++) {
      expect(service.getCached(0, tx, 0)).not.toBeNull();
    }
    // The earliest evictable tiles were dropped to stay under budget.
    expect(service.getCached(4, 0, 0)).toBeNull();
  });
});
