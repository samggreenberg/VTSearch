import { TestBed } from '@angular/core/testing';
import { Observable, Subject, of } from 'rxjs';
import { TILE_BATCH_SIZE, TileCacheService } from './tile-cache.service';
import { ProjectionApiService } from './projection-api.service';
import type { TileResponse } from '../models/projection.models';

/** One `getTiles` call: the batch the service sent. */
type TilesCall = [level: number, tiles: readonly (readonly [number, number])[], subset: boolean, cacheToken: string];

describe('TileCacheService', () => {
  let service: TileCacheService;
  let getTiles: ReturnType<typeof vi.fn>;

  function payload(level: number, tx: number, ty: number): TileResponse {
    return { level, tx, ty, cells: [] };
  }

  /** Let the end-of-task flush send what was queued (and an `of()` answer land). */
  const flush = () => new Promise<void>((resolve) => queueMicrotask(resolve));

  /** Every batch sent so far. */
  const calls = () => getTiles.mock.calls as TilesCall[];

  beforeEach(() => {
    getTiles = vi.fn(
      (level: number, tiles: readonly (readonly [number, number])[]): Observable<TileResponse[]> =>
        of(tiles.map(([tx, ty]) => payload(level, tx, ty))),
    );
    TestBed.configureTestingModule({
      providers: [TileCacheService, { provide: ProjectionApiService, useValue: { getTiles } }],
    });
    service = TestBed.inject(TileCacheService);
  });

  it('getTile returns null before a projection id is set', () => {
    expect(service.getTile(0, 0, 0)).toBeNull();
  });

  it('fetches a tile once, then serves it from cache without re-hitting the API', async () => {
    service.setProjectionId('p1');

    let first: TileResponse | undefined;
    service.getTile(2, 1, 1)?.subscribe((t) => (first = t));
    await flush();
    expect(first).toEqual(payload(2, 1, 1));
    expect(getTiles).toHaveBeenCalledTimes(1);

    let second: TileResponse | undefined;
    service.getTile(2, 1, 1)?.subscribe((t) => (second = t));
    await flush();
    expect(second).toEqual(payload(2, 1, 1));
    // Still one call: the second read hit the cache.
    expect(getTiles).toHaveBeenCalledTimes(1);
  });

  it('passes the subset flag and cache token through to the API', async () => {
    service.setProjectionId('proj-9');
    service.setContentVersion(4);
    service.setSubset(true);
    service.getTile(1, 0, 0)?.subscribe();
    await flush();
    expect(getTiles).toHaveBeenCalledWith(1, [[0, 0]], true, 'proj-9:4');
  });

  describe('batching (#5637)', () => {
    it('sends every tile asked for in one task as one request per level', async () => {
      service.setProjectionId('p1');
      service.getTile(3, 1, 0)?.subscribe();
      service.getTile(3, 0, 0)?.subscribe();
      service.getTile(2, 5, 5)?.subscribe();
      service.getTile(3, 0, 1)?.subscribe();
      // Nothing goes out until the task that asked ends.
      expect(getTiles).not.toHaveBeenCalled();
      await flush();

      expect(getTiles).toHaveBeenCalledTimes(2);
      const byLevel = new Map(calls().map(([level, tiles]) => [level, tiles]));
      expect(byLevel.get(2)).toEqual([[5, 5]]);
      expect(byLevel.get(3)).toHaveLength(3);
    });

    it('lists each batch row by row, so the same view asks for the same URL', async () => {
      service.setProjectionId('p1');
      for (const [tx, ty] of [[1, 1], [0, 1], [-1, 0], [2, 0]]) service.getTile(1, tx, ty)?.subscribe();
      await flush();
      expect(calls()[0][1]).toEqual([[-1, 0], [2, 0], [0, 1], [1, 1]]);
    });

    it('splits a big ask into batches of the batch size', async () => {
      service.setProjectionId('p1');
      const n = TILE_BATCH_SIZE * 2 + 5;
      for (let tx = 0; tx < n; tx++) service.getTile(4, tx, 0)?.subscribe();
      await flush();
      expect(calls().map(([, tiles]) => tiles.length)).toEqual([TILE_BATCH_SIZE, TILE_BATCH_SIZE, 5]);
      // Every tile landed, whichever batch carried it.
      for (let tx = 0; tx < n; tx++) expect(service.getCached(4, tx, 0)).not.toBeNull();
    });

    it('starts a new batch in a later task', async () => {
      service.setProjectionId('p1');
      service.getTile(1, 0, 0)?.subscribe();
      await flush();
      service.getTile(1, 1, 0)?.subscribe();
      await flush();
      expect(calls().map(([, tiles]) => tiles)).toEqual([[[0, 0]], [[1, 0]]]);
    });

    it('answers a tile the response left out with an empty, uncached tile', async () => {
      getTiles.mockReturnValueOnce(of([payload(1, 0, 0)]));
      service.setProjectionId('p1');
      let missing: TileResponse | undefined;
      service.getTile(1, 0, 0)?.subscribe();
      service.getTile(1, 1, 0)?.subscribe((t) => (missing = t));
      await flush();
      expect(service.getCached(1, 0, 0)).not.toBeNull();
      expect(missing).toEqual(payload(1, 1, 0));
      expect(service.getCached(1, 1, 0)).toBeNull();
    });

    it('does not send a tile whose view was switched away from before the batch went out', async () => {
      service.setProjectionId('p1');
      let got: TileResponse | undefined;
      service.getTile(2, 0, 0)?.subscribe((t) => (got = t));
      service.setProjectionId('p2');
      await flush();
      expect(getTiles).not.toHaveBeenCalled();
      // Its caller still hears back, with an empty tile.
      expect(got).toEqual(payload(2, 0, 0));
    });
  });

  it('dedupes concurrent in-flight requests for the same tile', async () => {
    const pending = new Subject<TileResponse[]>();
    getTiles.mockReturnValueOnce(pending.asObservable());
    service.setProjectionId('p1');

    const a = service.getTile(5, 2, 2);
    const b = service.getTile(5, 2, 2);
    expect(a).toBe(b);
    await flush();
    // Asked again while the batch is out: still the one request.
    expect(service.getTile(5, 2, 2)).toBe(a);
    await flush();
    expect(getTiles).toHaveBeenCalledTimes(1);
  });

  it('recovers from a failed batch by emitting an empty tile for each', async () => {
    getTiles.mockReturnValueOnce(new Observable<TileResponse[]>((o) => o.error(new Error('404'))));
    service.setProjectionId('p1');

    let a: TileResponse | undefined;
    let b: TileResponse | undefined;
    service.getTile(3, 7, 8)?.subscribe((t) => (a = t));
    service.getTile(3, 8, 8)?.subscribe((t) => (b = t));
    await flush();
    expect(a).toEqual({ level: 3, tx: 7, ty: 8, cells: [] });
    expect(b).toEqual({ level: 3, tx: 8, ty: 8, cells: [] });
    // Not cached, so a later frame asks again.
    expect(service.getCached(3, 7, 8)).toBeNull();
    service.getTile(3, 7, 8)?.subscribe();
    await flush();
    expect(getTiles).toHaveBeenCalledTimes(2);
  });

  it('changing the projection id clears the cache', async () => {
    service.setProjectionId('p1');
    service.getTile(2, 1, 1)?.subscribe();
    await flush();
    expect(service.getCached(2, 1, 1)).not.toBeNull();

    service.setProjectionId('p2');
    expect(service.getCached(2, 1, 1)).toBeNull();
  });

  it('re-setting the same projection id keeps the cache warm', async () => {
    service.setProjectionId('p1');
    service.getTile(2, 1, 1)?.subscribe();
    await flush();
    expect(service.getCached(2, 1, 1)).not.toBeNull();

    // A no-op id set must not clear the cache (only a *change* invalidates).
    service.setProjectionId('p1');
    expect(service.getCached(2, 1, 1)).not.toBeNull();
  });

  it('setContentVersion re-keys entries so a stale version misses', async () => {
    service.setProjectionId('p1');
    service.getTile(2, 1, 1)?.subscribe();
    await flush();
    expect(service.getCached(2, 1, 1)).not.toBeNull();

    service.setContentVersion(9);
    // Same layout, different membership version → prior tile is unreachable.
    expect(service.getCached(2, 1, 1)).toBeNull();
  });

  it('drops a tile that arrives after the projection switched', async () => {
    const pending = new Subject<TileResponse[]>();
    getTiles.mockReturnValueOnce(pending.asObservable());
    service.setProjectionId('p1');
    // Fire-and-forget subscriber, exactly like the canvas/minimap draw paths.
    service.getTile(2, 1, 3)?.subscribe();
    await flush();

    service.setProjectionId('p2');
    const loaded: TileResponse[] = [];
    service.tileLoaded$.subscribe((t) => loaded.push(t));
    // p1's response finally lands, after the switch cleared the cache.
    pending.next([
      {
        level: 2,
        tx: 1,
        ty: 3,
        cells: [{ q: 9, r: 9, cx: 1, cy: 1, count: 1, rep_id: 42 }],
      },
    ]);

    // p2 must not see p1's cells, and must not be told to redraw for them.
    expect(service.getCached(2, 1, 3)).toBeNull();
    expect(loaded).toEqual([]);
    // The tile is still fetchable for p2 (the stale put did not block it).
    let fresh: TileResponse | undefined;
    service.getTile(2, 1, 3)?.subscribe((t) => (fresh = t));
    await flush();
    expect(fresh).toEqual(payload(2, 1, 3));
  });

  it('drops a tile that arrives after clear()', async () => {
    const pending = new Subject<TileResponse[]>();
    getTiles.mockReturnValueOnce(pending.asObservable());
    service.setProjectionId('p1');
    service.getTile(1, 0, 0)?.subscribe();
    await flush();

    service.clear();
    pending.next([payload(1, 0, 0)]);

    service.setProjectionId('p1');
    expect(service.getCached(1, 0, 0)).toBeNull();
  });

  it('an older batch landing after clear() does not unhook the newer request for the same tile', async () => {
    const older = new Subject<TileResponse[]>();
    const newer = new Subject<TileResponse[]>();
    getTiles.mockReturnValueOnce(older.asObservable()).mockReturnValueOnce(newer.asObservable());
    service.setProjectionId('p2');
    service.getTile(1, 0, 0)?.subscribe();
    await flush();

    // A switch away and back, then the same tile asked for again.
    service.setProjectionId('p1');
    service.setProjectionId('p2');
    let got: TileResponse | undefined;
    service.getTile(1, 0, 0)?.subscribe((t) => (got = t));
    await flush();
    expect(getTiles).toHaveBeenCalledTimes(2);

    older.error(new Error('aborted'));
    // The newer request is still the one in flight: asking now sends nothing.
    service.getTile(1, 0, 0)?.subscribe();
    await flush();
    expect(getTiles).toHaveBeenCalledTimes(2);
    newer.next([payload(1, 0, 0)]);
    expect(got).toEqual(payload(1, 0, 0));
    expect(service.getCached(1, 0, 0)).toEqual(payload(1, 0, 0));
  });

  it('keys tiles by subset so the full and subset projections cannot collide', async () => {
    service.setProjectionId('p1');
    service.getTile(2, 1, 1)?.subscribe();
    await flush();
    expect(service.getCached(2, 1, 1)).not.toBeNull();

    service.setSubset(true);
    expect(service.getCached(2, 1, 1)).toBeNull();
  });

  it('getCached returns null for an uncached tile and the tile once cached', async () => {
    service.setProjectionId('p1');
    expect(service.getCached(4, 0, 0)).toBeNull();
    service.getTile(4, 0, 0)?.subscribe();
    await flush();
    expect(service.getCached(4, 0, 0)).toEqual(payload(4, 0, 0));
  });

  it('prefetch warms the cache and is a no-op when already cached', async () => {
    service.setProjectionId('p1');
    service.prefetch(1, 0, 0);
    await flush();
    expect(getTiles).toHaveBeenCalledTimes(1);
    service.prefetch(1, 0, 0);
    await flush();
    expect(getTiles).toHaveBeenCalledTimes(1);
  });

  it('clear resets the cache and requires a new projection id', async () => {
    service.setProjectionId('p1');
    service.getTile(2, 1, 1)?.subscribe();
    await flush();
    service.clear();
    expect(service.getCached(2, 1, 1)).toBeNull();
    expect(service.getTile(2, 1, 1)).toBeNull();
  });

  it('emits loaded tiles on tileLoaded$', async () => {
    const loaded: TileResponse[] = [];
    service.tileLoaded$.subscribe((t) => loaded.push(t));
    service.setProjectionId('p1');
    service.getTile(2, 3, 4)?.subscribe();
    await flush();
    expect(loaded).toEqual([payload(2, 3, 4)]);
  });

  it('evicts coarse-but-unpinned tiles under memory pressure while keeping pinned levels', async () => {
    service.setProjectionId('p1');
    // Pinned coarse tiles (levels 0..2) must survive eviction.
    for (let tx = 0; tx < 4; tx++) service.getTile(0, tx, 0)?.subscribe();
    await flush();
    // Overfill with evictable level-4 tiles to trip the LRU sweep, a task at a
    // time so earlier ones are already cached (older) when later ones land.
    for (let tx = 0; tx < 2200; tx++) {
      service.getTile(4, tx, 0)?.subscribe();
      if (tx % TILE_BATCH_SIZE === TILE_BATCH_SIZE - 1) await flush();
    }
    await flush();

    // Every pinned tile is still resident.
    for (let tx = 0; tx < 4; tx++) {
      expect(service.getCached(0, tx, 0)).not.toBeNull();
    }
    // The earliest evictable tiles were dropped to stay under budget.
    expect(service.getCached(4, 0, 0)).toBeNull();
  });
});
