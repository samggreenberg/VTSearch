import { TestBed } from '@angular/core/testing';
import { Observable, of } from 'rxjs';
import { MediaMetadataCacheService } from './media-metadata-cache.service';
import { MediasApiService } from './medias-api.service';
import { ActiveContextService } from './active-context.service';
import type { MediaBatchResponse } from '../generated/api-client/models/media-batch-response';

describe('MediaMetadataCacheService', () => {
  let service: MediaMetadataCacheService;
  let getMediasBatch: ReturnType<typeof vi.fn>;
  let datasetId: string;

  function media(id: number): MediaBatchResponse {
    return {
      id,
      filename: `f${id}`,
      md5: `m${id}`,
      media_type: 'image',
      custom_metadata: {},
    } as MediaBatchResponse;
  }

  function items(ids: number[]): MediaBatchResponse[] {
    return ids.map(media);
  }

  beforeEach(() => {
    datasetId = 'ds1';
    getMediasBatch = vi.fn((ids: number[]): Observable<MediaBatchResponse[]> => of(items(ids)));
    const activeContextStub = {
      get datasetId() {
        return datasetId;
      },
    };
    TestBed.configureTestingModule({
      providers: [
        MediaMetadataCacheService,
        { provide: MediasApiService, useValue: { getMediasBatch } },
        { provide: ActiveContextService, useValue: activeContextStub },
      ],
    });
    service = TestBed.inject(MediaMetadataCacheService);
  });

  it('starts empty', () => {
    expect(service.size).toBe(0);
    expect(service.get(1)).toBeUndefined();
    expect(service.has(1)).toBe(false);
  });

  it('ensureLoaded batches a request and caches the results', async () => {
    vi.useFakeTimers();
    try {
      service.ensureLoaded([1, 2]);
      // Nothing fetched until the coalescing microtask fires.
      expect(getMediasBatch).not.toHaveBeenCalled();

      await vi.advanceTimersByTimeAsync(0);
      expect(getMediasBatch).toHaveBeenCalledTimes(1);
      expect(getMediasBatch).toHaveBeenCalledWith([1, 2]);
      expect(service.has(1)).toBe(true);
      expect(service.get(2)).toEqual(media(2));
      expect(service.size).toBe(2);
    } finally {
      vi.useRealTimers();
    }
  });

  it('coalesces rapid consecutive ensureLoaded calls into a single batch', async () => {
    vi.useFakeTimers();
    try {
      service.ensureLoaded([1]);
      service.ensureLoaded([2, 3]);
      await vi.advanceTimersByTimeAsync(0);
      expect(getMediasBatch).toHaveBeenCalledTimes(1);
      expect(getMediasBatch).toHaveBeenCalledWith([1, 2, 3]);
    } finally {
      vi.useRealTimers();
    }
  });

  it('skips already-cached ids on the next ensureLoaded', async () => {
    vi.useFakeTimers();
    try {
      service.ensureLoaded([1, 2]);
      await vi.advanceTimersByTimeAsync(0);
      getMediasBatch.mockClear();

      service.ensureLoaded([2, 3]);
      await vi.advanceTimersByTimeAsync(0);
      // Only id 3 is new.
      expect(getMediasBatch).toHaveBeenCalledWith([3]);
    } finally {
      vi.useRealTimers();
    }
  });

  it('does nothing when every requested id is already cached', async () => {
    vi.useFakeTimers();
    try {
      service.ensureLoaded([1]);
      await vi.advanceTimersByTimeAsync(0);
      getMediasBatch.mockClear();

      service.ensureLoaded([1]);
      await vi.advanceTimersByTimeAsync(0);
      expect(getMediasBatch).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });

  it('bumps version$ on each batch arrival', async () => {
    vi.useFakeTimers();
    try {
      const versions: number[] = [];
      service.version$.subscribe((v) => versions.push(v));
      service.ensureLoaded([1]);
      await vi.advanceTimersByTimeAsync(0);
      expect(versions[versions.length - 1]).toBe(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it('toMediaItems returns cached entries and omits unknown ids', async () => {
    vi.useFakeTimers();
    try {
      service.ensureLoaded([1, 2]);
      await vi.advanceTimersByTimeAsync(0);
      expect(service.toMediaItems([1, 99, 2])).toEqual([media(1), media(2)]);
    } finally {
      vi.useRealTimers();
    }
  });

  it('keys cache entries per dataset so a switch never returns the wrong metadata', async () => {
    vi.useFakeTimers();
    try {
      service.ensureLoaded([1]);
      await vi.advanceTimersByTimeAsync(0);
      expect(service.get(1)).toEqual(media(1));

      datasetId = 'ds2';
      expect(service.get(1)).toBeUndefined();
      expect(service.has(1)).toBe(false);

      datasetId = 'ds1';
      expect(service.get(1)).toEqual(media(1));
    } finally {
      vi.useRealTimers();
    }
  });

  it('clear empties the cache and resets the version', async () => {
    vi.useFakeTimers();
    try {
      const versions: number[] = [];
      service.version$.subscribe((v) => versions.push(v));
      service.ensureLoaded([1]);
      await vi.advanceTimersByTimeAsync(0);

      service.clear();
      expect(service.size).toBe(0);
      expect(service.get(1)).toBeUndefined();
      expect(versions[versions.length - 1]).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });

  it('re-queues ids after a failed batch and retries on the next ensureLoaded', async () => {
    vi.useFakeTimers();
    try {
      getMediasBatch.mockReturnValueOnce(
        new Observable<MediaBatchResponse[]>((o) => o.error(new Error('500'))),
      );
      service.ensureLoaded([1]);
      await vi.advanceTimersByTimeAsync(0);
      expect(service.has(1)).toBe(false);

      // A later request re-drives the fetch; this time it succeeds.
      service.ensureLoaded([1]);
      await vi.advanceTimersByTimeAsync(0);
      expect(service.has(1)).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });
});
