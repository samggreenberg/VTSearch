import { Injectable, OnDestroy } from '@angular/core';
import { BehaviorSubject, Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { MediaItem } from '../models/api.models';
import { MediasApiService } from './medias-api.service';
import { MediaMetadataCacheService } from './media-metadata-cache.service';

/**
 * Tracks the dataset-wide list of media stubs (``{id, type, embedder?}``)
 * and the current selection.
 *
 * Each stub is the minimum the UI needs to build the virtual scroller, the
 * stripe overview, and the media-type / embedder gates.  Full per-item
 * metadata (``filename``, ``md5``, ``custom_metadata``, …) is fetched on
 * demand for whatever is currently in the viewport via
 * {@link MediaMetadataCacheService}.
 */
@Injectable({ providedIn: 'root' })
export class MediaStateService implements OnDestroy {
  private readonly mediasSubject = new BehaviorSubject<MediaItem[]>([]);
  private readonly selectedIdSubject = new BehaviorSubject<number | null>(null);
  private readonly destroy$ = new Subject<void>();

  readonly medias$ = this.mediasSubject.asObservable();
  readonly selectedId$ = this.selectedIdSubject.asObservable();

  constructor(
    private mediasApi: MediasApiService,
    private metadataCache: MediaMetadataCacheService,
  ) {}

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  get medias(): MediaItem[] {
    return this.mediasSubject.value;
  }

  get selectedId(): number | null {
    return this.selectedIdSubject.value;
  }

  get selectedMedia(): MediaItem | null {
    const id = this.selectedIdSubject.value;
    if (id === null) return null;
    const cached = this.metadataCache.get(id);
    if (cached) return cached;
    return this.mediasSubject.value.find((m) => m.id === id) ?? null;
  }

  selectMedia(id: number): void {
    this.selectedIdSubject.next(id);
    this.metadataCache.ensureLoaded([id]);
  }

  loadMedias(): void {
    this.mediasApi
      .getMediaIds()
      .pipe(takeUntil(this.destroy$))
      .subscribe((stubs) => {
        this.mediasSubject.next(stubs);
      });
  }

  clear(): void {
    this.mediasSubject.next([]);
    this.selectedIdSubject.next(null);
    this.metadataCache.clear();
  }
}
