import { Injectable, OnDestroy } from '@angular/core';
import { BehaviorSubject, Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { MediaItem } from '../models/api.models';
import { MediasApiService } from './medias-api.service';
import { MediaMetadataCacheService } from './media-metadata-cache.service';

/**
 * Threshold: datasets with more items than this use lazy metadata loading
 * instead of fetching everything in a single /api/medias call.
 */
const LAZY_THRESHOLD = 500;

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
    // Try cache first (works for both large and small datasets).
    const cached = this.metadataCache.get(id);
    if (cached) return cached;
    return this.mediasSubject.value.find((m) => m.id === id) ?? null;
  }

  selectMedia(id: number): void {
    this.selectedIdSubject.next(id);
    // Ensure the selected item's metadata is loaded for the center panel.
    this.metadataCache.ensureLoaded([id]);
  }

  loadMedias(): void {
    this.mediasApi
      .getMedias()
      .pipe(takeUntil(this.destroy$))
      .subscribe((medias) => {
        // Populate the metadata cache so batch lookups work for any ID.
        this.metadataCache.populate(medias);
        this.mediasSubject.next(medias);
      });
  }

  clear(): void {
    this.mediasSubject.next([]);
    this.selectedIdSubject.next(null);
    this.metadataCache.clear();
  }
}
