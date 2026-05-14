import { Injectable, OnDestroy } from '@angular/core';
import { BehaviorSubject, Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { MediaItem } from '../models/api.models';
import { MediasApiService } from './medias-api.service';
import { MediaMetadataCacheService } from './media-metadata-cache.service';

/**
 * Active-dataset media state.
 *
 * Holds the ID list for the loaded dataset; full per-item metadata is
 * fetched on demand through `MediaMetadataCacheService` (which posts to
 * `/api/medias/batch`).  The legacy `GET /api/medias` endpoint that
 * returned every media's metadata in one blob is no longer used by the
 * frontend — it does not scale to 100k-item datasets.
 */
@Injectable({ providedIn: 'root' })
export class MediaStateService implements OnDestroy {
  private readonly mediaIdsSubject = new BehaviorSubject<number[]>([]);
  private readonly mediaTypeSubject = new BehaviorSubject<string>('');
  private readonly embedderSubject = new BehaviorSubject<string>('');
  private readonly selectedIdSubject = new BehaviorSubject<number | null>(null);
  private readonly destroy$ = new Subject<void>();

  readonly mediaIds$ = this.mediaIdsSubject.asObservable();
  readonly mediaType$ = this.mediaTypeSubject.asObservable();
  readonly embedder$ = this.embedderSubject.asObservable();
  readonly selectedId$ = this.selectedIdSubject.asObservable();

  constructor(
    private mediasApi: MediasApiService,
    private metadataCache: MediaMetadataCacheService,
  ) {}

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  get mediaIds(): number[] {
    return this.mediaIdsSubject.value;
  }

  get mediaType(): string {
    return this.mediaTypeSubject.value;
  }

  get embedder(): string {
    return this.embedderSubject.value;
  }

  get selectedId(): number | null {
    return this.selectedIdSubject.value;
  }

  get selectedMedia(): MediaItem | null {
    const id = this.selectedIdSubject.value;
    if (id === null) return null;
    return this.metadataCache.get(id) ?? null;
  }

  selectMedia(id: number): void {
    this.selectedIdSubject.next(id);
    this.metadataCache.ensureLoaded([id]);
  }

  loadMedias(): void {
    this.mediasApi
      .getMediaIds()
      .pipe(takeUntil(this.destroy$))
      .subscribe((resp) => {
        this.mediaTypeSubject.next(resp.type ?? '');
        this.embedderSubject.next(resp.embedder ?? '');
        this.mediaIdsSubject.next(resp.ids);
      });
  }

  clear(): void {
    this.mediaIdsSubject.next([]);
    this.mediaTypeSubject.next('');
    this.embedderSubject.next('');
    this.selectedIdSubject.next(null);
    this.metadataCache.clear();
  }
}
