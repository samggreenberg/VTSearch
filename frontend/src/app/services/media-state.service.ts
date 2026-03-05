import { Injectable, OnDestroy } from '@angular/core';
import { BehaviorSubject, Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { MediaItem } from '../models/api.models';
import { MediasApiService } from './medias-api.service';

@Injectable({ providedIn: 'root' })
export class MediaStateService implements OnDestroy {
  private readonly mediasSubject = new BehaviorSubject<MediaItem[]>([]);
  private readonly selectedIdSubject = new BehaviorSubject<number | null>(null);
  private readonly destroy$ = new Subject<void>();

  readonly medias$ = this.mediasSubject.asObservable();
  readonly selectedId$ = this.selectedIdSubject.asObservable();

  constructor(private mediasApi: MediasApiService) {}

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
    return this.mediasSubject.value.find((m) => m.id === id) ?? null;
  }

  selectMedia(id: number): void {
    this.selectedIdSubject.next(id);
  }

  loadMedias(): void {
    this.mediasApi
      .getMedias()
      .pipe(takeUntil(this.destroy$))
      .subscribe((medias) => {
        this.mediasSubject.next(medias);
      });
  }

  clear(): void {
    this.mediasSubject.next([]);
    this.selectedIdSubject.next(null);
  }
}
