import { ChangeDetectionStrategy, Component, DestroyRef, computed, effect, inject, input, output, signal } from '@angular/core';
import { rxResource, takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { Observable, catchError, map, of, switchMap } from 'rxjs';

import { Media } from '../../../models/api.models';
import { ModalComponent } from '../../modal/modal.component';
import { ImageViewerComponent } from '../../center-panel/image-viewer/image-viewer.component';
import { DashboardLoadingTasksService } from '../../../services/dashboard-loading-tasks.service';
import { DatasetStateService } from '../../../services/dataset-state.service';
import { DatasetsRegistryApiService } from '../../../services/datasets-registry-api.service';
import { MediasApiService } from '../../../services/medias-api.service';
import { SourceLookup, SourcePhotoRequest, SourcePhotoService } from '../../../services/source-photo.service';

/** What the overlay shows: the lookup, and for a found photo its metadata. */
type SourcePhotoView =
  | Exclude<SourceLookup, { kind: 'found' }>
  | (Extract<SourceLookup, { kind: 'found' }> & { photo: Media });

/**
 * Show in photo (#4750): a face crop's source photo, with the face outlined.
 *
 * The photo is an item of the Image dataset the face's import produced, read
 * with that dataset's id alone while the Face pair stays active (see
 * `SourcePhotoService`), so the ranking behind the overlay is exactly where
 * the user left it when they go Back. When that dataset is not loaded the
 * lookup answers 409 and the overlay offers to load it, then asks again.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-source-photo-modal',
  standalone: true,
  imports: [ModalComponent, ImageViewerComponent],
  templateUrl: './source-photo-modal.component.html',
  styleUrl: './source-photo-modal.component.scss',
})
export class SourcePhotoModalComponent {
  private readonly sourcePhoto = inject(SourcePhotoService);
  private readonly mediasApi = inject(MediasApiService);
  private readonly datasetState = inject(DatasetStateService);
  private readonly datasetsRegistryApi = inject(DatasetsRegistryApiService);
  private readonly loadingTasks = inject(DashboardLoadingTasksService);
  private readonly destroyRef = inject(DestroyRef);

  readonly request = input.required<SourcePhotoRequest>();
  readonly closed = output<void>();

  private readonly resource = rxResource({
    params: () => this.request().mediaId,
    stream: ({ params }) => this.sourcePhoto.lookup(params).pipe(switchMap((found) => this.withPhoto(found))),
  });

  /** The lookup's answer, or null while it is still out. */
  readonly view = computed<SourcePhotoView | null>(() => this.resource.value() ?? null);

  /** The task id of the sibling load this overlay started, while it runs. */
  private readonly loadTaskId = signal<string | null>(null);
  readonly siblingLoading = computed(() => this.loadTaskId() !== null);
  /** The sibling is loading, or the lookup is being asked again after it. */
  readonly busy = computed(() => this.siblingLoading() || this.resource.isLoading());
  /** Why the sibling load failed, once it has. */
  readonly loadError = signal('');

  /** The name of the dataset the photo lives in (or would, once loaded). */
  readonly siblingName = computed(() => {
    const v = this.view();
    const id = v?.kind === 'found' || v?.kind === 'not-loaded' ? v.datasetId : '';
    return (id && this.datasetState.datasetById().get(id)?.name) || 'the photo dataset';
  });

  readonly title = computed(() => {
    const v = this.view();
    return v?.kind === 'found' ? this.siblingName() : 'Show in photo';
  });

  constructor() {
    // A sibling load that settles with an error never calls back (the
    // loading-task service fires completions only on success), so watch its
    // row for the error instead; the failure is also toasted app-wide.
    effect(() => {
      const taskId = this.loadTaskId();
      if (!taskId) return;
      const failed = this.loadingTasks.loadingTasks.find((t) => t.task_id === taskId && t.status === 'idle' && t.error);
      if (!failed) return;
      this.loadTaskId.set(null);
      this.loadError.set(failed.error ?? 'Loading failed.');
    });
  }

  close(): void {
    this.closed.emit();
  }

  /** Load the sibling the lookup named, then look again. */
  loadSibling(datasetId: string): void {
    if (this.siblingLoading()) return;
    this.loadError.set('');
    this.loadTaskId.set('');
    this.datasetsRegistryApi
      .loadRegistered(datasetId)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (resp) => {
          const taskId = resp?.task_id ?? '';
          if (!taskId) {
            this.afterSiblingLoad();
            return;
          }
          this.loadTaskId.set(taskId);
          this.loadingTasks.startProgressPolling(taskId, () => this.afterSiblingLoad(taskId));
        },
        error: () => this.loadTaskId.set(null),
      });
  }

  /** The sibling is loaded: ask again, unless the overlay moved on. */
  private afterSiblingLoad(taskId = ''): void {
    if (this.loadTaskId() !== taskId) return;
    this.loadTaskId.set(null);
    this.resource.reload();
  }

  private withPhoto(found: SourceLookup): Observable<SourcePhotoView> {
    if (found.kind !== 'found') return of(found);
    // The stub is enough to show the image; the metadata adds its name.
    const stub: Media = { id: found.mediaId, media_type: 'image' };
    return this.mediasApi.getMediasBatchIn(found.datasetId, [found.mediaId]).pipe(
      map((rows) => ({ ...found, photo: rows[0] ?? stub })),
      catchError(() => of({ ...found, photo: stub })),
    );
  }
}
