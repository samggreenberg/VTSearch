import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  computed,
  effect,
  inject,
  input,
  output,
  signal,
  untracked,
} from '@angular/core';
import { rxResource, takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { DecimalPipe } from '@angular/common';

import { MediasApiService } from '../../../../services/medias-api.service';
import { ActiveContextService } from '../../../../services/active-context.service';
import { MediaTypeCapabilityService } from '../../../../services/media-type-capability.service';
import { apiErrorMessage } from '../../../../utils/api-error';

/** Items added to the grid each time the user nears its bottom. */
export const PICKER_PAGE = 120;
/** Page in more once the scroll is this close to the bottom, in px. */
const SCROLL_SLACK_PX = 300;

/**
 * The items of one loaded dataset as a grid to pick an example from (#5205).
 *
 * New Detector's media tab offers it beside **Browse …**: on a Face dataset,
 * whose detectors can't start from text, it is the only way to start from a
 * face already in the app rather than one brought in from outside. A click on
 * an item emits its media id; the host turns that into an example.
 *
 * Everything is read out of {@link datasetId} by that id alone
 * (`DATASET_OVERRIDE`, `ActiveContextService.mediaUrlIn`), never with the
 * active pair, since the modal can open from the Dashboard with some other
 * pair active, or none.
 *
 * The grid grows a page at a time as the user scrolls, so a dataset of
 * thousands of items renders as many tiles as anyone looks at. Names come
 * from `/api/medias/batch` per page: the tile's tooltip, and the tile itself
 * for a media type with no thumbnail.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-dataset-example-picker',
  standalone: true,
  imports: [DecimalPipe],
  templateUrl: './dataset-example-picker.component.html',
  styleUrl: './dataset-example-picker.component.scss',
})
export class DatasetExamplePickerComponent {
  private readonly mediasApi = inject(MediasApiService);
  private readonly activeContext = inject(ActiveContextService);
  private readonly mediaTypeCaps = inject(MediaTypeCapabilityService);
  private readonly destroyRef = inject(DestroyRef);

  /** The loaded dataset to pick from. */
  readonly datasetId = input.required<string>();
  /** Its media type: thumbnails, waveforms, or named tiles. */
  readonly mediaType = input('');
  /** True while the host turns a pick into an example; picking waits. */
  readonly busy = input(false);

  /** The user picked this media id. */
  readonly picked = output<number>();

  private readonly idsResource = rxResource({
    params: () => this.datasetId() || undefined,
    stream: ({ params }) => this.mediasApi.getMediaIdsIn(params),
  });

  /** Every item id in the dataset, in dataset order. Empty until listed, and
   *  on a failed listing, whose value would throw if read. */
  readonly ids = computed(() => (this.idsResource.hasValue() ? this.idsResource.value() : []).map((m) => m.id));
  readonly loading = this.idsResource.isLoading;
  readonly error = computed(() => {
    const err = this.idsResource.error();
    return err ? apiErrorMessage(err, 'Could not list this dataset’s items.') : '';
  });

  private readonly shownCount = signal(PICKER_PAGE);
  /** The ids on screen: the first pages of {@link ids}. */
  readonly shown = computed(() => this.ids().slice(0, this.shownCount()));
  readonly hasMore = computed(() => this.shownCount() < this.ids().length);

  /** Names fetched so far, by id. */
  private readonly names = signal<ReadonlyMap<number, string>>(new Map());
  /** How many of {@link shown} have had their names asked for. */
  private namesRequested = 0;
  /** Thumbnails that failed to load; their tiles fall back to the name. */
  private readonly failedThumbs = signal<ReadonlySet<number>>(new Set());

  readonly usesThumbnails = computed(() => this.mediaTypeCaps.usesThumbnails(this.mediaType()));
  /** Audio thumbnails are waveform alpha masks tinted to the accent (#2369). */
  readonly isAudioWaveform = computed(() => this.mediaType() === 'audio');

  constructor() {
    this.mediaTypeCaps.ensureLoaded();
    // Ask for the names of each page as it comes on screen.
    effect(() => {
      const shown = this.shown();
      untracked(() => this.requestNames(shown));
    });
  }

  private requestNames(shown: number[]): void {
    if (shown.length <= this.namesRequested) return;
    const page = shown.slice(this.namesRequested);
    this.namesRequested = shown.length;
    this.mediasApi
      .getMediasBatchIn(this.datasetId(), page)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (items) => {
          const next = new Map(this.names());
          for (const item of items) next.set(item.id, item.filename);
          this.names.set(next);
        },
        // A tile without a name still shows its thumbnail and still picks.
        error: () => {},
      });
  }

  name(id: number): string {
    return this.names().get(id) || `Item #${id}`;
  }

  showsThumbnail(id: number): boolean {
    return this.usesThumbnails() && !this.failedThumbs().has(id);
  }

  thumbnailUrl(id: number): string {
    return this.activeContext.mediaUrlIn(this.datasetId(), `/api/medias/${id}/thumbnail`);
  }

  onThumbnailError(id: number): void {
    this.failedThumbs.update((failed) => new Set(failed).add(id));
  }

  pick(id: number): void {
    if (this.busy()) return;
    this.picked.emit(id);
  }

  /** Show the next page once the scroll nears the bottom of what is shown. */
  onScroll(event: Event): void {
    const el = event.target as HTMLElement;
    if (el.scrollTop + el.clientHeight >= el.scrollHeight - SCROLL_SLACK_PX) this.showMore();
  }

  showMore(): void {
    if (this.hasMore()) this.shownCount.update((n) => n + PICKER_PAGE);
  }
}
