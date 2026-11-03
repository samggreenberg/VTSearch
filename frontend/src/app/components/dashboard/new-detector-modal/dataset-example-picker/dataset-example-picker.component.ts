import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  ElementRef,
  computed,
  effect,
  inject,
  input,
  linkedSignal,
  output,
  signal,
  untracked,
  viewChild,
} from '@angular/core';
import { rxResource, takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { DecimalPipe } from '@angular/common';
import { CdkVirtualScrollViewport, ScrollingModule } from '@angular/cdk/scrolling';
import { Subscription } from 'rxjs';

import { MediasApiService } from '../../../../services/medias-api.service';
import { ActiveContextService } from '../../../../services/active-context.service';
import { MediaTypeCapabilityService } from '../../../../services/media-type-capability.service';
import { SettingsStateService } from '../../../../services/settings-state.service';
import type { MediaPickerListing } from '../../../../generated/api-client/models/media-picker-listing';
import { ViewControlsComponent } from '../../../view-controls/view-controls.component';
import { apiErrorMessage } from '../../../../utils/api-error';
import { iconSizeToGoalWidth } from '../../../../utils/grid-icon-size';
import { coerceNonEmptyString } from '../../../../utils/settings-coerce';

/** How long the scroll must rest before the tiles on screen load (ms). */
export const THUMB_SETTLE_MS = 120;
/** How long typing must pause before the search goes to the server (ms). */
export const SEARCH_DEBOUNCE_MS = 200;
/** Gap between tiles, across and down (px): the stylesheet's `--space-xs`. */
const GAP_PX = 4;
/** Grid width assumed until it can be measured (and always, under jsdom). */
const FALLBACK_WIDTH_PX = 640;

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
 * Built for a dataset of thousands (#5465):
 *
 * - **Order and search come from the server** (`GET /api/medias/picker`): a
 *   Face dataset lists in similarity order, so one person's faces sit
 *   together, and the search keeps the items whose name holds every word typed
 *   — a face's name starts with its photo's, so a photo's name finds its faces.
 * - **The grid is virtual, with a fixed row height**, so the scrollbar is
 *   sized for the whole list from the start and a drag lands where it points.
 * - **Thumbnails load only once the scroll rests** ({@link THUMB_SETTLE_MS}):
 *   a jump down the list would otherwise queue a request for every row it
 *   passed, and the browser would work through them before the rows the user
 *   stopped at. Tiles that have loaded once stay loaded.
 * - **Thumbnail size** has the same smaller/bigger buttons as every other
 *   grid, remembered per media type as `grid_icon_size_picker`.
 *
 * Names come from `/api/medias/batch` for the tiles that load: the tile's
 * tooltip, and the tile itself for a media type with no thumbnail.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-dataset-example-picker',
  standalone: true,
  imports: [DecimalPipe, ScrollingModule, ViewControlsComponent],
  templateUrl: './dataset-example-picker.component.html',
  styleUrl: './dataset-example-picker.component.scss',
})
export class DatasetExamplePickerComponent {
  private readonly mediasApi = inject(MediasApiService);
  private readonly activeContext = inject(ActiveContextService);
  private readonly mediaTypeCaps = inject(MediaTypeCapabilityService);
  private readonly settingsState = inject(SettingsStateService);
  private readonly destroyRef = inject(DestroyRef);

  /** The loaded dataset to pick from. */
  readonly datasetId = input.required<string>();
  /** Its media type: thumbnails, waveforms, or named tiles. */
  readonly mediaType = input('');
  /** True while the host turns a pick into an example; picking waits. */
  readonly busy = input(false);

  /** The user picked this media id. */
  readonly picked = output<number>();

  // --- Listing + search -----------------------------------------------------

  /** What is typed in the search box. */
  readonly query = signal('');
  /** {@link query}, trimmed, once typing pauses: what the server is asked. */
  private readonly searchTerm = signal('');
  private searchTimer: ReturnType<typeof setTimeout> | null = null;

  private readonly listingResource = rxResource({
    params: () => {
      const datasetId = this.datasetId();
      return datasetId ? { datasetId, q: this.searchTerm() } : undefined;
    },
    stream: ({ params }) => this.mediasApi.getPickerListingIn(params.datasetId, params.q),
  });

  /** The latest listing, kept on screen while a new search loads, so the
   *  search box (and its focus) survives every keystroke. */
  readonly listing = linkedSignal<MediaPickerListing | undefined, MediaPickerListing | undefined>({
    source: () => (this.listingResource.hasValue() ? this.listingResource.value() : undefined),
    computation: (next, previous) => next ?? previous?.value,
  });
  /** The ids on offer, in the server's order. */
  readonly ids = computed(() => this.listing()?.ids ?? []);
  /** Whether a search narrows the listing. */
  readonly searchActive = computed(() => this.searchTerm() !== '');
  readonly error = computed(() => {
    const err = this.listingResource.error();
    return err ? apiErrorMessage(err, 'Could not list this dataset’s items.') : '';
  });

  // --- Geometry ---------------------------------------------------------------

  private readonly sizePref = this.settingsState.perMediaType<string>('grid_icon_size_picker', this.mediaType, {
    fallback: 'M',
    coerce: coerceNonEmptyString,
  });
  /** Side of one square tile (px), from the thumbnail-size buttons. */
  readonly cellWidth = computed(() => iconSizeToGoalWidth(this.sizePref.value()));
  /** Pixel stride of one virtual row. */
  readonly rowSize = computed(() => this.cellWidth() + GAP_PX);
  /** Width the rows lay out in: the scroll area's, less its scrollbar. */
  private readonly gridWidth = signal(FALLBACK_WIDTH_PX);
  readonly columns = computed(() =>
    Math.max(1, Math.floor((this.gridWidth() + GAP_PX) / (this.cellWidth() + GAP_PX))),
  );
  /** {@link ids} chunked into rows of {@link columns}: the virtual list's data. */
  readonly rows = computed(() => {
    const ids = this.ids();
    const cols = this.columns();
    const rows: number[][] = [];
    for (let i = 0; i < ids.length; i += cols) rows.push(ids.slice(i, i + cols));
    return rows;
  });

  private readonly viewport = viewChild(CdkVirtualScrollViewport);
  private readonly viewportEl = viewChild(CdkVirtualScrollViewport, { read: ElementRef });
  private rangeSub: Subscription | null = null;
  private rangeSubscribedViewport: CdkVirtualScrollViewport | null = null;
  /** Index (into {@link ids}) of the first item in view, kept so a change of
   *  thumbnail size re-chunks the rows around it instead of jumping. */
  private anchorItem = 0;
  private resizeObserver: ResizeObserver | null = null;
  private observedEl: HTMLElement | null = null;

  // --- Tiles ------------------------------------------------------------------

  /** Ids whose thumbnails may load: on screen when the scroll last rested. */
  private readonly settledIds = signal<ReadonlySet<number>>(new Set());
  private settleTimer: ReturnType<typeof setTimeout> | null = null;
  /** Names fetched so far, by id. */
  private readonly names = signal<ReadonlyMap<number, string>>(new Map());
  /** Ids whose names have been asked for. */
  private readonly namesAsked = new Set<number>();
  /** Thumbnails that failed to load; their tiles fall back to the name. */
  private readonly failedThumbs = signal<ReadonlySet<number>>(new Set());

  readonly usesThumbnails = computed(() => this.mediaTypeCaps.usesThumbnails(this.mediaType()));
  /** Audio thumbnails are waveform alpha masks tinted to the accent (#2369). */
  readonly isAudioWaveform = computed(() => this.mediaType() === 'audio');

  constructor() {
    // No settingsState.load() here: the size buttons' own init loads the
    // settings, and until they land the size reads as its M fallback.
    this.mediaTypeCaps.ensureLoaded();
    // The viewport lives behind the template's `@if`s, so a search with no
    // match destroys it and the next one makes a new one: re-wire on each.
    effect(() => {
      this.viewport();
      untracked(() => this.wireViewport());
    });
    // New rows (a search, a resize) may put unseen ids where the rendered
    // range already was, which the range stream does not report.
    effect(() => {
      this.rows();
      untracked(() => this.scheduleSettle());
    });
    // A new search starts at the top of its results.
    effect(() => {
      this.ids();
      untracked(() => {
        this.anchorItem = 0;
        this.viewport()?.scrollToIndex(0);
      });
    });
    // A new thumbnail size keeps the first item in view where it was.
    effect(() => {
      const cols = this.columns();
      this.rowSize();
      untracked(() => {
        const row = Math.floor(this.anchorItem / cols);
        if (row > 0) requestAnimationFrame(() => this.viewport()?.scrollToIndex(row));
      });
    });
    this.destroyRef.onDestroy(() => {
      this.rangeSub?.unsubscribe();
      this.resizeObserver?.disconnect();
      if (this.settleTimer !== null) clearTimeout(this.settleTimer);
      if (this.searchTimer !== null) clearTimeout(this.searchTimer);
    });
  }

  onQuery(value: string): void {
    this.query.set(value);
    if (this.searchTimer !== null) clearTimeout(this.searchTimer);
    this.searchTimer = setTimeout(() => {
      this.searchTimer = null;
      this.searchTerm.set(value.trim());
    }, SEARCH_DEBOUNCE_MS);
  }

  private wireViewport(): void {
    const vp = this.viewport() ?? null;
    if (vp === this.rangeSubscribedViewport) return;
    this.rangeSubscribedViewport = vp;
    this.rangeSub?.unsubscribe();
    this.rangeSub = null;
    if (vp) {
      this.rangeSub = vp.renderedRangeStream.subscribe(() => this.scheduleSettle());
      this.rangeSub.add(vp.scrolledIndexChange.subscribe((row) => (this.anchorItem = row * this.columns())));
    }
    this.observeWidth(this.viewportEl()?.nativeElement ?? null);
  }

  /** Track the scroll area's width, so the column count follows the dialog. */
  private observeWidth(el: HTMLElement | null): void {
    if (el === this.observedEl) return;
    this.observedEl = el;
    this.resizeObserver?.disconnect();
    if (!el || typeof ResizeObserver === 'undefined') return;
    this.resizeObserver = new ResizeObserver(() => {
      // clientWidth leaves out the scrollbar, which the rows can't use.
      if (el.clientWidth > 0) this.gridWidth.set(el.clientWidth);
    });
    this.resizeObserver.observe(el);
  }

  /** Load the tiles on screen once the scroll has rested {@link THUMB_SETTLE_MS}. */
  private scheduleSettle(): void {
    if (this.settleTimer !== null) clearTimeout(this.settleTimer);
    this.settleTimer = setTimeout(() => {
      this.settleTimer = null;
      this.settle();
    }, THUMB_SETTLE_MS);
  }

  private settle(): void {
    const vp = this.viewport();
    const rows = this.rows();
    if (!vp || rows.length === 0) return;
    const { start, end } = vp.getRenderedRange();
    const settled = this.settledIds();
    const fresh = rows
      .slice(start, end)
      .flat()
      .filter((id) => !settled.has(id));
    if (fresh.length === 0) return;
    this.settledIds.update((ids) => {
      const next = new Set(ids);
      for (const id of fresh) next.add(id);
      return next;
    });
    this.requestNames(fresh);
  }

  private requestNames(ids: number[]): void {
    const wanted = ids.filter((id) => !this.namesAsked.has(id));
    if (wanted.length === 0) return;
    for (const id of wanted) this.namesAsked.add(id);
    this.mediasApi
      .getMediasBatchIn(this.datasetId(), wanted)
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

  /** Whether the tile has rested on screen, so its thumbnail may load. */
  thumbReady(id: number): boolean {
    return this.settledIds().has(id);
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

  trackByRow(index: number): number {
    return index;
  }
}
