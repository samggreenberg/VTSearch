import {
  ChangeDetectionStrategy,
  Component,
  computed,
  effect,
  inject,
  Input,
  input,
  OnChanges,
  OnInit,
  output,
  signal,
  SimpleChanges,
  ViewChild,
} from '@angular/core';
import { rxResource } from '@angular/core/rxjs-interop';
import { CommonModule } from '@angular/common';
import { SortBarComponent } from './sort-bar/sort-bar.component';
import { SelectModeComponent } from './select-mode/select-mode.component';
import { InclusionSliderComponent } from './inclusion-slider/inclusion-slider.component';
import { ProgressIndicatorsComponent } from './progress-indicators/progress-indicators.component';
import { MediaListComponent } from './media-list/media-list.component';
import { StripeOverviewComponent } from './stripe-overview/stripe-overview.component';
import { AutopilotPanelComponent } from './autopilot-panel/autopilot-panel.component';
import { ViewControlsComponent } from '../view-controls/view-controls.component';
import { IconComponent } from '../icon/icon.component';
import { Media, MediaTypeInfo } from '../../models/api.models';
import type { LabelingStatusResponse } from '../../generated/api-client/models/labeling-status-response';
import { DatasetsListingsApiService } from '../../services/datasets-listings-api.service';
import { EmbedderCapabilityService } from '../../services/embedder-capability.service';
import { MediaTypeCapabilityService } from '../../services/media-type-capability.service';
import { SortMode, SelectMode, SortedItem } from '../../services/sort-state.service';

export type { SortMode, SelectMode, SortedItem };

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-left-panel',
  standalone: true,
  imports: [
    CommonModule,
    SortBarComponent,
    SelectModeComponent,
    InclusionSliderComponent,
    ProgressIndicatorsComponent,
    MediaListComponent,
    StripeOverviewComponent,
    AutopilotPanelComponent,
    ViewControlsComponent,
    IconComponent,
  ],
  templateUrl: './left-panel.component.html',
  styleUrl: './left-panel.component.scss',
})
export class LeftPanelComponent implements OnInit, OnChanges {
  @Input() medias: Media[] = [];
  readonly sortOrder = input<SortedItem[] | null>(null);
  readonly threshold = input<number | null>(null);
  readonly selectedId = input<number | null>(null);
  readonly goodVotes = input<Set<number>>(new Set());
  readonly badVotes = input<Set<number>>(new Set());
  /**
   * True when the active detector (or active votes, when no detector is
   * loaded) has at least one good and one bad label.  Used to gate "Sort by
   * Learned"; distinct from ``goodVotes`` / ``badVotes`` because those Sets
   * only contain media IDs in the *currently loaded* dataset.
   */
  readonly learnedSortAvailable = input(false);
  /** Active detector's saved labelset counts (across all datasets). */
  readonly labelsetGoodCount = input(0);
  readonly labelsetBadCount = input(0);
  readonly sortMode = input<SortMode>('text');
  readonly selectMode = input<SelectMode>('top');
  readonly inclusion = input<number>(0);
  @Input() sortBusy = false;
  readonly sortStatus = input('');
  readonly sortProgress = input(0);
  readonly sortProgressTotal = input(0);
  readonly sortOverall = input<number | null>(null);
  readonly sortEtaSeconds = input<number | null>(null);
  readonly labelingStatus = input<LabelingStatusResponse | null>(null);
  readonly gridGoalWidth = input<number>(80);
  readonly focusMode = input<'click' | 'hover'>('click');
  readonly loadSortLabel = input('');
  readonly textQuery = input('');
  readonly autopilotCollapsed = input(false);
  @Input() autopilotEnabled = true;
  /**
   * True when Autopilot cannot run on the active (dataset, detector) pair: the
   * dataset's embedder can't search by text, the detector has no media-example
   * seed, and there aren't yet enough labels for Learn sort. In that state
   * Autopilot has no way to seed its first sort, so the tab is disabled and the
   * panel falls back to Manual. It re-enables once Learn sort becomes available
   * (the parent flips this back to ``false`` once both label classes exist).
   */
  @Input() autopilotDisabled = false;
  /** 'label' = full labeling UI (default), 'find' = simplified media-only view */
  readonly panelMode = input<'label' | 'find'>('label');
  /** Disable all interaction (used during Find scoring). */
  readonly disabled = input(false);
  /** Display name of the current dataset. */
  @Input() datasetName = '';

  readonly sortModeChange = output<SortMode>();
  readonly selectModeChange = output<SelectMode>();
  readonly inclusionChange = output<number>();
  readonly textSort = output<string>();
  readonly learnedSort = output<void>();
  readonly loadSort = output<void>();
  readonly modelSelected = output<string>();
  readonly exampleSortStarted = output<unknown>();
  readonly mediaSelect = output<number>();
  readonly mediaVote = output<{
    id: number;
    vote: 'good' | 'bad';
}>();
  readonly mediaContextRequest = output<{
    id: number;
    x: number;
    y: number;
}>();
  readonly indicatorClick = output<string>();
  /** User clicked the Cancel button on the running sort progress bar. */
  readonly sortCancel = output<void>();
  /** Find mode: browse the unverified positives as their own UMAP projection. */
  readonly browse = output<void>();
  /** Find mode: promote the unverified positives into their own dataset. */
  readonly toDataset = output<void>();
  /** Find mode: export the unverified positives (above-threshold work queue). */
  readonly unverifiedExport = output<void>();
  readonly autopilotStart = output<void>();
  readonly autopilotStop = output<void>();
  readonly autopilotRefocus = output<void>();
  readonly autopilotToggleCollapse = output<void>();
  readonly autopilotEnabledChange = output<boolean>();

  @ViewChild(MediaListComponent) mediaListComponent!: MediaListComponent;

  activeTab: 'manual' | 'autopilot' = 'autopilot';
  // Written from the constructor `effect()`s (when media-type / embedder
  // metadata arrives) as well as the sync `ngOnChanges` path, so signals — an
  // effect writing a plain template-bound field does not repaint under zoneless.
  readonly mediaTypeName = signal('Media');
  readonly textSortAvailable = signal(true);

  private readonly datasetsListingsApi = inject(DatasetsListingsApiService);
  private readonly embedderCaps = inject(EmbedderCapabilityService);
  private readonly mediaTypeCaps = inject(MediaTypeCapabilityService);

  // Media-type metadata rides `rxResource`: loads once on creation (no request
  // signal = eager), wrapping the existing generated-client read so the
  // interceptor chain still applies. Embedder capability metadata comes from
  // the shared `EmbedderCapabilityService` cache instead. The derived labels
  // live in plain fields recomputed by the effects below + `ngOnChanges`.
  private readonly mediaTypesResource = rxResource({
    stream: () => this.datasetsListingsApi.getMediaTypes(),
  });
  private readonly mediaTypeInfos = computed<MediaTypeInfo[]>(
    () => this.mediaTypesResource.value()?.media_types ?? [],
  );

  constructor() {
    // Re-derive the header label / text-sort gate when the metadata arrives
    // (it can land after the first batch of medias). Effects auto-dispose with
    // the component, so no manual unsubscribe is needed.
    effect(() => {
      this.mediaTypeInfos();
      this.updateMediaTypeName();
    });
    effect(() => {
      this.embedderCaps.infos();
      this.updateTextSortAvailable();
    });
  }

  ngOnInit(): void {
    this.embedderCaps.ensureLoaded();
    this.mediaTypeCaps.ensureLoaded();
    if (this.panelMode() === 'find') {
      // Find mode doesn't use tabs; keep manual as a no-op default
      this.activeTab = 'manual';
    } else {
      const startAutopilot = this.autopilotEnabled && !this.autopilotDisabled;
      this.activeTab = startAutopilot ? 'autopilot' : 'manual';
      if (startAutopilot) {
        this.autopilotStart.emit();
      }
    }
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['medias']) {
      this.updateMediaTypeName();
      this.updateTextSortAvailable();
    }
    if (changes['autopilotDisabled'] && !changes['autopilotDisabled'].firstChange) {
      // Autopilot just became impossible (e.g. medias loaded and revealed a
      // no-text embedder, after we optimistically started on entry). Fall back
      // to Manual so the user can label by hand; the doomed text sort is also
      // skipped upstream. When it flips back to available we leave the user
      // where they are and just re-enable the tab.
      if (this.autopilotDisabled && this.activeTab === 'autopilot') {
        this.activeTab = 'manual';
        this.autopilotStop.emit();
      }
    }
  }

  /**
   * Re-derive the media-type label shown in the grid header from the current
   * grid contents.  Always recomputes (no memo on the type id) so the header
   * never lags behind the grid: it resets to ``'Media'`` when the grid empties,
   * and upgrades from the capitalized fallback to the proper display name once
   * the media-type metadata finishes loading (which can arrive *after* the
   * first batch of medias).
   */
  private updateMediaTypeName(): void {
    const typeId = this.medias.length > 0 ? this.medias[0].media_type : '';
    if (!typeId) {
      this.mediaTypeName.set('Media');
      return;
    }
    const info = this.mediaTypeInfos().find((mt) => mt.type_id === typeId);
    this.mediaTypeName.set(info?.name ?? typeId.charAt(0).toUpperCase() + typeId.slice(1));
  }

  /**
   * Resolve whether the active dataset's embedder can embed text queries.
   * If the embedder is unknown (e.g. embedders haven't loaded yet, or the
   * media doesn't carry an embedder field), default to ``true`` so we never
   * hide a working feature.
   */
  private updateTextSortAvailable(): void {
    const first = this.medias.length > 0 ? this.medias[0] : null;
    const names = first?.embedders ?? (first?.embedder ? [first.embedder] : []);
    this.textSortAvailable.set(this.embedderCaps.supportsTextAny(names));
  }

  /**
   * Count of unverified positives: items above the cutoff in the work-queue
   * ranking. In Find mode ``sortOrder`` is the *unverified* ranking (verified
   * items are filtered out upstream), so the above-threshold slice is exactly
   * the unverified positives. Gates the Find work-queue action buttons —
   * Browse / To Dataset / Export all operate on exactly this set.
   */
  get unverifiedGoodCount(): number {
    const order = this.sortOrder();
    const threshold = this.threshold();
    if (!order || threshold == null) return 0;
    const cutoff = threshold;
    let n = 0;
    for (const item of order) {
      if (item.score >= cutoff) n++;
    }
    return n;
  }

  onStripeClick(index: number): void {
    if (this.mediaListComponent) {
      this.mediaListComponent.scrollToIndex(index);
    }
  }

  setTab(tab: 'manual' | 'autopilot'): void {
    // Autopilot can't be entered while it has no way to seed a first sort.
    if (tab === 'autopilot' && this.autopilotDisabled) return;
    if (tab === this.activeTab) {
      if (tab === 'autopilot') {
        this.autopilotRefocus.emit();
      }
      return;
    }
    const previous = this.activeTab;
    this.activeTab = tab;
    if (previous === 'autopilot') {
      this.autopilotStop.emit();
    }
    if (tab === 'autopilot') {
      this.autopilotStart.emit();
    }
    this.autopilotEnabledChange.emit(tab === 'autopilot');
  }
}
