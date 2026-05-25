import { Component, Input, Output, EventEmitter, OnInit, OnChanges, SimpleChanges, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SortBarComponent } from './sort-bar/sort-bar.component';
import { SelectModeComponent } from './select-mode/select-mode.component';
import { InclusionSliderComponent } from './inclusion-slider/inclusion-slider.component';
import { ProgressIndicatorsComponent } from './progress-indicators/progress-indicators.component';
import { MediaListComponent } from './media-list/media-list.component';
import { StripeOverviewComponent } from './stripe-overview/stripe-overview.component';
import { AutopilotPanelComponent } from './autopilot-panel/autopilot-panel.component';
import { ViewControlsComponent } from '../view-controls/view-controls.component';
import { Media, MediaTypeInfo, EmbedderInfo } from '../../models/api.models';
import type { LabelingStatusResponse } from '../../generated/api-client/models/labeling-status-response';
import { DatasetsListingsApiService } from '../../services/datasets-listings-api.service';
import { SortMode, SelectMode, SortedItem } from '../../services/sort-state.service';

export type { SortMode, SelectMode, SortedItem };

@Component({
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
  ],
  templateUrl: './left-panel.component.html',
  styleUrl: './left-panel.component.scss',
})
export class LeftPanelComponent implements OnInit, OnChanges {
  @Input() medias: Media[] = [];
  @Input() sortOrder: SortedItem[] | null = null;
  @Input() threshold: number | null = null;
  @Input() selectedId: number | null = null;
  @Input() goodVotes: Set<number> = new Set();
  @Input() badVotes: Set<number> = new Set();
  /**
   * True when the active detector (or active votes, when no detector is
   * loaded) has at least one good and one bad label.  Used to gate "Sort by
   * Learned" — distinct from ``goodVotes`` / ``badVotes`` because those Sets
   * only contain media IDs in the *currently loaded* dataset.
   */
  @Input() learnedSortAvailable = false;
  /** Active detector's saved labelset counts (across all datasets). */
  @Input() labelsetGoodCount = 0;
  @Input() labelsetBadCount = 0;
  @Input() sortMode: SortMode = 'text';
  @Input() selectMode: SelectMode = 'top';
  @Input() inclusion: number = 0;
  @Input() sortBusy = false;
  @Input() sortStatus = '';
  @Input() sortProgress = 0;
  @Input() sortProgressTotal = 0;
  @Input() labelingStatus: LabelingStatusResponse | null = null;
  @Input() viewMode: 'grid' | 'list' = 'list';
  @Input() gridGoalWidth: number = 80;
  @Input() focusMode: 'click' | 'hover' = 'click';
  @Input() loadSortLabel = '';
  @Input() textQuery = '';
  @Input() autopilotCollapsed = false;
  @Input() autopilotEnabled = true;
  /** 'label' = full labeling UI (default), 'find' = simplified media-only view */
  @Input() panelMode: 'label' | 'find' = 'label';
  /** Disable all interaction (used during Find scoring). */
  @Input() disabled = false;
  /** Display name of the current dataset. */
  @Input() datasetName = '';

  @Output() sortModeChange = new EventEmitter<SortMode>();
  @Output() selectModeChange = new EventEmitter<SelectMode>();
  @Output() inclusionChange = new EventEmitter<number>();
  @Output() textSort = new EventEmitter<string>();
  @Output() learnedSort = new EventEmitter<void>();
  @Output() loadSort = new EventEmitter<void>();
  @Output() modelSelected = new EventEmitter<string>();
  @Output() exampleSortStarted = new EventEmitter<unknown>();
  @Output() mediaSelect = new EventEmitter<number>();
  @Output() mediaVote = new EventEmitter<{ id: number; vote: 'good' | 'bad' }>();
  @Output() mediaContextRequest = new EventEmitter<{ id: number; x: number; y: number }>();
  @Output() indicatorClick = new EventEmitter<string>();
  /** User clicked the Cancel button on the running sort progress bar. */
  @Output() sortCancel = new EventEmitter<void>();
  @Output() autopilotStart = new EventEmitter<void>();
  @Output() autopilotStop = new EventEmitter<void>();
  @Output() autopilotRefocus = new EventEmitter<void>();
  @Output() autopilotToggleCollapse = new EventEmitter<void>();
  @Output() autopilotEnabledChange = new EventEmitter<boolean>();

  @ViewChild(MediaListComponent) mediaListComponent!: MediaListComponent;

  activeTab: 'manual' | 'autopilot' = 'autopilot';
  mediaTypeName = 'Media';
  textSortAvailable = true;
  private mediaTypeInfos: MediaTypeInfo[] = [];
  private currentTypeId = '';
  private embedderInfos: EmbedderInfo[] = [];

  constructor(private datasetsListingsApi: DatasetsListingsApiService) {}

  ngOnInit(): void {
    this.datasetsListingsApi.getMediaTypes().subscribe({
      next: (resp) => {
        this.mediaTypeInfos = resp.media_types;
        this.updateMediaTypeName();
      },
    });
    this.datasetsListingsApi.getEmbedders().subscribe({
      next: (embedders) => {
        this.embedderInfos = embedders;
        this.updateTextSortAvailable();
      },
    });
    if (this.panelMode === 'find') {
      // Find mode doesn't use tabs — keep manual as a no-op default
      this.activeTab = 'manual';
    } else {
      this.activeTab = this.autopilotEnabled ? 'autopilot' : 'manual';
      if (this.autopilotEnabled) {
        this.autopilotStart.emit();
      }
    }
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['medias']) {
      this.updateMediaTypeName();
      this.updateTextSortAvailable();
    }
  }

  private updateMediaTypeName(): void {
    const typeId = this.medias.length > 0 ? this.medias[0].media_type : '';
    if (typeId && typeId !== this.currentTypeId) {
      this.currentTypeId = typeId;
      const info = this.mediaTypeInfos.find((mt) => mt.type_id === typeId);
      this.mediaTypeName = info?.name ?? typeId.charAt(0).toUpperCase() + typeId.slice(1);
    }
  }

  /**
   * Resolve whether the active dataset's embedder can embed text queries.
   * If the embedder is unknown (e.g. embedders haven't loaded yet, or the
   * media doesn't carry an embedder field), default to ``true`` so we never
   * hide a working feature.
   */
  private updateTextSortAvailable(): void {
    const embedderName = this.medias.length > 0 ? this.medias[0].embedder : '';
    if (!embedderName || this.embedderInfos.length === 0) {
      this.textSortAvailable = true;
      return;
    }
    const info = this.embedderInfos.find((e) => e.name === embedderName);
    this.textSortAvailable = info ? info.supports_text !== false : true;
  }

  onStripeClick(index: number): void {
    if (this.mediaListComponent) {
      this.mediaListComponent.scrollToIndex(index);
    }
  }

  setTab(tab: 'manual' | 'autopilot'): void {
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
