import { Component, Input, Output, EventEmitter, OnInit, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SortBarComponent } from './sort-bar/sort-bar.component';
import { SelectModeComponent } from './select-mode/select-mode.component';
import { InclusionSliderComponent } from './inclusion-slider/inclusion-slider.component';
import { ProgressIndicatorsComponent } from './progress-indicators/progress-indicators.component';
import { MediaListComponent } from './media-list/media-list.component';
import { StripeOverviewComponent } from './stripe-overview/stripe-overview.component';
import { AutopilotPanelComponent } from './autopilot-panel/autopilot-panel.component';
import { LeftViewSettingsModalComponent } from '../modals/left-view-settings-modal/left-view-settings-modal.component';
import { MediaItem, LabelingStatusResponse } from '../../models/api.models';
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
    LeftViewSettingsModalComponent,
  ],
  templateUrl: './left-panel.component.html',
  styleUrl: './left-panel.component.scss',
})
export class LeftPanelComponent implements OnInit {
  @Input() medias: MediaItem[] = [];
  @Input() sortOrder: SortedItem[] | null = null;
  @Input() threshold: number | null = null;
  @Input() selectedId: number | null = null;
  @Input() goodVotes: Set<number> = new Set();
  @Input() badVotes: Set<number> = new Set();
  @Input() sortMode: SortMode = 'text';
  @Input() selectMode: SelectMode = 'top';
  @Input() inclusion: number = 0;
  @Input() sortBusy = false;
  @Input() sortStatus = '';
  @Input() labelingStatus: LabelingStatusResponse | null = null;
  @Input() viewMode: 'grid' | 'list' = 'list';
  @Input() gridColumns: number = 2;
  @Input() focusMode: 'click' | 'hover' = 'click';
  @Input() loadSortLabel = '';
  @Input() textQuery = '';
  @Input() autopilotCollapsed = false;
  @Input() autopilotEnabled = true;

  @Output() sortModeChange = new EventEmitter<SortMode>();
  @Output() selectModeChange = new EventEmitter<SelectMode>();
  @Output() inclusionChange = new EventEmitter<number>();
  @Output() textSort = new EventEmitter<string>();
  @Output() learnedSort = new EventEmitter<void>();
  @Output() loadSort = new EventEmitter<void>();
  @Output() detectorLoaded = new EventEmitter<unknown>();
  @Output() exampleSortStarted = new EventEmitter<unknown>();
  @Output() mediaSelect = new EventEmitter<number>();
  @Output() mediaVote = new EventEmitter<{ id: number; vote: 'good' | 'bad' }>();
  @Output() indicatorClick = new EventEmitter<string>();
  @Output() autopilotStart = new EventEmitter<void>();
  @Output() autopilotStop = new EventEmitter<void>();
  @Output() autopilotToggleCollapse = new EventEmitter<void>();
  @Output() autopilotEnabledChange = new EventEmitter<boolean>();

  @ViewChild(MediaListComponent) mediaListComponent!: MediaListComponent;

  activeTab: 'manual' | 'autopilot' = 'autopilot';
  showLeftViewSettings = false;

  ngOnInit(): void {
    this.activeTab = this.autopilotEnabled ? 'autopilot' : 'manual';
    if (this.autopilotEnabled) {
      this.autopilotStart.emit();
    }
  }

  onStripeClick(index: number): void {
    if (this.mediaListComponent) {
      this.mediaListComponent.scrollToIndex(index);
    }
  }

  setTab(tab: 'manual' | 'autopilot'): void {
    if (tab === this.activeTab) return;
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
