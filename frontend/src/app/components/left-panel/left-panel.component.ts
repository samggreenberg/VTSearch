import { Component, Input, Output, EventEmitter, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SortBarComponent } from './sort-bar/sort-bar.component';
import { SelectModeComponent } from './select-mode/select-mode.component';
import { InclusionSliderComponent } from './inclusion-slider/inclusion-slider.component';
import { ProgressIndicatorsComponent } from './progress-indicators/progress-indicators.component';
import { MediaListComponent } from './media-list/media-list.component';
import { StripeOverviewComponent } from './stripe-overview/stripe-overview.component';
import { AutopilotPanelComponent } from './autopilot-panel/autopilot-panel.component';
import { MediaItem, LabelingStatusResponse } from '../../models/api.models';

export type SortMode = 'text' | 'learned' | 'load';
export type SelectMode = 'top' | 'hard' | 'new';

export interface SortedItem {
  id: number;
  score: number;
}

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
  @Input() showThumbnails = true;
  @Input() loadSortLabel = '';

  @Output() sortModeChange = new EventEmitter<SortMode>();
  @Output() selectModeChange = new EventEmitter<SelectMode>();
  @Output() inclusionChange = new EventEmitter<number>();
  @Output() textSort = new EventEmitter<string>();
  @Output() learnedSort = new EventEmitter<void>();
  @Output() loadSort = new EventEmitter<void>();
  @Output() mediaSelect = new EventEmitter<number>();
  @Output() indicatorClick = new EventEmitter<string>();
  @Output() autopilotStart = new EventEmitter<void>();
  @Output() autopilotStop = new EventEmitter<void>();

  activeTab: 'manual' | 'autopilot' = 'autopilot';

  ngOnInit(): void {
    this.autopilotStart.emit();
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
  }
}
