import { Component, EventEmitter, Input, OnDestroy, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subject, timer } from 'rxjs';
import { switchMap, takeUntil } from 'rxjs/operators';
import { SortingApiService } from '../../services/sorting-api.service';
import { SettingsApiService } from '../../services/settings-api.service';
import { TrainableModelsApiService } from '../../services/trainable-models-api.service';
import { MediaItem, VotesResponse } from '../../models/api.models';
import { LabelSortComponent, LabelSortMode } from './label-sort/label-sort.component';
import { LabelListComponent } from './label-list/label-list.component';
import { DetectorContextBarComponent } from './detector-context-bar/detector-context-bar.component';

export interface TrainModeContext {
  model: { name: string; registry_id?: string };
}

@Component({
  selector: 'vt-right-panel',
  standalone: true,
  imports: [
    CommonModule,
    LabelSortComponent,
    LabelListComponent,
    DetectorContextBarComponent,
  ],
  templateUrl: './right-panel.component.html',
  styleUrl: './right-panel.component.scss',
})
export class RightPanelComponent implements OnInit, OnDestroy {
  @Input() medias: MediaItem[] = [];
  @Input() trainMode: TrainModeContext | null = null;
  @Output() mediaSelected = new EventEmitter<number>();

  goodIds: number[] = [];
  badIds: number[] = [];
  clickTimes: Record<string, number> = {};
  learnedScores: Record<string, number> = {};
  sortMode: LabelSortMode = 'time-desc';
  showThumbnails = true;

  private destroy$ = new Subject<void>();
  private refresh$ = new Subject<void>();

  constructor(
    private sortingApi: SortingApiService,
    private settingsApi: SettingsApiService,
    private modelsApi: TrainableModelsApiService,
  ) {}

  ngOnInit(): void {
    this.loadSettings();
    this.startPolling();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
    this.refresh$.next();
    this.refresh$.complete();
  }

  refreshVotes(): void {
    this.refresh$.next();
  }

  onSortModeChange(mode: LabelSortMode): void {
    this.sortMode = mode;
  }

  onMediaSelected(id: number): void {
    this.mediaSelected.emit(id);
  }

  onDetectorRenamed(newName: string): void {
    if (!this.trainMode?.model?.registry_id) return;
    this.modelsApi.renameInRegistry(this.trainMode.model.registry_id, newName).subscribe({
      next: () => {
        if (this.trainMode?.model) {
          this.trainMode.model.name = newName;
        }
      },
    });
  }

  private loadSettings(): void {
    this.settingsApi.getSettings()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: settings => {
          this.showThumbnails = settings.show_thumbnails_right ?? true;
        },
      });
  }

  private startPolling(): void {
    timer(0, 2000)
      .pipe(
        takeUntil(this.destroy$),
        switchMap(() => this.sortingApi.getVotes()),
      )
      .subscribe({
        next: (votes: VotesResponse) => {
          this.goodIds = votes.good;
          this.badIds = votes.bad;
          this.clickTimes = votes.click_times;
          this.learnedScores = votes.learned_scores;
        },
      });
  }
}
