import { Component, EventEmitter, Input, OnDestroy, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { TrainableModelsApiService } from '../../services/trainable-models-api.service';
import { MediaItem } from '../../models/api.models';
import { VoteStateService } from '../../services/vote-state.service';
import { SettingsStateService } from '../../services/settings-state.service';
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

  constructor(
    private modelsApi: TrainableModelsApiService,
    public voteState: VoteStateService,
    private settingsState: SettingsStateService,
  ) {}

  ngOnInit(): void {
    this.settingsState.load();
    this.loadSettings();
    this.voteState.startPolling();
    this.subscribeToVotes();
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
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
    this.settingsState.settings$
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: settings => {
          if (!settings) return;
          this.showThumbnails = settings.show_thumbnails_right ?? true;
        },
      });
  }

  private subscribeToVotes(): void {
    this.voteState.goodVotes$
      .pipe(takeUntil(this.destroy$))
      .subscribe((votes) => {
        this.goodIds = Array.from(votes);
      });
    this.voteState.badVotes$
      .pipe(takeUntil(this.destroy$))
      .subscribe((votes) => {
        this.badIds = Array.from(votes);
      });
    this.voteState.clickTimes$
      .pipe(takeUntil(this.destroy$))
      .subscribe((times) => {
        this.clickTimes = times;
      });
    this.voteState.learnedScores$
      .pipe(takeUntil(this.destroy$))
      .subscribe((scores) => {
        this.learnedScores = scores;
      });
  }
}
