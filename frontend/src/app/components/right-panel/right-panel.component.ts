import { Component, EventEmitter, Input, OnChanges, OnDestroy, OnInit, Output, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { TrainableModelsApiService } from '../../services/trainable-models-api.service';
import { MediaItem } from '../../models/api.models';
import { VoteStateService } from '../../services/vote-state.service';
import { SettingsStateService } from '../../services/settings-state.service';

import { iconSizeToGoalWidth } from '../../utils/grid-icon-size';
import { LabelSortComponent, LabelSortMode } from './label-sort/label-sort.component';
import { LabelListComponent } from './label-list/label-list.component';
import { DetectorContextBarComponent } from './detector-context-bar/detector-context-bar.component';
import { ExportModalComponent } from '../modals/export-modal/export-modal.component';
import { LabelImporterModalComponent } from '../modals/label-importer-modal/label-importer-modal.component';
import { ViewControlsComponent } from '../view-controls/view-controls.component';

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
    ExportModalComponent,
    LabelImporterModalComponent,
    ViewControlsComponent,
  ],
  templateUrl: './right-panel.component.html',
  styleUrl: './right-panel.component.scss',
})
export class RightPanelComponent implements OnInit, OnChanges, OnDestroy {
  @Input() medias: MediaItem[] = [];
  @Input() trainMode: TrainModeContext | null = null;
  @Input() focusMode: 'click' | 'hover' = 'click';
  /** 'label' = Labeling mode (detector export allowed), 'find' = Finding mode (no detector export). */
  @Input() mode: 'label' | 'find' = 'label';
  /** Disable interactive buttons (used during Find scoring). */
  @Input() disabled = false;
  @Output() mediaSelected = new EventEmitter<number>();
  @Output() mediaVoted = new EventEmitter<{ id: number; vote: 'good' | 'bad' }>();

  goodIds: number[] = [];
  badIds: number[] = [];
  clickTimes: Record<string, number> = {};
  learnedScores: Record<string, number> = {};
  sortMode: LabelSortMode = 'time-desc';
  viewMode: 'grid' | 'list' = 'grid';
  gridGoalWidth: number = 80;
  showLabelImport = false;
  showExport = false;

  private viewModeRightDict: Record<string, 'grid' | 'list'> = {};
  private gridIconSizeRightDict: Record<string, string> = {};
  protected currentMediaType = '';
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

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['medias'] && this.medias.length > 0) {
      const newType = this.medias[0].type;
      if (newType !== this.currentMediaType) {
        this.currentMediaType = newType;
        this.viewMode = this.viewModeRightDict[newType] ?? 'grid';
        this.gridGoalWidth = iconSizeToGoalWidth(this.gridIconSizeRightDict[newType] ?? 'M');
      }
    }
  }

  ngOnDestroy(): void {
    this.voteState.stopPolling();
    this.destroy$.next();
    this.destroy$.complete();
  }

  onSortModeChange(mode: LabelSortMode): void {
    this.sortMode = mode;
  }

  onMediaSelected(id: number): void {
    this.mediaSelected.emit(id);
  }

  onMediaVote(event: { id: number; vote: 'good' | 'bad' }): void {
    this.mediaVoted.emit(event);
  }

  onImportLabels(): void {
    this.showLabelImport = true;
  }

  onExport(): void {
    this.showExport = true;
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
          const dict = settings.view_mode_right;
          if (dict && typeof dict === 'object') {
            this.viewModeRightDict = dict as Record<string, 'grid' | 'list'>;
            if (this.currentMediaType) {
              this.viewMode = this.viewModeRightDict[this.currentMediaType] ?? 'grid';
            }
          }
          const sizeDict = settings.grid_icon_size_right;
          if (sizeDict && typeof sizeDict === 'object') {
            this.gridIconSizeRightDict = sizeDict as Record<string, string>;
            if (this.currentMediaType) {
              this.gridGoalWidth = iconSizeToGoalWidth(this.gridIconSizeRightDict[this.currentMediaType] ?? 'M');
            }
          }
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
