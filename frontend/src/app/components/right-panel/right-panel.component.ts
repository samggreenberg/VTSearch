import { Component, EventEmitter, Input, OnChanges, OnDestroy, OnInit, Output, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { TrainableModelsApiService } from '../../services/trainable-models-api.service';
import { LabelElement, MediaItem } from '../../models/api.models';
import { VoteStateService } from '../../services/vote-state.service';
import { LabelsetStateService } from '../../services/labelset-state.service';
import { SettingsStateService } from '../../services/settings-state.service';

import { iconSizeToGoalWidth } from '../../utils/grid-icon-size';
import { LabelSortComponent, LabelSortMode } from './label-sort/label-sort.component';
import { LabelListComponent } from './label-list/label-list.component';
import { LabelsetListComponent } from './labelset-list/labelset-list.component';
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
    LabelsetListComponent,
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
  /**
   * Trainable model name that owns the labels shown in the right pane.
   * When set in label mode, the pane is sourced from the on-disk labelset
   * (so detector labels survive across dataset switches).  When empty, the
   * pane falls back to cid-based vote display.
   */
  @Input() trainableModelName: string | null = null;
  /** Disable interactive buttons (used during Find scoring). */
  @Input() disabled = false;
  @Output() mediaSelected = new EventEmitter<number>();
  @Output() mediaVoted = new EventEmitter<{ id: number; vote: 'good' | 'bad' }>();

  goodIds: number[] = [];
  badIds: number[] = [];
  clickTimes: Record<string, number> = {};
  learnedScores: Record<string, number> = {};
  goodElements: LabelElement[] = [];
  badElements: LabelElement[] = [];
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
    public labelsetState: LabelsetStateService,
    private settingsState: SettingsStateService,
  ) {}

  /** True when the right pane should be sourced from the labelset (not /api/votes). */
  get useLabelset(): boolean {
    return this.mode === 'label' && !!this.trainableModelName;
  }

  ngOnInit(): void {
    this.settingsState.load();
    this.loadSettings();
    this.voteState.startPolling();
    this.subscribeToVotes();
    if (this.useLabelset) {
      this.labelsetState.setModel(this.trainableModelName);
      this.labelsetState.startPolling();
    }
    this.subscribeToLabelset();
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
    if (changes['trainableModelName'] || changes['mode']) {
      if (this.useLabelset) {
        this.labelsetState.setModel(this.trainableModelName);
        this.labelsetState.startPolling();
      } else {
        this.labelsetState.stopPolling();
        this.labelsetState.setModel(null);
      }
    }
  }

  ngOnDestroy(): void {
    this.voteState.stopPolling();
    this.labelsetState.stopPolling();
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

  /** Right-pane element click in labelset mode.  Focus the matching cid
   *  on the left when the element resolves into the active dataset;
   *  otherwise the element exists only in the labelset (e.g. trained on a
   *  different dataset) and there's nothing to focus. */
  onLabelsetElementSelected(elem: LabelElement): void {
    if (elem.cid !== null && elem.cid !== undefined) {
      this.mediaSelected.emit(elem.cid);
    }
  }

  onLabelsetElementVote(event: { id: string; vote: 'good' | 'bad' }): void {
    this.labelsetState.vote(event.id, event.vote);
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

  private subscribeToLabelset(): void {
    this.labelsetState.good$
      .pipe(takeUntil(this.destroy$))
      .subscribe((elements) => {
        this.goodElements = elements;
      });
    this.labelsetState.bad$
      .pipe(takeUntil(this.destroy$))
      .subscribe((elements) => {
        this.badElements = elements;
      });
    this.labelsetState.mediaType$
      .pipe(takeUntil(this.destroy$))
      .subscribe((mt) => {
        if (this.useLabelset && mt && mt !== this.currentMediaType) {
          this.currentMediaType = mt;
          this.viewMode = this.viewModeRightDict[mt] ?? 'grid';
          this.gridGoalWidth = iconSizeToGoalWidth(this.gridIconSizeRightDict[mt] ?? 'M');
        }
      });
  }
}
