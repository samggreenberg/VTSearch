import { Component, EventEmitter, Input, OnChanges, OnDestroy, OnInit, Output, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { DetectorsRegistryApiService } from '../../services/detectors-registry-api.service';
import type { DetectorLabelView } from '../../generated/api-client/models/detector-label-view';
import { Media } from '../../models/api.models';
import { VoteStateService } from '../../services/vote-state.service';
import { LabelsetStateService } from '../../services/labelset-state.service';
import { SettingsStateService } from '../../services/settings-state.service';
import { VtDialogService } from '../../services/dialog.service';

import { iconSizeToGoalWidth } from '../../utils/grid-icon-size';
import { LabelSortComponent, LabelSortMode } from './label-sort/label-sort.component';
import { LabelListComponent } from './label-list/label-list.component';
import { LabelsetListComponent } from './labelset-list/labelset-list.component';
import { DetectorContextBarComponent } from './detector-context-bar/detector-context-bar.component';
import { ExportModalComponent } from '../modals/export-modal/export-modal.component';
import { LabelImporterModalComponent } from '../modals/label-importer-modal/label-importer-modal.component';
import { ViewControlsComponent } from '../view-controls/view-controls.component';
import { IconComponent } from '../icon/icon.component';

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
    IconComponent,
  ],
  templateUrl: './right-panel.component.html',
  styleUrl: './right-panel.component.scss',
})
export class RightPanelComponent implements OnInit, OnChanges, OnDestroy {
  @Input() medias: Media[] = [];
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
  /** Find mode: browse the positive results as their own UMAP projection. */
  @Output() browse = new EventEmitter<void>();
  /** Find mode: promote the goods into their own dataset. */
  @Output() toDataset = new EventEmitter<void>();
  /** Find mode: export a label partition (good / bad). */
  @Output() exportLabels = new EventEmitter<'good' | 'bad'>();
  /** Find mode: open the detector-evaluation Stats modal. */
  @Output() stats = new EventEmitter<void>();

  goodIds: number[] = [];
  badIds: number[] = [];
  verifiedIds: Set<number> = new Set();
  clickTimes: Record<string, number> = {};
  learnedScores: Record<string, number> = {};
  goodElements: DetectorLabelView[] = [];
  badElements: DetectorLabelView[] = [];
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
    private detectorsRegistryApi: DetectorsRegistryApiService,
    public voteState: VoteStateService,
    public labelsetState: LabelsetStateService,
    private settingsState: SettingsStateService,
    private dialog: VtDialogService,
  ) {}

  /** True when the right pane should be sourced from the labelset (not /api/votes). */
  get useLabelset(): boolean {
    return this.mode === 'label' && !!this.trainableModelName;
  }

  /**
   * Ids shown in the good bucket. In Find mode the right pane is the verified
   * pile only (the unverified goods live on the left work queue), so it shows
   * just ``good ∩ verified``. In Label mode it shows every good vote.
   */
  get goodDisplayIds(): number[] {
    if (this.mode !== 'find') return this.goodIds;
    return this.goodIds.filter((id) => this.verifiedIds.has(id));
  }

  /** Bad-bucket counterpart of {@link goodDisplayIds}. */
  get badDisplayIds(): number[] {
    if (this.mode !== 'find') return this.badIds;
    return this.badIds.filter((id) => this.verifiedIds.has(id));
  }

  /** Find mode: count of unverified goods (the left work queue above the cutoff). */
  get unverifiedGoodCount(): number {
    return this.goodIds.length - this.goodDisplayIds.length;
  }

  /** Find mode: count of unverified bads (the left work queue below the cutoff). */
  get unverifiedBadCount(): number {
    return this.badIds.length - this.badDisplayIds.length;
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
      const newType = this.medias[0].media_type;
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
  onLabelsetElementSelected(elem: DetectorLabelView): void {
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

  onBrowse(): void {
    this.browse.emit();
  }

  onToDataset(): void {
    this.toDataset.emit();
  }

  onExportGood(): void {
    this.exportLabels.emit('good');
  }

  onExportBad(): void {
    this.exportLabels.emit('bad');
  }

  onStats(): void {
    this.stats.emit();
  }

  onDetectorRenamed(newName: string): void {
    if (!this.trainMode?.model?.registry_id) return;
    const registryId = this.trainMode.model.registry_id;
    this.detectorsRegistryApi.renameInRegistry(registryId, newName).subscribe({
      next: response => {
        if (this.trainMode?.model) {
          this.trainMode.model.name = newName;
        }
        this.detectorsRegistryApi.promptMoveOrphanedLabelsetFile(
          this.dialog,
          registryId,
          response.pending_labelset_move,
        );
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
    this.voteState.verifiedIds$
      .pipe(takeUntil(this.destroy$))
      .subscribe((ids) => {
        this.verifiedIds = ids;
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
