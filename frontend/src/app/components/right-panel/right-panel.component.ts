import { ChangeDetectionStrategy, Component, computed, DestroyRef, effect, inject, input, OnDestroy, OnInit, output, signal, untracked } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { CommonModule } from '@angular/common';
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
  changeDetection: ChangeDetectionStrategy.OnPush,
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
export class RightPanelComponent implements OnInit, OnDestroy {
  private detectorsRegistryApi = inject(DetectorsRegistryApiService);
  voteState = inject(VoteStateService);
  labelsetState = inject(LabelsetStateService);
  private settingsState = inject(SettingsStateService);
  private dialog = inject(VtDialogService);

  readonly medias = input<Media[]>([]);
  readonly trainMode = input<TrainModeContext | null>(null);
  readonly focusMode = input<'click' | 'hover'>('click');
  /** 'label' = Labeling mode (detector export allowed), 'find' = Finding mode (no detector export). */
  readonly mode = input<'label' | 'find'>('label');
  /**
   * Trainable model name that owns the labels shown in the right pane.
   * When set in label mode, the pane is sourced from the on-disk labelset
   * (so detector labels survive across dataset switches).  When empty, the
   * pane falls back to cid-based vote display.
   */
  readonly trainableModelName = input<string | null>(null);
  /** Disable interactive buttons (used during Find scoring). */
  readonly disabled = input(false);
  readonly mediaSelected = output<number>();
  readonly mediaVoted = output<{
    id: number;
    vote: 'good' | 'bad';
}>();
  /** Find mode: browse the positive results as their own UMAP projection. */
  readonly browse = output<void>();
  /** Find mode: promote the goods into their own dataset. */
  readonly toDataset = output<void>();
  /** Find mode: export a label partition (good / bad). */
  readonly exportLabels = output<'good' | 'bad'>();
  /** Find mode: open the detector-evaluation Stats modal. */
  readonly stats = output<void>();
  /** Find mode: fold the corrections into the detector's labelset and retrain. */
  readonly addCorrections = output<void>();

  // Vote piles mirror VoteStateService, which is signal-backed. `computed`
  // wrappers track the service getters and recompute (with stable, memoised
  // identity between changes) only when the underlying votes change, so under
  // zoneless the bound label-lists repaint on a vote/poll without the former
  // `subscribe`-into-plain-field plumbing (zoneless-migration.md, Phase 2.5/2.8).
  readonly goodIds = computed(() => Array.from(this.voteState.goodVotes));
  readonly badIds = computed(() => Array.from(this.voteState.badVotes));
  readonly verifiedIds = computed(() => this.voteState.verifiedIds);
  readonly clickTimes = computed(() => this.voteState.clickTimes);
  readonly learnedScores = computed(() => this.voteState.learnedScores);
  /** Normalised region boxes for good votes, keyed by media id; drives cropped
   *  Good-pile thumbnails when an item was region-voted. */
  readonly goodRegionBoxes = computed(() => this.voteState.goodRegionBoxes);
  // Template-bound and written from the LabelsetStateService `good$`/`bad$`
  // subscribes — which do not schedule CD for a plain field under zoneless —
  // so they are signals. (`sortMode` stays plain: only written from the bound
  // `(sortModeChange)` handler.)
  readonly goodElements = signal<DetectorLabelView[]>([]);
  readonly badElements = signal<DetectorLabelView[]>([]);
  sortMode: LabelSortMode = 'time-desc';
  showLabelImport = false;
  showExport = false;

  protected readonly currentMediaType = signal('');

  /** Right-pane thumbnail size for the active media type. A `computed` over the
   *  settings signal, so a size change made anywhere (the view controls, the
   *  Settings modal, another panel) repaints this one with no mirroring. */
  private readonly gridIconSizeRight = this.settingsState.perMediaType<string>(
    'grid_icon_size_right',
    this.currentMediaType,
    { fallback: 'M' },
  );
  readonly gridGoalWidth = computed(() => iconSizeToGoalWidth(this.gridIconSizeRight.value()));
  private readonly destroyRef = inject(DestroyRef);

  constructor() {
    // Track the active media type off the bound medias so the grid icon size
    // follows the dataset (`gridGoalWidth` is keyed on it). (Replaces the
    // former `medias` ngOnChanges branch.)
    effect(() => {
      const medias = this.medias();
      if (medias.length === 0) return;
      const newType = medias[0].media_type;
      if (newType !== untracked(this.currentMediaType)) {
        this.currentMediaType.set(newType);
      }
    });

    // Keep the labelset polling in sync with the labelset/vote source. Reads
    // `mode` and `trainableModelName` via `useLabelset`, so it re-runs whenever
    // either changes. (Replaces the former `trainableModelName`/`mode`
    // ngOnChanges branch and the labelset block of ngOnInit.)
    effect(() => {
      if (this.useLabelset) {
        this.labelsetState.setModel(this.trainableModelName());
        this.labelsetState.startPolling();
      } else {
        this.labelsetState.stopPolling();
        this.labelsetState.setModel(null);
      }
    });
  }

  /** True when the right pane should be sourced from the labelset (not /api/votes). */
  get useLabelset(): boolean {
    return this.mode() === 'label' && !!this.trainableModelName();
  }

  /**
   * Ids shown in the good bucket. In Find mode the right pane is the verified
   * pile only (the unverified goods live on the left work queue), so it shows
   * just ``good ∩ verified``. In Label mode it shows every good vote.
   */
  get goodDisplayIds(): number[] {
    if (this.mode() !== 'find') return this.goodIds();
    const verified = this.verifiedIds();
    return this.goodIds().filter((id) => verified.has(id));
  }

  /** Bad-bucket counterpart of {@link goodDisplayIds}. */
  get badDisplayIds(): number[] {
    if (this.mode() !== 'find') return this.badIds();
    const verified = this.verifiedIds();
    return this.badIds().filter((id) => verified.has(id));
  }

  /** Find mode: count of unverified goods (the left work queue above the cutoff). */
  get unverifiedGoodCount(): number {
    return this.goodIds().length - this.goodDisplayIds.length;
  }

  /** Find mode: count of unverified bads (the left work queue below the cutoff). */
  get unverifiedBadCount(): number {
    return this.badIds().length - this.badDisplayIds.length;
  }

  ngOnInit(): void {
    this.settingsState.load();
    this.voteState.startPolling();
    this.subscribeToLabelset();
  }

  ngOnDestroy(): void {
    this.voteState.stopPolling();
    this.labelsetState.stopPolling();
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

  onAddCorrections(): void {
    this.addCorrections.emit();
  }

  onDetectorRenamed(newName: string): void {
    const trainMode = this.trainMode();
    if (!trainMode?.model?.registry_id) return;
    const registryId = trainMode.model.registry_id;
    this.detectorsRegistryApi.renameInRegistry(registryId, newName).subscribe({
      next: response => {
        if (trainMode.model) {
          trainMode.model.name = newName;
        }
        this.detectorsRegistryApi.promptMoveOrphanedLabelsetFile(
          this.dialog,
          registryId,
          response.pending_labelset_move,
        );
      },
    });
  }

  private subscribeToLabelset(): void {
    this.labelsetState.good$
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((elements) => {
        this.goodElements.set(elements);
      });
    this.labelsetState.bad$
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((elements) => {
        this.badElements.set(elements);
      });
    this.labelsetState.mediaType$
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((mt) => {
        if (this.useLabelset && mt && mt !== this.currentMediaType()) {
          this.currentMediaType.set(mt);
        }
      });
  }
}
