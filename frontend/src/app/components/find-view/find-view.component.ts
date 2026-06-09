import { Component, OnInit, OnDestroy, ElementRef, ViewChild, NgZone, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { Subject, Subscription } from 'rxjs';
import { finalize, takeUntil } from 'rxjs/operators';
import { LeftPanelComponent } from '../left-panel/left-panel.component';
import { CenterPanelComponent } from '../center-panel/center-panel.component';
import { RightPanelComponent } from '../right-panel/right-panel.component';
import { ProgressBarComponent } from '../progress-bar/progress-bar.component';
import { ExportModalComponent } from '../modals/export-modal/export-modal.component';
import { FindStatsModalComponent } from '../modals/find-stats-modal/find-stats-modal.component';
import type { LabelFilter } from '../../services/sorting-api.service';
import { MediasApiService } from '../../services/medias-api.service';
import { DetectorsFindApiService } from '../../services/detectors-find-api.service';
import { DatasetsRegistryApiService } from '../../services/datasets-registry-api.service';
import { DatasetsCrudApiService } from '../../services/datasets-crud-api.service';
import { ToastService } from '../../services/toast.service';
import { VtDialogService } from '../../services/dialog.service';
import { ActiveContextService } from '../../services/active-context.service';
import { DatasetStateService } from '../../services/dataset-state.service';
import { MediaStateService } from '../../services/media-state.service';
import { VoteStateService } from '../../services/vote-state.service';
import { SortStateService } from '../../services/sort-state.service';
import { SortingApiService } from '../../services/sorting-api.service';
import { SettingsStateService } from '../../services/settings-state.service';
import { ProgressEventsService } from '../../services/progress-events.service';
import { BrowseSubsetService } from '../../services/browse-subset.service';
import { ProgressEvent } from '../../models/api.models';
import { formatProgressMessage } from '../../utils/format-progress';
import { iconSizeToGoalWidth, snapPanelWidthToGridColumns } from '../../utils/grid-icon-size';

@Component({
  selector: 'vt-find-view',
  standalone: true,
  imports: [
    CommonModule,
    LeftPanelComponent,
    CenterPanelComponent,
    RightPanelComponent,
    ProgressBarComponent,
    ExportModalComponent,
    FindStatsModalComponent,
  ],
  templateUrl: './find-view.component.html',
  styleUrl: './find-view.component.scss',
})
export class FindViewComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('layout', { static: true }) layoutRef!: ElementRef<HTMLElement>;
  @ViewChild(CenterPanelComponent) centerPanel?: CenterPanelComponent;

  datasetName = '';
  viewModeLeft: 'grid' | 'list' = 'list';
  gridGoalWidthLeft: number = 80;
  focusModeLeft: 'click' | 'hover' = 'click';
  focusModeRight: 'click' | 'hover' = 'click';
  private viewModeLeftDict: Record<string, 'grid' | 'list'> = {};
  private gridIconSizeLeftDict: Record<string, string> = {};
  private focusModeLeftDict: Record<string, 'click' | 'hover'> = {};
  private focusModeRightDict: Record<string, 'click' | 'hover'> = {};
  private panelPxLeftDict: Record<string, number> = {};
  private panelPxRightDict: Record<string, number> = {};
  private currentMediaType = '';
  leftWidth = 260;
  rightWidth = 300;

  /** Verified ids (Find mode): the right-panel confirmed pile. */
  verifiedIds: Set<number> = new Set();
  /** Export modal visibility + the label filter it opens on. */
  showExport = false;
  exportFilter: LabelFilter = 'good';
  /** Detector-evaluation Stats modal visibility. */
  showStats = false;

  private readonly LEFT_MIN = 180;
  private readonly RIGHT_MIN = 150;
  private readonly CENTER_MIN = 100;
  private readonly DIVIDER_TOTAL = 16; // 2 × 8px dividers
  private destroy$ = new Subject<void>();
  private dragging = false;
  private draggingRight = false;
  private boundMouseMove = this.onMouseMove.bind(this);
  private boundMouseUp = this.onMouseUp.bind(this);
  private boundRightMouseMove = this.onRightMouseMove.bind(this);
  private boundRightMouseUp = this.onRightMouseUp.bind(this);

  constructor(
    private mediasApi: MediasApiService,
    private detectorsFindApi: DetectorsFindApiService,
    private datasetsRegistryApi: DatasetsRegistryApiService,
    private datasetsCrudApi: DatasetsCrudApiService,
    private toast: ToastService,
    private dialog: VtDialogService,
    private ngZone: NgZone,
    private activeContext: ActiveContextService,
    private datasetState: DatasetStateService,
    public mediaState: MediaStateService,
    public voteState: VoteStateService,
    public sortState: SortStateService,
    private sortingApi: SortingApiService,
    private settingsState: SettingsStateService,
    private progressEvents: ProgressEventsService,
    private browseSubset: BrowseSubsetService,
    private router: Router,
  ) {}

  ngOnInit(): void {
    this.layoutRef.nativeElement.style.setProperty('--left-width', `${this.leftWidth}px`);
    this.layoutRef.nativeElement.style.setProperty('--right-width', `${this.rightWidth}px`);
    this.mediaState.loadMedias();
    this.voteState.loadVotes();
    this.voteState.verifiedIds$
      .pipe(takeUntil(this.destroy$))
      .subscribe((ids) => (this.verifiedIds = ids));
    this.loadSettings();
    this.datasetsRegistryApi.getStatus().pipe(takeUntil(this.destroy$)).subscribe({
      next: (status) => { this.datasetName = status.display_name || ''; },
    });

    // When medias arrive, run the find-label scoring
    this.mediaState.medias$
      .pipe(takeUntil(this.destroy$))
      .subscribe((medias) => {
        if (medias.length > 0) {
          const newType = medias[0].media_type;
          if (newType !== this.currentMediaType) {
            this.currentMediaType = newType;
            this.viewModeLeft = this.viewModeLeftDict[newType] ?? 'list';
            this.gridGoalWidthLeft = iconSizeToGoalWidth(this.gridIconSizeLeftDict[newType] ?? 'M');
            this.focusModeLeft = this.focusModeLeftDict[newType] ?? 'click';
            this.focusModeRight = this.focusModeRightDict[newType] ?? 'click';
            this.applyPanelPx(newType);
          }
        }
      });

    // Run find-label to score and label all medias — unless we're returning
    // from the Browser after a "Remove from Good" cull. In that case the
    // backend vote lists already reflect the removed items (now Bad), and the
    // loadVotes() above refreshes them; re-running find here would re-score
    // with the unchanged model and re-promote those items to Good, undoing
    // the cull. Keep the cull instead (the user's decision).
    // Seed the inclusion slider from the active detector's context value
    // (GET /api/inclusion resolves per-detector, falling back to the
    // user-settings default the first time it's read). This keeps Find's
    // slider in step with whatever the detector was last trained at.
    this.seedInclusion();

    if (!this.browseSubset.consumeReturningToFind()) {
      this.runFindLabel();
    }

    // Reload + rescore when the active pair changes via the top-bar
    // switcher or a route-param swap (`/find/:ds/:det` → `/find/:ds2/:det2`).
    // Skip the first emission (ngOnInit already triggered the initial
    // loads + runFindLabel call above).
    let firstPair = true;
    this.activeContext.pair$
      .pipe(takeUntil(this.destroy$))
      .subscribe(() => {
        if (firstPair) {
          firstPair = false;
          return;
        }
        this.reloadForNewPair();
      });
  }

  private reloadForNewPair(): void {
    this.sortState.setSortResults([], 0);
    this.sortState.setSortStatus('');
    this.sortState.setSortProgress(0, 0);
    this.voteState.clear();
    this.mediaState.loadMedias();
    this.voteState.loadVotes();
    this.datasetsRegistryApi.getStatus().pipe(takeUntil(this.destroy$)).subscribe({
      next: (status) => { this.datasetName = status.display_name || ''; },
    });
    this.seedInclusion();
    this.runFindLabel();
  }

  /** Pull the active detector's per-detector inclusion into the slider. */
  private seedInclusion(): void {
    this.sortingApi
      .getInclusion()
      .pipe(takeUntil(this.destroy$))
      .subscribe({ next: (resp) => this.sortState.setInclusion(resp.inclusion) });
  }

  ngAfterViewInit(): void {
    setTimeout(() => this.centerPanel?.init());
  }

  ngOnDestroy(): void {
    this.stopProgressPolling();
    this.destroy$.next();
    this.destroy$.complete();
    this.voteState.stopPolling();
    document.removeEventListener('mousemove', this.boundMouseMove);
    document.removeEventListener('mouseup', this.boundMouseUp);
    document.removeEventListener('mousemove', this.boundRightMouseMove);
    document.removeEventListener('mouseup', this.boundRightMouseUp);
  }

  // --- Find-label scoring ---

  private progressPollSub: Subscription | null = null;

  private startProgressPolling(): void {
    this.stopProgressPolling();
    this.progressPollSub = this.progressEvents.find$
      .pipe(takeUntil(this.destroy$))
      .subscribe((prog: ProgressEvent) => {
        if (prog.status === 'running') {
          this.sortState.setSortStatus(formatProgressMessage(prog, 'Scoring with detector…'));
          this.sortState.setSortProgress(prog.current ?? 0, prog.total ?? 0);
        }
      });
  }

  private stopProgressPolling(): void {
    if (this.progressPollSub) {
      this.progressPollSub.unsubscribe();
      this.progressPollSub = null;
    }
  }

  private runFindLabel(): void {
    const modelId = this.activeContext.modelId;
    if (!modelId) return;

    this.sortState.setSortBusy(true);
    this.sortState.setSortStatus('Scoring with detector…');
    this.sortState.setSortProgress(0, 0);

    // Start polling for progress concurrently
    this.startProgressPolling();

    const modelName =
      this.datasetState.detectors.find((d) => d.id === modelId)?.name || 'Detector';
    this.detectorsFindApi.findLabel({ detector_id: modelId })
      .pipe(
        takeUntil(this.destroy$),
        finalize(() => {
          this.stopProgressPolling();
          this.sortState.setSortBusy(false);
        }),
      )
      .subscribe({
        next: (response: any) => {
          const sorted = response.results.map((r: any) => ({ id: r.id, score: r.score, bestRegion: r.best_region }));
          const threshold = response.threshold;
          // Set sort results for stripe display
          this.sortState.setSortResults(sorted, threshold);
          this.sortState.setLoadSortLabel(modelName);
          this.sortState.setSortStatus('');
          this.sortState.setSortProgress(0, 0);
          // Seed the centre on the marginal positive (lowest item ≥ cutoff).
          // With nothing verified yet this is the same rule auto-advance uses,
          // so seed and advance unify.
          this.queueEmptyNotified = false;
          this.advanceToMarginalPositive();
          // Reload votes to reflect newly applied labels
          this.voteState.loadVotes();
        },
        error: (err: any) => {
          // Extract the server error message so the user sees why scoring failed
          const body = err?.error;
          const warning = body?.warning || body?.error || 'Scoring failed';
          this.sortState.setSortStatus(warning);
          this.sortState.setSortProgress(0, 0);
        },
      });
  }

  // --- Divider drag ---

  onDividerMouseDown(event: MouseEvent): void {
    event.preventDefault();
    this.dragging = true;
    this.ngZone.runOutsideAngular(() => {
      document.addEventListener('mousemove', this.boundMouseMove);
      document.addEventListener('mouseup', this.boundMouseUp);
    });
  }

  private onMouseMove(event: MouseEvent): void {
    if (!this.dragging) return;
    const layoutRect = this.layoutRef.nativeElement.getBoundingClientRect();
    let newWidth = event.clientX - layoutRect.left;
    const leftMax = layoutRect.width - this.DIVIDER_TOTAL - this.CENTER_MIN - this.rightWidth;
    newWidth = Math.max(this.LEFT_MIN, Math.min(leftMax, newWidth));
    this.ngZone.run(() => {
      this.leftWidth = newWidth;
      this.layoutRef.nativeElement.style.setProperty('--left-width', `${newWidth}px`);
    });
  }

  private onMouseUp(): void {
    this.dragging = false;
    document.removeEventListener('mousemove', this.boundMouseMove);
    document.removeEventListener('mouseup', this.boundMouseUp);
    const leftPanelEl = this.layoutRef.nativeElement.querySelector('vt-left-panel') as HTMLElement | null;
    if (leftPanelEl) {
      const snapped = snapPanelWidthToGridColumns(leftPanelEl, this.leftWidth);
      if (snapped !== null) {
        const leftMax = this.layoutRef.nativeElement.getBoundingClientRect().width - this.DIVIDER_TOTAL - this.CENTER_MIN - this.rightWidth;
        const clamped = Math.max(this.LEFT_MIN, Math.min(leftMax, snapped));
        this.ngZone.run(() => {
          this.leftWidth = clamped;
          this.layoutRef.nativeElement.style.setProperty('--left-width', `${clamped}px`);
        });
      }
    }
    this.savePanelPx('left');
  }

  // --- Right divider drag ---

  onRightDividerMouseDown(event: MouseEvent): void {
    event.preventDefault();
    this.draggingRight = true;
    this.ngZone.runOutsideAngular(() => {
      document.addEventListener('mousemove', this.boundRightMouseMove);
      document.addEventListener('mouseup', this.boundRightMouseUp);
    });
  }

  private onRightMouseMove(event: MouseEvent): void {
    if (!this.draggingRight) return;
    const layoutRect = this.layoutRef.nativeElement.getBoundingClientRect();
    let newWidth = layoutRect.right - event.clientX;
    const rightMax = layoutRect.width - this.DIVIDER_TOTAL - this.CENTER_MIN - this.leftWidth;
    newWidth = Math.max(this.RIGHT_MIN, Math.min(rightMax, newWidth));
    this.ngZone.run(() => {
      this.rightWidth = newWidth;
      this.layoutRef.nativeElement.style.setProperty('--right-width', `${newWidth}px`);
    });
  }

  private onRightMouseUp(): void {
    this.draggingRight = false;
    document.removeEventListener('mousemove', this.boundRightMouseMove);
    document.removeEventListener('mouseup', this.boundRightMouseUp);
    const rightPanelEl = this.layoutRef.nativeElement.querySelector('vt-right-panel') as HTMLElement | null;
    if (rightPanelEl) {
      const snapped = snapPanelWidthToGridColumns(rightPanelEl, this.rightWidth);
      if (snapped !== null) {
        const layoutWidth = this.layoutRef.nativeElement.getBoundingClientRect().width;
        const rightMax = layoutWidth - this.DIVIDER_TOTAL - this.CENTER_MIN - this.leftWidth;
        const clamped = Math.max(this.RIGHT_MIN, Math.min(rightMax, snapped));
        this.ngZone.run(() => {
          this.rightWidth = clamped;
          this.layoutRef.nativeElement.style.setProperty('--right-width', `${clamped}px`);
        });
      }
    }
    this.savePanelPx('right');
  }

  // --- Data loading ---

  private loadSettings(): void {
    this.settingsState.load();
    this.settingsState.settings$
      .pipe(takeUntil(this.destroy$))
      .subscribe((settings) => {
        if (!settings) return;
        const dict = settings.view_mode_left;
        if (dict && typeof dict === 'object') {
          this.viewModeLeftDict = dict as Record<string, 'grid' | 'list'>;
          if (this.currentMediaType) {
            this.viewModeLeft = this.viewModeLeftDict[this.currentMediaType] ?? 'list';
          }
        }
        const sizeDict = settings.grid_icon_size_left;
        if (sizeDict && typeof sizeDict === 'object') {
          this.gridIconSizeLeftDict = sizeDict as Record<string, string>;
          if (this.currentMediaType) {
            this.gridGoalWidthLeft = iconSizeToGoalWidth(this.gridIconSizeLeftDict[this.currentMediaType] ?? 'M');
          }
        }
        const fmLeft = settings.focus_mode_left;
        if (fmLeft && typeof fmLeft === 'object') {
          this.focusModeLeftDict = fmLeft as Record<string, 'click' | 'hover'>;
          if (this.currentMediaType) {
            this.focusModeLeft = this.focusModeLeftDict[this.currentMediaType] ?? 'click';
          }
        }
        const fmRight = settings.focus_mode_right;
        if (fmRight && typeof fmRight === 'object') {
          this.focusModeRightDict = fmRight as Record<string, 'click' | 'hover'>;
          if (this.currentMediaType) {
            this.focusModeRight = this.focusModeRightDict[this.currentMediaType] ?? 'click';
          }
        }
        const pplDict = settings.panel_pct_left;
        if (pplDict && typeof pplDict === 'object') {
          this.panelPxLeftDict = pplDict as Record<string, number>;
          if (this.currentMediaType) {
            this.applyPanelPx(this.currentMediaType);
          }
        }
        const pprDict = settings.panel_pct_right;
        if (pprDict && typeof pprDict === 'object') {
          this.panelPxRightDict = pprDict as Record<string, number>;
          if (this.currentMediaType) {
            this.applyPanelPx(this.currentMediaType);
          }
        }
      });
  }

  // --- Media selection ---

  onMediaSelect(id: number): void {
    this.mediaState.selectMedia(id);
  }

  /**
   * Inclusion change in Find: a pure cutoff slide, **no retrain**. Inclusion
   * is the FP/FN cost weight in the labelset min-cost threshold search; the
   * model and every item's frozen score are inclusion-independent, so the
   * slider only moves the green/red line over the cached scores. POST
   * /api/inclusion re-derives the cutoff from the cached fold orderings and
   * re-splits the *unverified* items server-side (verified items hold). We
   * reconcile the new threshold (moves the line) and the re-split votes on the
   * cheap response — there is no scoring spinner.
   */
  onInclusionChange(value: number): void {
    if (this.sortState.sortBusy) return;
    this.sortState.setInclusion(value);
    this.sortingApi
      .setInclusion(value)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (resp) => {
          if (resp.threshold != null && this.sortState.sortOrder) {
            this.sortState.setSortResults(this.sortState.sortOrder, resp.threshold);
          }
          // The server re-thresholded the unverified items over the frozen
          // scores; pull the new good/bad split back for the left/right panes.
          this.voteState.loadVotes();
        },
      });
  }

  onHoverVote(event: { id: number; vote: 'good' | 'bad' }): void {
    if (this.sortState.sortBusy) return;
    const m = this.mediaState.getMedia(event.id);
    const name = m?.filename || m?.origin_name || `#${event.id}`;
    this.voteState
      .submitToggleVoteAndRecord(event.id, event.vote, name)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => {
          this.onMediaVoted(event);
        },
      });
  }

  onMediaVoted(event: { id: number; vote: 'good' | 'bad' }): void {
    // A single-item manual vote (big button or hover) verifies the item: it
    // moves out of the left work queue into the right verified pile. Mirror
    // the server's mark-verified optimistically so the move feels instant —
    // the resulting local polarity (good/bad vs. un-voted) decides verified.
    const verified =
      this.voteState.goodVotes.has(event.id) || this.voteState.badVotes.has(event.id);
    this.voteState.setOptimisticVerified(event.id, verified);
    // Re-fetch labelset counters + the authoritative verified set.
    this.voteState.loadVotes();
    // Auto-advance to the marginal positive (the lowest unverified item still
    // above the cutoff), so "just sit and vote" walks the boundary upward.
    this.advanceToMarginalPositive();
  }

  /**
   * Select the next item to review: the lowest-scored unverified item still
   * above the cutoff (the Unverified Good nearest the line). This is a cheap
   * active-learning order — always the most marginal positive next. The queue
   * is empty exactly when no unverified item remains above the cutoff; that is
   * the done state.  Mirrors the initial seed in {@link runFindLabel}.
   */
  private advanceToMarginalPositive(): void {
    const order = this.sortState.sortOrder;
    const threshold = this.sortState.threshold;
    if (!order || threshold == null) return;
    const verified = this.voteState.verifiedIds;
    let target: number | null = null;
    for (const item of order) {
      if (item.score < threshold) break; // below the cutoff: nothing left above
      if (!verified.has(item.id)) target = item.id;
    }
    if (target != null) {
      this.queueEmptyNotified = false;
      this.mediaState.selectMedia(target);
    } else if (!this.queueEmptyNotified) {
      this.queueEmptyNotified = true;
      this.toast.success({
        message: 'All positives reviewed',
        detail: 'Every item above the cutoff has been verified. Check Stats or Export your results.',
        dedupKey: 'find-queue-empty',
      });
    }
  }

  private queueEmptyNotified = false;

  /** Open the detector-evaluation Stats modal. */
  onStats(): void {
    this.showStats = true;
  }

  /** Open the export modal pre-set to a label filter (good / bad / unverified). */
  onExportRequest(filter: LabelFilter): void {
    this.exportFilter = filter;
    this.showExport = true;
  }

  /**
   * Browse the positive results of this Find run as their own UMAP projection.
   * The positives are the current "good" list (the above-threshold items plus
   * any manual corrections). We stash the ids for the browse view and navigate
   * to `/browse/:datasetId?subset=1`, where they're UMAP'd on their own.
   */
  onBrowse(): void {
    const datasetId = this.activeContext.datasetId;
    if (!datasetId) return;
    const ids = Array.from(this.voteState.goodVotes);
    if (ids.length === 0) return;
    const modelId = this.activeContext.modelId;
    const detectorName =
      this.datasetState.detectors.find((d) => d.id === modelId)?.name || 'Detector';
    this.browseSubset.set({
      datasetId,
      ids,
      label: `${detectorName} — positives`,
    });
    this.router.navigate(['/browse', datasetId], { queryParams: { subset: 1 } });
  }

  /**
   * Promote the current Goods pile into its own saved dataset. The
   * promoted items keep their origins and embeddings; the new dataset
   * gets a fresh created date but inherits this dataset's death date.
   * We prompt for a name (prefilled "<dataset> <detector> Results"),
   * then create + register it and confirm with a toast (staying in Find).
   */
  onToDataset(): void {
    const ids = Array.from(this.voteState.goodVotes);
    if (ids.length === 0) return;
    const modelId = this.activeContext.modelId;
    const detectorName =
      this.datasetState.detectors.find((d) => d.id === modelId)?.name || 'Detector';
    const base = [this.datasetName, detectorName, 'Results'].filter((s) => !!s).join(' ');

    this.dialog.prompt('Name the new dataset', base).then((name) => {
      const trimmed = (name ?? '').trim();
      if (!trimmed) return;
      this.datasetsCrudApi
        .promote(trimmed, ids)
        .pipe(takeUntil(this.destroy$))
        .subscribe({
          next: (res) => {
            this.toast.success({
              message: `Created dataset "${res.name}"`,
              detail: `${res.num_items} item${res.num_items === 1 ? '' : 's'} promoted. Find it in the Datasets dashboard.`,
            });
          },
        });
    });
  }

  /** Cancel the find-label scoring run; the HTTP request will surface
   *  a cancelled status via its progress channel and the finalize() in
   *  runFindLabel() takes care of clearing sortBusy. */
  onSortCancel(): void {
    this.detectorsFindApi.cancelFind().pipe(takeUntil(this.destroy$)).subscribe();
  }

  // --- Panel percentage helpers ---

  private savePanelPx(side: 'left' | 'right'): void {
    if (!this.currentMediaType) return;
    const px = side === 'left' ? this.leftWidth : this.rightWidth;
    const key = side === 'left' ? 'panel_pct_left' : 'panel_pct_right';
    const dict = side === 'left' ? this.panelPxLeftDict : this.panelPxRightDict;
    dict[this.currentMediaType] = px;
    this.settingsState.update({ [key]: { ...dict } }).subscribe();
  }

  private applyPanelPx(mediaType: string): void {
    const layoutWidth = this.layoutRef.nativeElement.getBoundingClientRect().width || 1200;
    const leftPx = this.panelPxLeftDict[mediaType];
    if (leftPx != null) {
      const leftMax = layoutWidth - this.DIVIDER_TOTAL - this.CENTER_MIN - this.rightWidth;
      this.leftWidth = Math.max(this.LEFT_MIN, Math.min(leftMax, leftPx));
      this.layoutRef.nativeElement.style.setProperty('--left-width', `${this.leftWidth}px`);
    }
    const rightPx = this.panelPxRightDict[mediaType];
    if (rightPx != null) {
      const rightMax = layoutWidth - this.DIVIDER_TOTAL - this.CENTER_MIN - this.leftWidth;
      this.rightWidth = Math.max(this.RIGHT_MIN, Math.min(rightMax, rightPx));
      this.layoutRef.nativeElement.style.setProperty('--right-width', `${this.rightWidth}px`);
    }
  }
}
