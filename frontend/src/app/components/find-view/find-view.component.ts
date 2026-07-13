import { AfterViewInit, ChangeDetectionStrategy, Component, computed, effect, ElementRef, inject, NgZone, OnDestroy, OnInit, signal, ViewChild } from '@angular/core';
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
import { SortStateService, SortedItem } from '../../services/sort-state.service';
import { SortingApiService } from '../../services/sorting-api.service';
import { SettingsStateService } from '../../services/settings-state.service';
import { ProgressEventsService } from '../../services/progress-events.service';
import { BrowseSubsetService } from '../../services/browse-subset.service';
import { ProgressEvent } from '../../models/api.models';
import {
  ProgressBarState,
  formatEta,
  formatProgressMessage,
  progressBarState,
} from '../../utils/format-progress';
import { iconSizeToGoalWidth, snapPanelWidthToGridColumns } from '../../utils/grid-icon-size';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
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
  private mediasApi = inject(MediasApiService);
  private detectorsFindApi = inject(DetectorsFindApiService);
  private datasetsRegistryApi = inject(DatasetsRegistryApiService);
  private datasetsCrudApi = inject(DatasetsCrudApiService);
  private toast = inject(ToastService);
  private dialog = inject(VtDialogService);
  private ngZone = inject(NgZone);
  private activeContext = inject(ActiveContextService);
  private datasetState = inject(DatasetStateService);
  mediaState = inject(MediaStateService);
  voteState = inject(VoteStateService);
  sortState = inject(SortStateService);
  private sortingApi = inject(SortingApiService);
  private settingsState = inject(SettingsStateService);
  private progressEvents = inject(ProgressEventsService);
  private browseSubset = inject(BrowseSubsetService);
  private router = inject(Router);

  @ViewChild('layout', { static: true }) layoutRef!: ElementRef<HTMLElement>;
  @ViewChild(CenterPanelComponent) centerPanel?: CenterPanelComponent;

  // Written from non-bound callbacks (HTTP status subscribe, the settings-mirror
  // effect, the media-type effect) and read in the template, so under zoneless
  // they must be signals — a plain-field write from those contexts would not
  // schedule CD and the view would go stale (zoneless-migration.md, Phase 2.5,
  // Recipe B & F).
  readonly datasetName = signal('');
  readonly gridGoalWidthLeft = signal(80);
  readonly focusModeLeft = signal<'click' | 'hover'>('click');
  readonly focusModeRight = signal<'click' | 'hover'>('click');
  private gridIconSizeLeftDict: Record<string, string> = {};
  private focusModeLeftDict: Record<string, 'click' | 'hover'> = {};
  private focusModeRightDict: Record<string, 'click' | 'hover'> = {};
  private panelPxLeftDict: Record<string, number> = {};
  private panelPxRightDict: Record<string, number> = {};
  private currentMediaType = '';
  leftWidth = 260;
  rightWidth = 300;

  /**
   * The left work queue: the scored ranking with verified items removed. The
   * left panel is the *unverified* pile — once an item is verified it knows
   * its colour and moves to the right, so it drops off the left (and out of
   * the stripe).  A `computed` over the sort ranking + the verified set, so it
   * recomputes only when one of those signals changes (not every change-detection
   * cycle) and the memoised identity keeps the media-list / stripe from
   * rebuilding needlessly.  Fed to both the media-list and the stripe so their
   * index spaces stay aligned for stripe-click navigation.
   */
  readonly unverifiedSortOrder = computed<SortedItem[] | null>(() => {
    const order = this.sortState.sortOrder;
    const verified = this.voteState.verifiedIds;
    return order && verified.size > 0 ? order.filter((item) => !verified.has(item.id)) : order;
  });
  /**
   * Stable empty vote sets handed to the left panel in Find mode: the left
   * shows only unverified items, which carry no colour, so the media-list and
   * stripe must never paint green/red there.  A shared frozen reference keeps
   * the inputs identity-stable across change detection.
   */
  readonly noVotes: Set<number> = new Set();
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

  constructor() {
    effect(() => {
      const settings = this.settingsState.settingsSignal();
      if (!settings) return;
      const sizeDict = settings.grid_icon_size_left;
      if (sizeDict && typeof sizeDict === 'object') {
        this.gridIconSizeLeftDict = sizeDict as Record<string, string>;
        if (this.currentMediaType) {
          this.gridGoalWidthLeft.set(
            iconSizeToGoalWidth(this.gridIconSizeLeftDict[this.currentMediaType] ?? 'M'),
          );
        }
      }
      const fmLeft = settings.focus_mode_left;
      if (fmLeft && typeof fmLeft === 'object') {
        this.focusModeLeftDict = fmLeft as Record<string, 'click' | 'hover'>;
        if (this.currentMediaType) {
          this.focusModeLeft.set(this.focusModeLeftDict[this.currentMediaType] ?? 'click');
        }
      }
      const fmRight = settings.focus_mode_right;
      if (fmRight && typeof fmRight === 'object') {
        this.focusModeRightDict = fmRight as Record<string, 'click' | 'hover'>;
        if (this.currentMediaType) {
          this.focusModeRight.set(this.focusModeRightDict[this.currentMediaType] ?? 'click');
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

    effect(() => {
      const medias = this.mediaState.mediasSignal();
      if (medias.length > 0) {
        const newType = medias[0].media_type;
        if (newType !== this.currentMediaType) {
          this.currentMediaType = newType;
          this.gridGoalWidthLeft.set(iconSizeToGoalWidth(this.gridIconSizeLeftDict[newType] ?? 'M'));
          this.focusModeLeft.set(this.focusModeLeftDict[newType] ?? 'click');
          this.focusModeRight.set(this.focusModeRightDict[newType] ?? 'click');
          this.applyPanelPx(newType);
        }
      }
    });
  }

  ngOnInit(): void {
    this.layoutRef.nativeElement.style.setProperty('--left-width', `${this.leftWidth}px`);
    this.layoutRef.nativeElement.style.setProperty('--right-width', `${this.rightWidth}px`);
    // Find mode: an item's good/bad is the detector's presumption until a
    // human verifies it, so the big buttons read neutral and a click verifies
    // (rather than toggling the presumption off).  Cleared in ngOnDestroy.
    this.voteState.setFindMode(true);
    // Find/Train share singleton sub-view state (SortStateService /
    // VoteStateService), so a fresh Dashboard → Find navigation still holds the
    // previous session's ranking and votes.  Against a *smaller* dataset that
    // stale ranking briefly renders ids that only existed in the prior dataset,
    // firing a storm of image 404s.  Reset it here (mirroring reloadForNewPair)
    // before loading — but NOT when returning from the Browser, where the
    // preserved ranking + verifications are exactly what we want to keep (see
    // the runFindLabel guard below).
    const returningFromBrowse = this.browseSubset.consumeReturningToFind();
    if (!returningFromBrowse) {
      this.sortState.setSortResults([], 0);
      this.sortState.setSortStatus('');
      this.sortState.setSortProgress(0, 0);
      this.voteState.clear();
    }
    this.mediaState.loadMedias();
    this.voteState.loadVotes();
    // The left work queue (ranking minus verified items) is the
    // `unverifiedSortOrder` computed, which tracks sortOrder + verifiedIds.
    this.loadSettings();
    this.datasetsRegistryApi.getStatus().pipe(takeUntil(this.destroy$)).subscribe({
      next: (status) => { this.datasetName.set(status.display_name || ''); },
    });

    // When medias arrive, the media-type sync runs from a constructor effect
    // on `mediaState.mediasSignal()`.

    // Run find-label to score and label all medias — unless we're returning
    // from the Browser after verifying a selection there. In that case the
    // backend vote lists already reflect the verified items (now Verified
    // Good/Bad), and the loadVotes() above refreshes them; re-running find here
    // would re-score with the unchanged model and could re-promote those items,
    // undoing the verification. Keep the verifications instead.
    // Seed the inclusion slider from the active detector's context value
    // (GET /api/inclusion resolves per-detector, falling back to the
    // user-settings default the first time it's read). This keeps Find's
    // slider in step with whatever the detector was last trained at.
    this.seedInclusion();

    if (!returningFromBrowse) {
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
      next: (status) => { this.datasetName.set(status.display_name || ''); },
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
    this.voteState.setFindMode(false);
    this.voteState.stopPolling();
    document.removeEventListener('mousemove', this.boundMouseMove);
    document.removeEventListener('mouseup', this.boundMouseUp);
    document.removeEventListener('mousemove', this.boundRightMouseMove);
    document.removeEventListener('mouseup', this.boundRightMouseUp);
  }

  // --- Find-label scoring ---

  private progressPollSub: Subscription | null = null;

  /** Unified bar state for the scoring overlay: prefers the whole-job
   *  ``overall`` fraction so the bar fills once across all Find phases. */
  get sortBar(): ProgressBarState {
    return progressBarState({
      current: this.sortState.sortProgress,
      total: this.sortState.sortProgressTotal,
      overall: this.sortState.sortOverall,
    });
  }

  /** Overall ETA chip for the scoring overlay (empty when unavailable). */
  get sortEta(): string {
    return formatEta(this.sortState.sortEtaSeconds);
  }

  private startProgressPolling(): void {
    this.stopProgressPolling();
    this.progressPollSub = this.progressEvents.find$
      .pipe(takeUntil(this.destroy$))
      .subscribe((prog: ProgressEvent) => {
        if (prog.status === 'running') {
          this.sortState.setSortStatus(formatProgressMessage(prog, 'Scoring with detector…'));
          this.sortState.setSortProgress(
            prog.current ?? 0,
            prog.total ?? 0,
            prog.overall ?? null,
            prog.eta_seconds ?? null,
          );
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
          // Seed the centre on the marginal positive (lowest item ≥ cutoff):
          // a fresh score restarts the boundary walk on the `above` side, so
          // the seed and the first auto-advance unify on the same item.
          this.queueEmptyNotified = false;
          this.nextFindSide = 'above';
          this.advanceToBoundary();
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
    // `leftWidth` is not template-bound — it only drives the `--left-width` CSS
    // custom property set imperatively here — so no CD is needed and the former
    // `ngZone.run` (a zoneless no-op anyway) is dropped.
    this.leftWidth = newWidth;
    this.layoutRef.nativeElement.style.setProperty('--left-width', `${newWidth}px`);
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
        this.leftWidth = clamped;
        this.layoutRef.nativeElement.style.setProperty('--left-width', `${clamped}px`);
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
    this.rightWidth = newWidth;
    this.layoutRef.nativeElement.style.setProperty('--right-width', `${newWidth}px`);
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
        this.rightWidth = clamped;
        this.layoutRef.nativeElement.style.setProperty('--right-width', `${clamped}px`);
      }
    }
    this.savePanelPx('right');
  }

  // --- Data loading ---

  private loadSettings(): void {
    this.settingsState.load();
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
    // Auto-advance to the next item on the boundary walk, so "just sit and
    // vote" samples both faces of the cutoff instead of only the positives.
    this.advanceToBoundary();
  }

  /**
   * Select the next item to review by walking *outward from the cutoff*,
   * alternating sides: the nearest unverified item above the line, then the
   * nearest below, then the next above, and so on. "Just sit and vote" then
   * samples both faces of the decision boundary — the marginal positives the
   * detector barely accepted and the marginal negatives it barely rejected —
   * rather than only marching up the positive pile (the old marginal-positive
   * order). A fresh score resets {@link nextFindSide} to `'above'`, so the
   * initial seed in {@link runFindLabel} still lands on the marginal positive.
   *
   * The queue is empty only when no unverified item remains on *either* side;
   * that is the done state.
   */
  private advanceToBoundary(): void {
    const order = this.sortState.sortOrder;
    const threshold = this.sortState.threshold;
    if (!order || threshold == null) return;
    const verified = this.voteState.verifiedIds;
    // `order` is descending by score. The unverified item closest above the
    // line is the *lowest* one still ≥ threshold (keep overwriting as we
    // descend); the closest below is the *highest* one < threshold (the first
    // sub-threshold item we hit). One pass finds both.
    let closestAbove: number | null = null;
    let closestBelow: number | null = null;
    for (const item of order) {
      if (verified.has(item.id)) continue;
      if (item.score >= threshold) {
        closestAbove = item.id;
      } else if (closestBelow == null) {
        closestBelow = item.id;
      }
    }
    // Prefer the side it's this turn; fall back to the other side when the
    // preferred one is exhausted so the walk continues until both are empty.
    let target: number | null = null;
    let took: 'above' | 'below' | null = null;
    if (this.nextFindSide === 'above') {
      if (closestAbove != null) {
        [target, took] = [closestAbove, 'above'];
      } else if (closestBelow != null) {
        [target, took] = [closestBelow, 'below'];
      }
    } else {
      if (closestBelow != null) {
        [target, took] = [closestBelow, 'below'];
      } else if (closestAbove != null) {
        [target, took] = [closestAbove, 'above'];
      }
    }
    if (target != null && took != null) {
      // Flip so the next advance samples the opposite face of the boundary.
      this.nextFindSide = took === 'above' ? 'below' : 'above';
      this.queueEmptyNotified = false;
      this.mediaState.selectMedia(target);
    } else if (!this.queueEmptyNotified) {
      this.queueEmptyNotified = true;
      this.toast.success({
        message: 'All items reviewed',
        detail: 'Every item on both sides of the cutoff has been verified. Check Stats or Export your results.',
        dedupKey: 'find-queue-empty',
      });
    }
  }

  /**
   * Which face of the cutoff {@link advanceToBoundary} serves next. Starts on
   * `'above'` (so the seed is the marginal positive) and flips after each pick,
   * alternating above/below as the user votes down the boundary.
   */
  private nextFindSide: 'above' | 'below' = 'above';

  private queueEmptyNotified = false;

  /** Open the detector-evaluation Stats modal. */
  onStats(): void {
    this.showStats = true;
  }

  /**
   * Fold the corrections (items whose adopted label differs from the detector's
   * original call) into the active detector's labelset for future use. The
   * current Find session stays frozen — its scores, queue, votes, and Stats keep
   * showing the detector version that produced them — so the only visible effect
   * is the Stats being flagged out of date. The retrained detector applies the
   * next time the dataset is scored.
   */
  onAddCorrections(): void {
    if (this.sortState.sortBusy) return;
    this.dialog
      .confirmDestructive(
        'Add your corrections to this detector?',
        "Every item you changed from the detector's call is added to its labelset, so the detector learns from them next time you score. " +
          'Your current results and evaluation stay as they are — the Stats will be marked out of date — and nothing is re-scored now.',
        'Add Corrections',
      )
      .then((ok) => {
        if (!ok) return;
        this.detectorsFindApi
          .addCorrectionsToDetector()
          .pipe(takeUntil(this.destroy$))
          .subscribe({
            next: (resp) => {
              if (resp.corrections_added === 0) {
                this.toast.success({
                  message: 'No corrections to add',
                  detail: "Every item still matches the detector's original call.",
                  dedupKey: 'find-corrections-none',
                });
                return;
              }
              this.toast.success({
                message: `Added ${resp.corrections_added} correction${resp.corrections_added === 1 ? '' : 's'} to the detector`,
                detail: `The detector now has ${resp.num_labels} label${resp.num_labels === 1 ? '' : 's'} and will use them next time you score. Your current results stay put; Stats are now marked out of date.`,
                dedupKey: 'find-corrections-added',
              });
            },
            error: (err: { error?: { message?: string; error?: string } }) => {
              const body = err?.error;
              const message = body?.message || body?.error || 'Failed to add corrections';
              this.toast.error({ message, dedupKey: 'find-corrections-error' });
            },
          });
      });
  }

  /** Open the export modal pre-set to a label filter (good / bad / unverified). */
  onExportRequest(filter: LabelFilter): void {
    this.exportFilter = filter;
    this.showExport = true;
  }

  /**
   * Ids of the unverified positives: the above-cutoff items the human hasn't
   * acted on yet. The left work-queue actions (Browse / To Dataset / Export)
   * all scope over exactly this set, as opposed to the right panel's
   * full-good-set actions.
   *
   * Derived from the *frozen scores + current cutoff* (`sortOrder` + `threshold`)
   * rather than the `goodVotes` signal, because those two move **synchronously**
   * when the Inclusion slider does (`onInclusionChange` sets them on the POST
   * response), whereas `goodVotes` only catches up on the follow-up
   * `loadVotes()` GET. Reading `goodVotes` here let a Browse fired right after a
   * slide pick up the *previous* cutoff's positives (the stale-superset bug);
   * scoring against the live cutoff keeps Browse in lock-step with the green
   * line the user sees. Falls back to `goodVotes` only when scores are absent
   * (no scoring pass yet).
   */
  private unverifiedGoodIds(): number[] {
    const verified = this.voteState.verifiedIds;
    const order = this.sortState.sortOrder;
    const threshold = this.sortState.threshold;
    if (order && threshold != null) {
      return order
        .filter((item) => item.score >= threshold && !verified.has(item.id))
        .map((item) => item.id);
    }
    return Array.from(this.voteState.goodVotes).filter((id) => !verified.has(id));
  }

  /**
   * The full positive set of this Find run: every verified-good item (pinned by
   * the human, wherever its score lands) plus the unverified positives (above
   * the live cutoff). The unverified half rides {@link unverifiedGoodIds}, so it
   * tracks the cutoff synchronously and never lags a slide; the verified half is
   * cutoff-independent and read straight off the votes.
   */
  private goodIds(): number[] {
    const verified = this.voteState.verifiedIds;
    const verifiedGood = Array.from(this.voteState.goodVotes).filter((id) => verified.has(id));
    return [...verifiedGood, ...this.unverifiedGoodIds()];
  }

  /**
   * Browse the full positive set of this Find run (verified + unverified good)
   * as their own UMAP projection. Right-panel action; the left panel's Browse
   * scopes to the unverified positives instead ({@link onBrowseUnverified}).
   */
  onBrowse(): void {
    this.browseIds(this.goodIds());
  }

  /**
   * Browse only the unverified positives (above the cutoff, not yet acted on).
   * Verifying a selection in the browse drops it from the canvas — it's no
   * longer unverified. Left-panel work-queue action.
   */
  onBrowseUnverified(): void {
    this.browseIds(this.unverifiedGoodIds());
  }

  /**
   * Stash *ids* for the browse view and navigate to
   * `/browse/:datasetId?subset=1`, where they're UMAP'd on their own.
   */
  private browseIds(ids: number[]): void {
    const datasetId = this.activeContext.datasetId;
    if (!datasetId || ids.length === 0) return;
    this.browseSubset.set({
      datasetId,
      ids,
    });
    this.router.navigate(['/browse', datasetId], { queryParams: { subset: 1 } });
  }

  /**
   * Promote the full Goods pile (verified + unverified) into its own saved
   * dataset. Right-panel action; the left panel promotes the unverified
   * positives instead ({@link onToDatasetUnverified}).
   */
  onToDataset(): void {
    this.toDatasetFromIds(this.goodIds());
  }

  /** Promote only the unverified positives into their own dataset. */
  onToDatasetUnverified(): void {
    this.toDatasetFromIds(this.unverifiedGoodIds());
  }

  /**
   * Promote *ids* into their own saved dataset. The promoted items keep their
   * origins and embeddings; the new dataset gets a fresh created date but
   * inherits this dataset's death date. We prompt for a name (prefilled
   * "<dataset> <detector> Results"), then create + register it and confirm
   * with a toast (staying in Find).
   */
  private toDatasetFromIds(ids: number[]): void {
    if (ids.length === 0) return;
    const modelId = this.activeContext.modelId;
    const detectorName =
      this.datasetState.detectors.find((d) => d.id === modelId)?.name || 'Detector';
    const base = [this.datasetName(), detectorName, 'Results'].filter((s) => !!s).join(' ');

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
