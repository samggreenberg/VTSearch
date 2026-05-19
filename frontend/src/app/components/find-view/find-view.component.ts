import { Component, OnInit, OnDestroy, ElementRef, ViewChild, NgZone, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subject, Subscription } from 'rxjs';
import { finalize, takeUntil } from 'rxjs/operators';
import { LeftPanelComponent } from '../left-panel/left-panel.component';
import { CenterPanelComponent } from '../center-panel/center-panel.component';
import { RightPanelComponent } from '../right-panel/right-panel.component';
import { MediasApiService } from '../../services/medias-api.service';
import { DetectorsApiService } from '../../services/detectors-api.service';
import { DatasetsApiService } from '../../services/datasets-api.service';
import { ActiveContextService } from '../../services/active-context.service';
import { DatasetStateService } from '../../services/dataset-state.service';
import { MediaStateService } from '../../services/media-state.service';
import { VoteStateService } from '../../services/vote-state.service';
import { SortStateService } from '../../services/sort-state.service';
import { SettingsStateService } from '../../services/settings-state.service';
import { ProgressEventsService } from '../../services/progress-events.service';
import { ProgressEvent } from '../../models/api.models';
import { formatProgressMessage } from '../../utils/format-progress';
import { iconSizeToGoalWidth, snapPanelWidthToGridColumns } from '../../utils/grid-icon-size';

@Component({
  selector: 'vt-find-view',
  standalone: true,
  imports: [CommonModule, LeftPanelComponent, CenterPanelComponent, RightPanelComponent],
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
    private detectorsApi: DetectorsApiService,
    private datasetsApi: DatasetsApiService,
    private ngZone: NgZone,
    private activeContext: ActiveContextService,
    private datasetState: DatasetStateService,
    public mediaState: MediaStateService,
    public voteState: VoteStateService,
    public sortState: SortStateService,
    private settingsState: SettingsStateService,
    private progressEvents: ProgressEventsService,
  ) {}

  ngOnInit(): void {
    this.layoutRef.nativeElement.style.setProperty('--left-width', `${this.leftWidth}px`);
    this.layoutRef.nativeElement.style.setProperty('--right-width', `${this.rightWidth}px`);
    this.mediaState.loadMedias();
    this.voteState.loadVotes();
    this.loadSettings();
    this.datasetsApi.getStatus().pipe(takeUntil(this.destroy$)).subscribe({
      next: (status) => { this.datasetName = status.display_name || ''; },
    });

    // When medias arrive, run the find-label scoring
    this.mediaState.medias$
      .pipe(takeUntil(this.destroy$))
      .subscribe((medias) => {
        if (medias.length > 0) {
          const newType = medias[0].type;
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

    // Run find-label to score and label all medias
    this.runFindLabel();

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
    this.datasetsApi.getStatus().pipe(takeUntil(this.destroy$)).subscribe({
      next: (status) => { this.datasetName = status.display_name || ''; },
    });
    this.runFindLabel();
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

    const datasetId = this.activeContext.datasetId;
    const modelName =
      this.datasetState.detectors.find((d) => d.id === modelId)?.name || 'Detector';
    this.detectorsApi.findLabel({ detector_id: modelId, ...(datasetId ? { dataset_id: datasetId } : {}) })
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
          // Select the item just above the threshold (last item with score >= threshold)
          if (threshold != null && sorted.length > 0) {
            let aboveId: number | null = null;
            for (const item of sorted) {
              if (item.score >= threshold) {
                aboveId = item.id;
              } else {
                break;
              }
            }
            if (aboveId != null) {
              this.mediaState.selectMedia(aboveId);
            }
          }
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

  onHoverVote(event: { id: number; vote: 'good' | 'bad' }): void {
    if (this.sortState.sortBusy) return;
    const m = this.mediaState.getMedia(event.id);
    const name = m?.filename || m?.origin_name || `#${event.id}`;
    this.voteState.recordVote(event.id, event.vote, name);
    this.mediasApi.vote(event.id, event.vote).pipe(takeUntil(this.destroy$)).subscribe({
      next: () => {
        this.onMediaVoted(event);
      },
    });
  }

  onMediaVoted(event: { id: number; vote: 'good' | 'bad' }): void {
    // In find mode: update labels but do NOT re-train or re-sort.
    // Just update the vote state so the right panel and left panel stripe update.
    this.voteState.applyOptimisticVote(event.id, event.vote);
    this.voteState.loadVotes();
  }

  /** Cancel the find-label scoring run; the HTTP request will surface
   *  a cancelled status via its progress channel and the finalize() in
   *  runFindLabel() takes care of clearing sortBusy. */
  onSortCancel(): void {
    this.detectorsApi.cancelFind().pipe(takeUntil(this.destroy$)).subscribe();
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
