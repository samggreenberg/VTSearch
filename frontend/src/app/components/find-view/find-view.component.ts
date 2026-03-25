import { Component, OnInit, OnDestroy, ElementRef, ViewChild, NgZone, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subject, Subscription, timer } from 'rxjs';
import { switchMap, takeUntil } from 'rxjs/operators';
import { LeftPanelComponent } from '../left-panel/left-panel.component';
import { CenterPanelComponent } from '../center-panel/center-panel.component';
import { RightPanelComponent } from '../right-panel/right-panel.component';
import { MediasApiService } from '../../services/medias-api.service';
import { DetectorsApiService } from '../../services/detectors-api.service';
import { DatasetsApiService } from '../../services/datasets-api.service';
import { FindSessionService } from '../../services/find-session.service';
import { MediaStateService } from '../../services/media-state.service';
import { VoteStateService } from '../../services/vote-state.service';
import { SortStateService } from '../../services/sort-state.service';
import { SettingsStateService } from '../../services/settings-state.service';
import { iconSizeToGoalWidth } from '../../utils/grid-icon-size';

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
  private readonly DIVIDER_TOTAL = 8; // 2 × 4px dividers
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
    private findSession: FindSessionService,
    public mediaState: MediaStateService,
    public voteState: VoteStateService,
    public sortState: SortStateService,
    private settingsState: SettingsStateService,
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
    this.progressPollSub = timer(200, 500)
      .pipe(
        switchMap(() => this.detectorsApi.getFindProgress()),
        takeUntil(this.destroy$),
      )
      .subscribe((prog: any) => {
        if (prog.status === 'running') {
          const msg = prog.message || 'Scoring with detector…';
          const current = prog.current || 0;
          const total = prog.total || 0;
          this.sortState.setSortStatus(msg);
          this.sortState.setSortProgress(current, total);
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
    const modelId = this.findSession.modelId;
    if (!modelId) return;

    this.sortState.setSortBusy(true);
    this.sortState.setSortStatus('Scoring with detector…');
    this.sortState.setSortProgress(0, 0);

    // Start polling for progress concurrently
    this.startProgressPolling();

    const datasetId = this.findSession.datasetId;
    this.detectorsApi.findLabel({ model_id: modelId, ...(datasetId ? { dataset_id: datasetId } : {}) })
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response: any) => {
          this.stopProgressPolling();
          const sorted = response.results.map((r: any) => ({ id: r.id, score: r.score }));
          const threshold = response.threshold;
          // Set sort results for stripe display
          this.sortState.setSortResults(sorted, threshold);
          this.sortState.setLoadSortLabel(this.findSession.modelName || 'Detector');
          this.sortState.setSortBusy(false);
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
          this.stopProgressPolling();
          this.sortState.setSortBusy(false);
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
