import { Component, OnInit, OnDestroy, ElementRef, ViewChild, NgZone, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subject, timer, Subscription, pairwise } from 'rxjs';
import { takeUntil, switchMap } from 'rxjs/operators';
import { LeftPanelComponent } from '../left-panel/left-panel.component';
import { CenterPanelComponent } from '../center-panel/center-panel.component';
import { RightPanelComponent } from '../right-panel/right-panel.component';
import { SortingApiService } from '../../services/sorting-api.service';
import { DetectorsApiService } from '../../services/detectors-api.service';
import { MediasApiService } from '../../services/medias-api.service';
import { LabelSessionService } from '../../services/label-session.service';
import { MediaStateService } from '../../services/media-state.service';
import { VoteStateService } from '../../services/vote-state.service';
import { SortStateService, SortMode, SelectMode, SortedItem } from '../../services/sort-state.service';
import { SettingsStateService } from '../../services/settings-state.service';
import { AutopilotStateService } from '../../services/autopilot-state.service';
import { ProgressModalComponent, ProgressMetric } from '../modals/progress-modal/progress-modal.component';
import { ResortPromptModalComponent, ResortResult } from '../modals/resort-prompt-modal/resort-prompt-modal.component';
import { LabelingStatusResponse } from '../../models/api.models';
import { iconSizeToGoalWidth } from '../../utils/grid-icon-size';

@Component({
  selector: 'vt-label-view',
  standalone: true,
  imports: [CommonModule, LeftPanelComponent, CenterPanelComponent, RightPanelComponent, ProgressModalComponent, ResortPromptModalComponent],
  templateUrl: './label-view.component.html',
  styleUrl: './label-view.component.scss',
})
export class LabelViewComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('layout', { static: true }) layoutRef!: ElementRef<HTMLElement>;
  @ViewChild(CenterPanelComponent) centerPanel?: CenterPanelComponent;

  labelingStatus: LabelingStatusResponse | null = null;
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
  autopilotCollapsed = false;
  autopilotEnabled = true;
  progressModalMetric: ProgressMetric | null = null;

  // Re-sort prompt state
  showResortPrompt = false;
  resortCurrentType: 'text' | 'media' = 'text';
  resortCurrentDisplay = '';
  private resortInterval = 10;
  private resortVoteCount = 0;
  private resortNextThreshold = 0;

  get nextResortThreshold(): number {
    return Math.round(this.resortNextThreshold * 1.5);
  }

  private readonly COLLAPSED_WIDTH = 48;
  private savedLeftWidth = 260;
  private readonly LEFT_MIN = 180;
  private readonly RIGHT_MIN = 150;
  private readonly CENTER_MIN = 100;
  private readonly DIVIDER_TOTAL = 8; // 2 × 4px dividers
  private destroy$ = new Subject<void>();
  private statusPolling$: Subscription | null = null;
  private scoringProgressPoll$: Subscription | null = null;
  private learnedSortPending = false;
  private autopilotTextSortPending = false;
  private autopilotMediaSortPending = false;
  private dragging = false;
  private draggingRight = false;
  private boundMouseMove = this.onMouseMove.bind(this);
  private boundMouseUp = this.onMouseUp.bind(this);
  private boundRightMouseMove = this.onRightMouseMove.bind(this);
  private boundRightMouseUp = this.onRightMouseUp.bind(this);

  constructor(
    private sortingApi: SortingApiService,
    private detectorsApi: DetectorsApiService,
    private mediasApi: MediasApiService,
    private ngZone: NgZone,
    private labelSession: LabelSessionService,
    public mediaState: MediaStateService,
    public voteState: VoteStateService,
    public sortState: SortStateService,
    private settingsState: SettingsStateService,
    private autopilotStateService: AutopilotStateService,
  ) {}

  ngOnInit(): void {
    this.autopilotStateService.clear();
    this.voteState.clear();
    this.layoutRef.nativeElement.style.setProperty('--left-width', `${this.leftWidth}px`);
    this.layoutRef.nativeElement.style.setProperty('--right-width', `${this.rightWidth}px`);
    this.mediaState.loadMedias();
    this.voteState.loadVotes();
    this.loadSettings();
    this.startStatusPolling();

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
        if (this.autopilotTextSortPending && medias.length > 0) {
          this.autopilotTextSortPending = false;
          this.triggerAutopilotTextSort();
        }
        if (this.autopilotMediaSortPending && medias.length > 0) {
          this.autopilotMediaSortPending = false;
          this.triggerAutopilotMediaSort();
        }
      });

    this.autopilotStateService.state$
      .pipe(pairwise(), takeUntil(this.destroy$))
      .subscribe(([prev, curr]) => {
        if (prev.phase === curr.phase) return;
        if (curr.phase === 'good') this.sortState.setSelectMode('top');
        else if (curr.phase === 'bad') this.sortState.setSelectMode('hard');
        else if (curr.phase === 'hard') {
          this.sortState.setSelectMode('hard');
          this.sortState.setSortMode('learned');
          this.onLearnedSort(false);
        }
        else if (curr.phase === 'new') this.sortState.setSelectMode('new');
      });
  }

  ngAfterViewInit(): void {
    setTimeout(() => this.centerPanel?.init());
  }

  ngOnDestroy(): void {
    this.stopScoringProgressPoll();
    this.destroy$.next();
    this.destroy$.complete();
    this.voteState.stopPolling();
    document.removeEventListener('mousemove', this.boundMouseMove);
    document.removeEventListener('mouseup', this.boundMouseUp);
    document.removeEventListener('mousemove', this.boundRightMouseMove);
    document.removeEventListener('mouseup', this.boundRightMouseUp);
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
    const minWidth = this.autopilotCollapsed ? this.COLLAPSED_WIDTH : this.LEFT_MIN;
    const leftMax = layoutRect.width - this.DIVIDER_TOTAL - this.CENTER_MIN - this.rightWidth;
    newWidth = Math.max(minWidth, Math.min(leftMax, newWidth));
    this.ngZone.run(() => {
      if (this.autopilotCollapsed && newWidth >= this.LEFT_MIN) {
        this.autopilotCollapsed = false;
        this.settingsState.update({ hide_autopilot: false }).subscribe();
      }
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
        if (settings.autopilot_enabled != null) {
          this.autopilotEnabled = settings.autopilot_enabled;
        }
        if (settings.hide_autopilot && !this.autopilotCollapsed) {
          this.setAutopilotCollapsed(true);
        } else if (settings.hide_autopilot === false && this.autopilotCollapsed) {
          this.setAutopilotCollapsed(false);
        }
        if (settings.inclusion != null) {
          this.sortState.setInclusion(settings.inclusion);
        }
        if (settings.autopilot_resort_interval != null) {
          this.resortInterval = settings.autopilot_resort_interval;
          // Initialize the threshold if not yet set
          if (this.resortNextThreshold === 0) {
            this.resortNextThreshold = this.resortInterval;
          }
        }
      });
  }

  private startStatusPolling(): void {
    this.statusPolling$ = timer(0, 2000)
      .pipe(
        takeUntil(this.destroy$),
        switchMap(() => this.sortingApi.getLabelingStatus()),
      )
      .subscribe({
        next: (status) => {
          this.labelingStatus = status;
        },
      });
  }

  // --- Sort handlers ---

  onSortModeChange(mode: SortMode): void {
    this.sortState.setSortMode(mode);
    this.autoSelectNext();
  }

  onTextSort(text: string): void {
    this.sortState.setTextQuery(text);
    this.sortState.setSortBusy(true);
    this.sortState.setSortStatus('Sorting...');
    this.sortingApi.sort({ text }).pipe(takeUntil(this.destroy$)).subscribe({
      next: (response) => {
        this.sortState.setSortResults(
          response.results.map((r) => ({ id: r.id, score: r.similarity })),
          response.threshold,
        );
        this.sortState.setSortBusy(false);
        this.sortState.setSortStatus('');
        this.autoSelectNext();
      },
      error: () => {
        this.sortState.setSortBusy(false);
        this.sortState.setSortStatus('Sort failed');
      },
    });
  }

  onLearnedSort(autoSelect = true): void {
    if (this.voteState.goodVotes.size === 0 || this.voteState.badVotes.size === 0) return;
    this.sortState.setSortBusy(true);
    this.sortState.setSortStatus('Training...');
    this.sortingApi.learnedSort().pipe(takeUntil(this.destroy$)).subscribe({
      next: (response) => {
        this.sortState.setSortResults(
          response.results.map((r) => ({ id: r.id, score: r.score })),
          response.threshold,
        );
        this.sortState.setSortBusy(false);
        this.sortState.setSortStatus('');
        if (autoSelect) {
          this.autoSelectNext();
        }
      },
      error: () => {
        this.sortState.setSortBusy(false);
        this.sortState.setSortStatus('Training failed');
      },
    });
  }

  onLoadSort(): void {
    // Re-sort using existing load sort results when switching back to load mode
  }

  private startScoringProgressPoll(): void {
    this.stopScoringProgressPoll();
    this.scoringProgressPoll$ = timer(200, 500)
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

  private stopScoringProgressPoll(): void {
    if (this.scoringProgressPoll$) {
      this.scoringProgressPoll$.unsubscribe();
      this.scoringProgressPoll$ = null;
    }
  }

  onDetectorLoaded(data: unknown): void {
    const detector = data as Record<string, unknown>;
    const name = (detector['name'] as string) || 'Detector';
    this.sortState.setSortMode('load');
    this.sortState.setSortBusy(true);
    this.sortState.setSortStatus('Scoring with detector…');
    this.sortState.setSortProgress(0, 0);

    this.startScoringProgressPoll();

    this.detectorsApi.detectorSort({ detector }).pipe(takeUntil(this.destroy$)).subscribe({
      next: (response) => {
        this.stopScoringProgressPoll();
        this.sortState.setSortResults(
          response.results.map((r) => ({ id: r.id, score: r.score })),
          response.threshold,
        );
        this.sortState.setLoadSortLabel(name);
        this.sortState.setSortBusy(false);
        this.sortState.setSortStatus('');
        this.sortState.setSortProgress(0, 0);
        this.autoSelectNext();
      },
      error: () => {
        this.stopScoringProgressPoll();
        this.sortState.setSortBusy(false);
        this.sortState.setSortStatus('Detector sort failed');
        this.sortState.setSortProgress(0, 0);
      },
    });
  }

  onExampleSortStarted(data: unknown): void {
    const response = data as { results: { id: number; similarity: number }[]; threshold: number };
    this.sortState.setSortMode('load');
    this.sortState.setSortResults(
      response.results.map((r) => ({ id: r.id, score: r.similarity })),
      response.threshold,
    );
    this.sortState.setLoadSortLabel('Example media');
    this.sortState.setSortBusy(false);
    this.sortState.setSortStatus('');
    this.autoSelectNext();
  }

  // --- Select mode ---

  onSelectModeChange(mode: SelectMode): void {
    this.sortState.setSelectMode(mode);
    this.autoSelectNext();
  }

  private fetchDiversityNext(): void {
    const sortOrder = this.sortState.sortOrder;
    const scores = sortOrder
      ? Object.fromEntries(sortOrder.map((s) => [String(s.id), s.score]))
      : undefined;
    this.sortingApi
      .getDiversityTreeNext(scores, this.sortState.threshold ?? undefined)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          if (response.id !== null) {
            this.mediaState.selectMedia(response.id);
          }
        },
      });
  }

  // --- Inclusion ---

  onInclusionChange(value: number): void {
    this.sortState.setInclusion(value);
    this.sortingApi.setInclusion(value).pipe(takeUntil(this.destroy$)).subscribe();
    this.autoSelectNext();
    if (this.sortState.sortMode === 'learned' && this.voteState.goodVotes.size > 0 && this.voteState.badVotes.size > 0) {
      this.scheduleLearnedSort(false);
    }
  }

  private scheduleLearnedSort(autoSelect = true): void {
    if (this.learnedSortPending) return;
    this.learnedSortPending = true;
    setTimeout(() => {
      this.learnedSortPending = false;
      this.onLearnedSort(autoSelect);
    }, 300);
  }

  // --- Media selection ---

  onMediaSelect(id: number): void {
    this.mediaState.selectMedia(id);
  }

  onHoverVote(event: { id: number; vote: 'good' | 'bad' }): void {
    this.mediasApi.vote(event.id, event.vote).pipe(takeUntil(this.destroy$)).subscribe({
      next: () => {
        this.onMediaVoted(event);
      },
    });
  }

  onMediaVoted(event: { id: number; vote: 'good' | 'bad' }): void {
    this.voteState.applyOptimisticVote(event.id, event.vote);
    this.voteState.loadVotes();
    this.autoSelectNext(event.id);
    if (this.sortState.sortMode === 'learned' && this.voteState.goodVotes.size > 0 && this.voteState.badVotes.size > 0) {
      this.scheduleLearnedSort(false);
    }
    this.checkResortPrompt();
  }

  // --- Indicators ---

  onIndicatorClick(name: string): void {
    const metricMap: Record<string, ProgressMetric> = {
      smart: 'smart',
      stable: 'stable',
      span: 'diverse',
    };
    const metric = metricMap[name];
    if (metric) {
      this.progressModalMetric = metric;
    }
  }

  onProgressModalClosed(): void {
    this.progressModalMetric = null;
  }

  // --- Autopilot ---

  onAutopilotStart(): void {
    // Initialize re-sort tracking
    this.resortVoteCount = 0;
    this.resortNextThreshold = this.resortInterval;

    const phase = this.autopilotStateService.state.phase;

    // For phases beyond 'good', the phase-transition subscription already set
    // the correct selectMode and (for 'hard') triggered a learned sort.
    // Only override selectMode for the initial 'good' phase.
    if (phase === 'good') {
      this.sortState.setSelectMode('top');
    }

    // For 'hard' and later phases the subscription already triggered learned
    // sort; no text/media sort needed.  For 'good' and 'bad' phases, kick off
    // the text/media sort so the user has results to vote on.
    if (phase === 'good' || phase === 'bad') {
      const textQuery = this.labelSession.textQuery;
      const mediaExample = this.labelSession.mediaExample;
      if (textQuery) {
        if (this.mediaState.medias.length > 0) {
          this.triggerAutopilotTextSort();
        } else {
          this.autopilotTextSortPending = true;
        }
      } else if (mediaExample) {
        if (this.mediaState.medias.length > 0) {
          this.triggerAutopilotMediaSort();
        } else {
          this.autopilotMediaSortPending = true;
        }
      } else {
        // No sort query configured; try to select from existing sort results.
        this.autoSelectNext();
      }
    }
  }

  private triggerAutopilotTextSort(): void {
    const textQuery = this.labelSession.textQuery;
    if (textQuery) {
      this.onTextSort(textQuery);
    }
  }

  private triggerAutopilotMediaSort(): void {
    const mediaExample = this.labelSession.mediaExample;
    if (mediaExample) {
      this.sortState.setSortBusy(true);
      this.sortState.setSortStatus('Sorting by example...');
      this.sortingApi.exampleSortServer({ filename: mediaExample }).pipe(takeUntil(this.destroy$)).subscribe({
        next: (response) => {
          this.sortState.setSortResults(
            response.results.map((r) => ({ id: r.id, score: r.similarity })),
            response.threshold,
          );
          this.sortState.setSortBusy(false);
          this.sortState.setSortStatus('');
          this.sortState.setSortMode('load');
          this.autoSelectNext();
        },
        error: () => {
          this.sortState.setSortBusy(false);
          this.sortState.setSortStatus('Example sort failed');
        },
      });
    }
  }

  // --- Re-sort prompt ---

  private checkResortPrompt(): void {
    // Only show during autopilot's "good" phase (sorting by example in top mode)
    if (!this.autopilotStateService.running) return;
    // Eagerly check phase transition so we don't show the prompt after the user
    // has already found enough greens (the panel's ngOnChanges may not have run yet).
    this.autopilotStateService.checkPhaseTransition(
      this.voteState.goodVotes.size, this.voteState.badVotes.size,
    );
    const phase = this.autopilotStateService.state.phase;
    if (phase !== 'good') return;

    this.resortVoteCount++;
    if (this.resortVoteCount >= this.resortNextThreshold) {
      // Determine current example info for the prompt
      if (this.labelSession.textQuery) {
        this.resortCurrentType = 'text';
        this.resortCurrentDisplay = this.labelSession.textQuery;
      } else if (this.labelSession.mediaExample) {
        this.resortCurrentType = 'media';
        this.resortCurrentDisplay = this.labelSession.mediaExample;
      } else {
        return; // No example to prompt about
      }
      this.showResortPrompt = true;
    }
  }

  onResortKeep(): void {
    this.showResortPrompt = false;
    // Multiply threshold by 1.5 for next prompt
    this.resortNextThreshold = Math.round(this.resortNextThreshold * 1.5);
    this.resortVoteCount = 0;
  }

  onResortNewExample(result: ResortResult): void {
    this.showResortPrompt = false;
    this.resortVoteCount = 0;
    // Reset threshold back to the base interval
    this.resortNextThreshold = this.resortInterval;

    if (result.type === 'text') {
      this.labelSession.textQuery = result.value;
      this.labelSession.mediaExample = '';
      this.sortState.setSelectMode('top');
      this.triggerAutopilotTextSort();
    } else {
      this.labelSession.mediaExample = result.value;
      this.labelSession.textQuery = '';
      this.sortState.setSelectMode('top');
      this.triggerAutopilotMediaSort();
    }
  }

  onResortClosed(): void {
    // Treat closing the modal as "keep"
    this.onResortKeep();
  }

  onAutopilotToggleCollapse(): void {
    const newVal = !this.autopilotCollapsed;
    this.setAutopilotCollapsed(newVal);
    this.settingsState.update({ hide_autopilot: newVal }).subscribe();
  }

  private setAutopilotCollapsed(collapsed: boolean): void {
    this.autopilotCollapsed = collapsed;
    if (collapsed) {
      this.savedLeftWidth = this.leftWidth;
      this.leftWidth = this.COLLAPSED_WIDTH;
    } else {
      this.leftWidth = this.savedLeftWidth;
    }
    this.layoutRef.nativeElement.style.setProperty('--left-width', `${this.leftWidth}px`);
  }

  onAutopilotEnabledChange(enabled: boolean): void {
    this.autopilotEnabled = enabled;
    this.settingsState.update({ autopilot_enabled: enabled }).subscribe();
  }

  onAutopilotStop(): void {
    const phase = this.autopilotStateService.state.phase;
    const isMediaBased = !!this.labelSession.mediaExample && !this.labelSession.textQuery;

    // Map autopilot phase to the same Sort + Select that autopilot was using.
    if (phase === 'good') {
      this.sortState.setSortMode(isMediaBased ? 'load' : 'text');
      this.sortState.setSelectMode('top');
    } else if (phase === 'bad') {
      this.sortState.setSortMode(isMediaBased ? 'load' : 'text');
      this.sortState.setSelectMode('hard');
    } else if (phase === 'hard') {
      this.sortState.setSortMode('learned');
      this.sortState.setSelectMode('hard');
    } else if (phase === 'new' || phase === 'done') {
      this.sortState.setSortMode('learned');
      this.sortState.setSelectMode('new');
    }

    // Deactivate autopilot state so resort prompt and phase logic stop firing.
    this.autopilotStateService.deactivate();
    this.showResortPrompt = false;
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
    if (leftPx != null && !this.autopilotCollapsed) {
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

  // --- Helpers ---

  private autoSelectNext(excludeId?: number): void {
    const sortOrder = this.sortState.sortOrder;
    if (!sortOrder || sortOrder.length === 0) return;
    const goodVotes = this.voteState.goodVotes;
    const badVotes = this.voteState.badVotes;

    const isVoted = (id: number): boolean =>
      id === excludeId || goodVotes.has(id) || badVotes.has(id);

    if (this.sortState.selectMode === 'top') {
      const next = sortOrder.find((s) => !isVoted(s.id));
      if (next) this.mediaState.selectMedia(next.id);
    } else if (this.sortState.selectMode === 'bottom') {
      for (let i = sortOrder.length - 1; i >= 0; i--) {
        const s = sortOrder[i];
        if (!isVoted(s.id)) {
          this.mediaState.selectMedia(s.id);
          break;
        }
      }
    } else if (this.sortState.selectMode === 'hard' && this.sortState.threshold !== null) {
      let best: SortedItem | null = null;
      let bestDist = Infinity;
      for (const s of sortOrder) {
        if (isVoted(s.id)) continue;
        const dist = Math.abs(s.score - this.sortState.threshold!);
        if (dist < bestDist) {
          bestDist = dist;
          best = s;
        }
      }
      if (best) this.mediaState.selectMedia(best.id);
    } else if (this.sortState.selectMode === 'new') {
      this.fetchDiversityNext();
    }
  }
}
