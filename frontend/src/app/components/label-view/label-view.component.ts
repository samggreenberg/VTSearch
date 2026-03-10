import { Component, OnInit, OnDestroy, ElementRef, ViewChild, NgZone, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subject, timer, Subscription, pairwise } from 'rxjs';
import { takeUntil, switchMap } from 'rxjs/operators';
import { LeftPanelComponent } from '../left-panel/left-panel.component';
import { CenterPanelComponent } from '../center-panel/center-panel.component';
import { RightPanelComponent } from '../right-panel/right-panel.component';
import { SortingApiService } from '../../services/sorting-api.service';
import { LabelSessionService } from '../../services/label-session.service';
import { MediaStateService } from '../../services/media-state.service';
import { VoteStateService } from '../../services/vote-state.service';
import { SortStateService, SortMode, SelectMode, SortedItem } from '../../services/sort-state.service';
import { SettingsStateService } from '../../services/settings-state.service';
import { AutopilotStateService } from '../../services/autopilot-state.service';
import { ProgressModalComponent, ProgressMetric } from '../modals/progress-modal/progress-modal.component';
import { LabelingStatusResponse } from '../../models/api.models';

@Component({
  selector: 'vt-label-view',
  standalone: true,
  imports: [CommonModule, LeftPanelComponent, CenterPanelComponent, RightPanelComponent, ProgressModalComponent],
  templateUrl: './label-view.component.html',
  styleUrl: './label-view.component.scss',
})
export class LabelViewComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('layout', { static: true }) layoutRef!: ElementRef<HTMLElement>;
  @ViewChild(CenterPanelComponent) centerPanel?: CenterPanelComponent;

  labelingStatus: LabelingStatusResponse | null = null;
  showThumbnails = true;
  leftWidth = 260;
  rightWidth = 300;
  autopilotCollapsed = false;
  progressModalMetric: ProgressMetric | null = null;

  private readonly COLLAPSED_WIDTH = 36;
  private savedLeftWidth = 260;
  private readonly LEFT_MIN = 180;
  private readonly LEFT_MAX = 500;
  private readonly RIGHT_MIN = 150;
  private readonly RIGHT_MAX = 500;
  private destroy$ = new Subject<void>();
  private statusPolling$: Subscription | null = null;
  private learnedSortPending = false;
  private autopilotTextSortPending = false;
  private dragging = false;
  private draggingRight = false;
  private boundMouseMove = this.onMouseMove.bind(this);
  private boundMouseUp = this.onMouseUp.bind(this);
  private boundRightMouseMove = this.onRightMouseMove.bind(this);
  private boundRightMouseUp = this.onRightMouseUp.bind(this);

  constructor(
    private sortingApi: SortingApiService,
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
    this.layoutRef.nativeElement.style.setProperty('--left-width', `${this.leftWidth}px`);
    this.layoutRef.nativeElement.style.setProperty('--right-width', `${this.rightWidth}px`);
    this.mediaState.loadMedias();
    this.voteState.loadVotes();
    this.loadSettings();
    this.startStatusPolling();

    this.mediaState.medias$
      .pipe(takeUntil(this.destroy$))
      .subscribe((medias) => {
        if (this.autopilotTextSortPending && medias.length > 0) {
          this.autopilotTextSortPending = false;
          this.triggerAutopilotTextSort();
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
          this.onLearnedSort();
        }
        else if (curr.phase === 'new') this.sortState.setSelectMode('new');
      });
  }

  ngAfterViewInit(): void {
    setTimeout(() => this.centerPanel?.init());
  }

  ngOnDestroy(): void {
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
    newWidth = Math.max(this.LEFT_MIN, Math.min(this.LEFT_MAX, newWidth));
    this.ngZone.run(() => {
      this.leftWidth = newWidth;
      this.layoutRef.nativeElement.style.setProperty('--left-width', `${newWidth}px`);
    });
  }

  private onMouseUp(): void {
    this.dragging = false;
    document.removeEventListener('mousemove', this.boundMouseMove);
    document.removeEventListener('mouseup', this.boundMouseUp);
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
    newWidth = Math.max(this.RIGHT_MIN, Math.min(this.RIGHT_MAX, newWidth));
    this.ngZone.run(() => {
      this.rightWidth = newWidth;
      this.layoutRef.nativeElement.style.setProperty('--right-width', `${newWidth}px`);
    });
  }

  private onRightMouseUp(): void {
    this.draggingRight = false;
    document.removeEventListener('mousemove', this.boundRightMouseMove);
    document.removeEventListener('mouseup', this.boundRightMouseUp);
  }

  // --- Data loading ---

  private loadSettings(): void {
    this.settingsState.load();
    this.settingsState.settings$
      .pipe(takeUntil(this.destroy$))
      .subscribe((settings) => {
        if (!settings) return;
        this.showThumbnails = settings.show_thumbnails_left !== false;
        if (settings.hide_autopilot && !this.autopilotCollapsed) {
          this.setAutopilotCollapsed(true);
        } else if (settings.hide_autopilot === false && this.autopilotCollapsed) {
          this.setAutopilotCollapsed(false);
        }
        if (settings.inclusion != null) {
          this.sortState.setInclusion(settings.inclusion);
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
    // Will be fully wired in Phase 7 (modals for loading detectors/examples)
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

  onMediaVoted(event: { id: number; vote: 'good' | 'bad' }): void {
    this.voteState.loadVotes();
    this.autoSelectNext(event.id);
    if (this.sortState.sortMode === 'learned' && this.voteState.goodVotes.size > 0 && this.voteState.badVotes.size > 0) {
      this.scheduleLearnedSort(false);
    }
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
    this.sortState.setSelectMode('top');
    const textQuery = this.labelSession.textQuery;
    if (textQuery) {
      if (this.mediaState.medias.length > 0) {
        this.triggerAutopilotTextSort();
      } else {
        this.autopilotTextSortPending = true;
      }
    }
  }

  private triggerAutopilotTextSort(): void {
    const textQuery = this.labelSession.textQuery;
    if (textQuery) {
      this.onTextSort(textQuery);
    }
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

  onAutopilotStop(): void {
    const phase = this.autopilotStateService.state.phase;

    // Map autopilot phase to the same Sort + Select that autopilot was using.
    if (phase === 'good') {
      this.sortState.setSortMode('text');
      this.sortState.setSelectMode('top');
    } else if (phase === 'bad') {
      this.sortState.setSortMode('text');
      this.sortState.setSelectMode('hard');
    } else if (phase === 'hard') {
      this.sortState.setSortMode('learned');
      this.sortState.setSelectMode('hard');
    } else if (phase === 'new' || phase === 'done') {
      this.sortState.setSortMode('learned');
      this.sortState.setSelectMode('new');
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
