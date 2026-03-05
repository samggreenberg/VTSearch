import { Component, OnInit, OnDestroy, ElementRef, ViewChild, NgZone, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subject, timer, Subscription } from 'rxjs';
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
import { LabelingStatusResponse } from '../../models/api.models';

@Component({
  selector: 'vt-label-view',
  standalone: true,
  imports: [CommonModule, LeftPanelComponent, CenterPanelComponent, RightPanelComponent],
  templateUrl: './label-view.component.html',
  styleUrl: './label-view.component.scss',
})
export class LabelViewComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('layout', { static: true }) layoutRef!: ElementRef<HTMLElement>;
  @ViewChild(CenterPanelComponent) centerPanel?: CenterPanelComponent;

  labelingStatus: LabelingStatusResponse | null = null;
  showThumbnails = true;
  leftWidth = 260;

  private readonly LEFT_MIN = 180;
  private readonly LEFT_MAX = 500;
  private destroy$ = new Subject<void>();
  private statusPolling$: Subscription | null = null;
  private learnedSortPending = false;
  private autopilotTextSortPending = false;
  private dragging = false;
  private boundMouseMove = this.onMouseMove.bind(this);
  private boundMouseUp = this.onMouseUp.bind(this);

  constructor(
    private sortingApi: SortingApiService,
    private ngZone: NgZone,
    private labelSession: LabelSessionService,
    public mediaState: MediaStateService,
    public voteState: VoteStateService,
    public sortState: SortStateService,
    private settingsState: SettingsStateService,
  ) {}

  ngOnInit(): void {
    this.layoutRef.nativeElement.style.setProperty('--left-width', `${this.leftWidth}px`);
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

  // --- Data loading ---

  private loadSettings(): void {
    this.settingsState.load();
    this.settingsState.settings$
      .pipe(takeUntil(this.destroy$))
      .subscribe((settings) => {
        if (!settings) return;
        this.showThumbnails = settings.show_thumbnails_left !== false;
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
  }

  onTextSort(text: string): void {
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

  onLearnedSort(): void {
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
        this.autoSelectNext();
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
    if (mode === 'new') {
      this.fetchDiversityNext();
    }
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
    if (this.sortState.sortMode === 'learned' && this.voteState.goodVotes.size > 0 && this.voteState.badVotes.size > 0) {
      this.scheduleLearnedSort();
    }
  }

  private scheduleLearnedSort(): void {
    if (this.learnedSortPending) return;
    this.learnedSortPending = true;
    setTimeout(() => {
      this.learnedSortPending = false;
      this.onLearnedSort();
    }, 300);
  }

  // --- Media selection ---

  onMediaSelect(id: number): void {
    this.mediaState.selectMedia(id);
  }

  onMediaVoted(event: { id: number; vote: 'good' | 'bad' }): void {
    this.voteState.loadVotes();
    this.autoSelectNext();
    if (this.sortState.sortMode === 'learned' && this.voteState.goodVotes.size > 0 && this.voteState.badVotes.size > 0) {
      this.scheduleLearnedSort();
    }
  }

  // --- Indicators ---

  onIndicatorClick(_name: string): void {
    // Will open progress modal in Phase 7
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

  onAutopilotStop(): void {
    // Reset to manual defaults
  }

  // --- Helpers ---

  private autoSelectNext(): void {
    const sortOrder = this.sortState.sortOrder;
    if (!sortOrder || sortOrder.length === 0) return;
    const goodVotes = this.voteState.goodVotes;
    const badVotes = this.voteState.badVotes;

    if (this.sortState.selectMode === 'top') {
      const next = sortOrder.find(
        (s) => !goodVotes.has(s.id) && !badVotes.has(s.id),
      );
      if (next) this.mediaState.selectMedia(next.id);
    } else if (this.sortState.selectMode === 'hard' && this.sortState.threshold !== null) {
      let best: SortedItem | null = null;
      let bestDist = Infinity;
      for (const s of sortOrder) {
        if (goodVotes.has(s.id) || badVotes.has(s.id)) continue;
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
