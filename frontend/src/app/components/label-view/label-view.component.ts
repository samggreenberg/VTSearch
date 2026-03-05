import { Component, OnInit, OnDestroy, ElementRef, ViewChild, NgZone, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subject, timer, Subscription } from 'rxjs';
import { takeUntil, switchMap } from 'rxjs/operators';
import { LeftPanelComponent, SortMode, SelectMode, SortedItem } from '../left-panel/left-panel.component';
import { CenterPanelComponent } from '../center-panel/center-panel.component';
import { RightPanelComponent } from '../right-panel/right-panel.component';
import { MediasApiService } from '../../services/medias-api.service';
import { SortingApiService } from '../../services/sorting-api.service';
import { SettingsApiService } from '../../services/settings-api.service';
import { MediaItem, LabelingStatusResponse, AppSettings } from '../../models/api.models';
import { LabelSessionService } from '../../services/label-session.service';

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

  medias: MediaItem[] = [];
  sortOrder: SortedItem[] | null = null;
  threshold: number | null = null;
  selectedId: number | null = null;
  goodVotes = new Set<number>();
  badVotes = new Set<number>();
  sortMode: SortMode = 'text';
  selectMode: SelectMode = 'top';
  inclusion = 0;
  sortBusy = false;
  sortStatus = '';
  labelingStatus: LabelingStatusResponse | null = null;
  showThumbnails = true;
  loadSortLabel = '';
  settings: AppSettings | null = null;
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
    private mediasApi: MediasApiService,
    private sortingApi: SortingApiService,
    private settingsApi: SettingsApiService,
    private ngZone: NgZone,
    private labelSession: LabelSessionService,
  ) {}

  ngOnInit(): void {
    this.layoutRef.nativeElement.style.setProperty('--left-width', `${this.leftWidth}px`);
    this.loadMedias();
    this.loadVotes();
    this.loadSettings();
    this.startStatusPolling();
  }

  ngAfterViewInit(): void {
    setTimeout(() => this.centerPanel?.init());
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
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

  private loadMedias(): void {
    this.mediasApi.getMedias().pipe(takeUntil(this.destroy$)).subscribe({
      next: (medias) => {
        this.medias = medias;
        if (this.autopilotTextSortPending) {
          this.autopilotTextSortPending = false;
          this.triggerAutopilotTextSort();
        }
      },
    });
  }

  private loadVotes(): void {
    this.sortingApi.getVotes().pipe(takeUntil(this.destroy$)).subscribe({
      next: (votes) => {
        this.goodVotes = new Set(votes.good);
        this.badVotes = new Set(votes.bad);
      },
    });
  }

  private loadSettings(): void {
    this.settingsApi.getSettings().pipe(takeUntil(this.destroy$)).subscribe({
      next: (settings) => {
        this.settings = settings;
        this.showThumbnails = settings.show_thumbnails_left !== false;
        if (settings.inclusion != null) {
          this.inclusion = settings.inclusion;
        }
      },
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
    this.sortMode = mode;
  }

  onTextSort(text: string): void {
    this.sortBusy = true;
    this.sortStatus = 'Sorting...';
    this.sortingApi.sort({ text }).pipe(takeUntil(this.destroy$)).subscribe({
      next: (response) => {
        this.sortOrder = response.results.map((r) => ({ id: r.id, score: r.similarity }));
        this.threshold = response.threshold;
        this.sortBusy = false;
        this.sortStatus = '';
        this.autoSelectNext();
      },
      error: () => {
        this.sortBusy = false;
        this.sortStatus = 'Sort failed';
      },
    });
  }

  onLearnedSort(): void {
    if (this.goodVotes.size === 0 || this.badVotes.size === 0) return;
    this.sortBusy = true;
    this.sortStatus = 'Training...';
    this.sortingApi.learnedSort().pipe(takeUntil(this.destroy$)).subscribe({
      next: (response) => {
        this.sortOrder = response.results.map((r) => ({ id: r.id, score: r.score }));
        this.threshold = response.threshold;
        this.sortBusy = false;
        this.sortStatus = '';
        this.autoSelectNext();
      },
      error: () => {
        this.sortBusy = false;
        this.sortStatus = 'Training failed';
      },
    });
  }

  onLoadSort(): void {
    // Will be fully wired in Phase 7 (modals for loading detectors/examples)
  }

  // --- Select mode ---

  onSelectModeChange(mode: SelectMode): void {
    this.selectMode = mode;
    if (mode === 'new') {
      this.fetchDiversityNext();
    }
  }

  private fetchDiversityNext(): void {
    const scores = this.sortOrder
      ? Object.fromEntries(this.sortOrder.map((s) => [String(s.id), s.score]))
      : undefined;
    this.sortingApi
      .getDiversityTreeNext(scores, this.threshold ?? undefined)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (response) => {
          if (response.id !== null) {
            this.selectedId = response.id;
          }
        },
      });
  }

  // --- Inclusion ---

  onInclusionChange(value: number): void {
    this.inclusion = value;
    this.sortingApi.setInclusion(value).pipe(takeUntil(this.destroy$)).subscribe();
    if (this.sortMode === 'learned' && this.goodVotes.size > 0 && this.badVotes.size > 0) {
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

  get selectedMedia(): MediaItem | null {
    if (this.selectedId === null) return null;
    return this.medias.find((m) => m.id === this.selectedId) ?? null;
  }

  onMediaSelect(id: number): void {
    this.selectedId = id;
  }

  onMediaVoted(event: { id: number; vote: 'good' | 'bad' }): void {
    // Refresh votes from backend (center panel already did the API call)
    this.loadVotes();
    // Auto-advance to next media
    this.autoSelectNext();
    // Kick off background learned sort if in learned mode
    if (this.sortMode === 'learned' && this.goodVotes.size > 0 && this.badVotes.size > 0) {
      this.scheduleLearnedSort();
    }
  }

  // --- Indicators ---

  onIndicatorClick(_name: string): void {
    // Will open progress modal in Phase 7
  }

  // --- Autopilot ---

  onAutopilotStart(): void {
    this.selectMode = 'top';
    const textQuery = this.labelSession.textQuery;
    if (textQuery) {
      if (this.medias.length > 0) {
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
    if (!this.sortOrder || this.sortOrder.length === 0) return;

    if (this.selectMode === 'top') {
      const next = this.sortOrder.find(
        (s) => !this.goodVotes.has(s.id) && !this.badVotes.has(s.id),
      );
      if (next) this.selectedId = next.id;
    } else if (this.selectMode === 'hard' && this.threshold !== null) {
      let best: SortedItem | null = null;
      let bestDist = Infinity;
      for (const s of this.sortOrder) {
        if (this.goodVotes.has(s.id) || this.badVotes.has(s.id)) continue;
        const dist = Math.abs(s.score - this.threshold);
        if (dist < bestDist) {
          bestDist = dist;
          best = s;
        }
      }
      if (best) this.selectedId = best.id;
    } else if (this.selectMode === 'new') {
      this.fetchDiversityNext();
    }
  }
}
