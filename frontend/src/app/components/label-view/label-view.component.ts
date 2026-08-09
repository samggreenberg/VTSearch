import { AfterViewInit, ChangeDetectionStrategy, Component, effect, ElementRef, inject, OnDestroy, OnInit, signal, untracked, viewChild } from '@angular/core';

import { Subject, timer, Subscription, pairwise } from 'rxjs';
import { takeUntil, switchMap, filter, take } from 'rxjs/operators';
import { LeftPanelComponent } from '../left-panel/left-panel.component';
import { CenterPanelComponent } from '../center-panel/center-panel.component';
import { RightPanelComponent } from '../right-panel/right-panel.component';
import {
  ContextMenuComponent,
  ContextMenuItem,
} from '../context-menu/context-menu.component';
import {
  MediaCropModalComponent,
  MediaCropResult,
} from '../modals/media-crop-modal/media-crop-modal.component';
import { NewThingFlowsService } from '../../services/new-thing-flows.service';
import { ToastService } from '../../services/toast.service';
import { SortingApiService } from '../../services/sorting-api.service';
import { adaptivePoll } from '../../services/adaptive-poll';
import { DetectorsFindApiService } from '../../services/detectors-find-api.service';
import { DetectorsRegistryApiService } from '../../services/detectors-registry-api.service';
import { MediasApiService } from '../../services/medias-api.service';
import { DatasetsRegistryApiService } from '../../services/datasets-registry-api.service';
import { LabelSessionService } from '../../services/label-session.service';
import { MediaStateService } from '../../services/media-state.service';
import { VoteStateService } from '../../services/vote-state.service';
import { LabelsetStateService } from '../../services/labelset-state.service';
import { SortStateService, SortMode, SelectMode, SortedItem } from '../../services/sort-state.service';
import { SettingsStateService } from '../../services/settings-state.service';
import { AutopilotStateService } from '../../services/autopilot-state.service';
import { EmbedderCapabilityService } from '../../services/embedder-capability.service';
import { ActiveContextService } from '../../services/active-context.service';
import { ProgressEventsService } from '../../services/progress-events.service';
import { ProgressEvent } from '../../models/api.models';
import { DetectorRegistryEntry } from '../../generated/api-client/models/detector-registry-entry';
import { ProgressModalComponent, ProgressMetric } from '../modals/progress-modal/progress-modal.component';
import { ResortPromptModalComponent, ResortResult } from '../modals/resort-prompt-modal/resort-prompt-modal.component';
import type { LabelingStatusResponse } from '../../generated/api-client/models/labeling-status-response';
import type { LearnedSortResponse } from '../../generated/api-client/models/learned-sort-response';
import { formatProgressMessage } from '../../utils/format-progress';
import { snapPanelWidthToGridColumns, iconSizeToGoalWidth } from '../../utils/grid-icon-size';
import { PanelResizeDirective } from './panel-resize.directive';
import { LabelViewPanelStateService } from './label-view-panel-state.service';
import { buildMediaContextMenuItems } from './media-context-menu-items';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-label-view',
  standalone: true,
  imports: [
    LeftPanelComponent,
    CenterPanelComponent,
    RightPanelComponent,
    ProgressModalComponent,
    ResortPromptModalComponent,
    ContextMenuComponent,
    MediaCropModalComponent,
    PanelResizeDirective
],
  providers: [LabelViewPanelStateService],
  templateUrl: './label-view.component.html',
  styleUrl: './label-view.component.scss',
})
export class LabelViewComponent implements OnInit, AfterViewInit, OnDestroy {
  private sortingApi = inject(SortingApiService);
  private detectorsFindApi = inject(DetectorsFindApiService);
  private detectorsRegistryApi = inject(DetectorsRegistryApiService);
  private mediasApi = inject(MediasApiService);
  private datasetsRegistryApi = inject(DatasetsRegistryApiService);
  private labelSession = inject(LabelSessionService);
  mediaState = inject(MediaStateService);
  voteState = inject(VoteStateService);
  private labelsetState = inject(LabelsetStateService);
  sortState = inject(SortStateService);
  private settingsState = inject(SettingsStateService);
  private autopilotStateService = inject(AutopilotStateService);
  private embedderCaps = inject(EmbedderCapabilityService);
  private activeContext = inject(ActiveContextService);
  private progressEvents = inject(ProgressEventsService);
  private newThingFlows = inject(NewThingFlowsService);
  private toast = inject(ToastService);
  panelState = inject(LabelViewPanelStateService);

  readonly layoutRef = viewChild.required<ElementRef<HTMLElement>>('layout');
  readonly centerPanel = viewChild(CenterPanelComponent);

  readonly datasetName = signal('');
  /** Name of the trainable model owning the labels shown on the right pane.
   *  Empty when no trainable model is active; the right pane then falls
   *  back to cid-based vote display. */
  readonly trainableModelName = signal<string | null>(null);
  readonly labelingStatus = signal<LabelingStatusResponse | null>(null);
  readonly leftWidth = signal(260);
  readonly rightWidth = signal(300);
  readonly autopilotCollapsed = signal(false);
  readonly autopilotEnabled = signal(true);
  /** True when autopilot has reached its terminal "exhausted" state — every
   *  item in a tiny dataset is labeled but the indicators never went green.
   *  Drives the center-pane "nothing left to label" message so the pane is
   *  never left blank with a stale item stuck in the metadata strip. */
  readonly autopilotExhausted = signal(false);
  /** True while a windowed-sort "Load more" page fetch is in flight. */
  readonly loadingMoreSort = signal(false);
  progressModalMetric: ProgressMetric | null = null;

  // SortStateService / VoteStateService are now signal-backed (their value
  // getters read private signals), so the template binds those getters directly
  // — `sortState.sortBusy`, `voteState.goodVotes`, … — and they repaint under
  // zoneless when the state changes from async callbacks. The per-consumer
  // `toSignal` bridges this component used to carry are gone (Phase 2.5).

  // Per-media-type panel preferences (grid size, focus mode, saved widths) live
  // on `panelState`; the template reads getters on it directly.
  get gridGoalWidthLeft(): number { return this.panelState.gridGoalWidthLeft; }
  get focusModeLeft(): 'click' | 'hover' { return this.panelState.focusModeLeft; }
  get focusModeRight(): 'click' | 'hover' { return this.panelState.focusModeRight; }

  // Re-sort prompt state
  readonly showResortPrompt = signal(false);
  resortCurrentType: 'text' | 'media' = 'text';
  resortCurrentDisplay = '';
  private resortInterval = 10;
  private resortVoteCount = 0;
  private resortNextThreshold = 0;

  get nextResortThreshold(): number {
    return Math.round(this.resortNextThreshold * 1.5);
  }

  readonly COLLAPSED_WIDTH = 48;
  private savedLeftWidth = 260;

  // --- Auto-pop after an icon-size change ---
  // Resizing the divider pops the panel tight to the grid columns on release.
  // Changing icon size reflows the grid but leaves the panel at its old snapped
  // width, so a gap remains. We re-pop tight once the user has *settled* on a
  // size: a debounce coalesces a flurry of size bumps ("up, up, no — down") into
  // a single pop, instead of lunging narrower on every click.
  /** ms of no icon-size changes before a settled panel auto-pops tight. */
  private readonly AUTO_POP_DELAY = 700;
  /** Class-removal timeout for the animated pop; ≥ the SCSS transition so it
   *  always completes before live dragging goes instant again. */
  private readonly AUTO_POP_ANIM_MS = 220;
  private leftPopTimer: ReturnType<typeof setTimeout> | null = null;
  private rightPopTimer: ReturnType<typeof setTimeout> | null = null;
  private animatePopTimer: ReturnType<typeof setTimeout> | null = null;
  /** Last grid goal width observed per side, used to detect icon-size changes.
   *  Null until a baseline is captured for the active media type. */
  private lastGoalWidthLeft: number | null = null;
  private lastGoalWidthRight: number | null = null;
  /** Media type the goal-width baselines above belong to. A change here means a
   *  media-type switch (not a user resize), so we re-baseline without popping. */
  private autoPopMediaType = '';
  readonly LEFT_MIN = 180;
  readonly RIGHT_MIN = 150;
  readonly CENTER_MIN = 100;
  readonly DIVIDER_TOTAL = 16; // 2 × 8px dividers
  private destroy$ = new Subject<void>();
  /**
   * Fires whenever the active (dataset, detector) pair changes — and on
   * destroy. Every request whose response writes *pair-scoped* state (the sort
   * window, the inclusion slider, the dataset name, the selected media) is
   * piped through `takeUntil(this.pairScope$)` rather than `destroy$`, so the
   * work started for the pair we're leaving is torn down the instant the pair
   * switches.
   *
   * Without it, a detector-scoring POST or a learned-sort job poll outlives the
   * switch and calls `applySortWindow` into whatever pair happens to be active
   * when it finally settles — installing the previous pair's ranking and
   * threshold, then auto-selecting an id that may not exist in the new dataset.
   * `takeUntil` also aborts the stale request client-side. NOTE: subscriptions
   * started from `modelId$` (which emits *before* `pair$`) must stay on
   * `destroy$`, or `reloadForNewPair`'s teardown would kill the request they
   * just issued for the new pair.
   */
  private pairScope$ = new Subject<void>();
  private statusPolling$: Subscription | null = null;
  private scoringProgressPoll$: Subscription | null = null;
  private learnedSortPending = false;
  /** Active learned-sort job id while a training run is in flight. Set in
   *  ``onLearnedSort`` once the backend returns a job id, cleared in
   *  ``applyLearnedSortResult`` / the error/cancel paths. Used by the
   *  Cancel button on the sort progress bar to target the right job. */
  private currentLearnedSortJobId: string | null = null;
  /** Set by `reloadForNewPair` when the user was in `learned` sort mode at the
   *  time of a pair switch: a constructor effect watches the labelset counts and
   *  re-fires `onLearnedSort` once, after the reloaded votes make both classes
   *  available. Replaces the old one-shot `labelsetGoodCount$` subscription now
   *  that VoteStateService is signal-backed. */
  private pendingRehydrateLearned = false;
  private autopilotTextSortPending = false;
  private autopilotMediaSortPending = false;
  /** Armed on entry and on each pair reload; consumed once medias first render
   *  to snap both panels tight to the grid (see ``snapPanelsOnLoad``). */
  private pendingSnapOnLoad = false;

  constructor() {
    effect(() => {
      const settings = this.settingsState.settingsSignal();
      if (!settings) return;
      // Settings is the only intended dependency. The body reads and writes the
      // panel-width / autopilot-collapsed signals (directly and via
      // `applyPanelPx`/`setAutopilotCollapsed`); without `untracked` those reads
      // would make the effect depend on signals it also writes — an infinite
      // loop, plus spurious re-runs that revert a manual collapse toggle while
      // the settings write is still in flight.
      untracked(() => {
        this.panelState.loadFromSettings(settings);
        if (this.panelState.currentMediaType) {
          this.applyPanelPx();
          // Detect an icon-size change (same media type) and debounce a re-pop.
          // Skipped when the media type just switched: the media effect owns
          // re-baselining in that case (a switch must never auto-pop).
          if (this.panelState.currentMediaType === this.autoPopMediaType) {
            this.maybeScheduleAutoPop('left', this.panelState.gridGoalWidthLeft);
            this.maybeScheduleAutoPop('right', this.currentRightGoalWidth());
          }
        }
        if (settings.autopilot_enabled != null) {
          this.autopilotEnabled.set(settings.autopilot_enabled);
        }
        if (settings.hide_autopilot && !this.autopilotCollapsed()) {
          this.setAutopilotCollapsed(true);
        } else if (settings.hide_autopilot === false && this.autopilotCollapsed()) {
          this.setAutopilotCollapsed(false);
        }
        if (settings.autopilot_resort_interval != null) {
          this.resortInterval = settings.autopilot_resort_interval;
          // Initialize the threshold if not yet set
          if (this.resortNextThreshold === 0) {
            this.resortNextThreshold = this.resortInterval;
          }
        }
      });
    });

    effect(() => {
      const medias = this.mediaState.mediasSignal();
      // Track the embedder registry too, so the text-support check inside
      // `triggerAutopilotTextSort` is reliable (and we never fire a doomed text
      // sort on a no-text dataset); reading `infos()` here makes this effect
      // re-run once the registry resolves. Everything else runs `untracked` so
      // `applyPanelPx`'s width-signal reads/writes don't loop the effect.
      const infos = this.embedderCaps.infos();
      untracked(() => {
        if (medias.length > 0) {
          const newType = medias[0].media_type;
          if (newType !== this.panelState.currentMediaType) {
            this.panelState.setMediaType(newType);
            this.applyPanelPx();
            // Re-baseline the auto-pop tracking for the new media type so the
            // legitimate goal-width change a switch carries doesn't pop, and the
            // first real size change afterward does.
            this.captureAutoPopBaseline();
          }
          // First content of a fresh entry/reload just rendered: snap both
          // panels tight to the grid so the restored width doesn't leave a gap.
          if (this.pendingSnapOnLoad) {
            this.pendingSnapOnLoad = false;
            this.snapPanelsOnLoad();
          }
        }
        if (this.autopilotTextSortPending && medias.length > 0 && infos !== null) {
          this.autopilotTextSortPending = false;
          this.triggerAutopilotTextSort();
        }
        if (this.autopilotMediaSortPending && medias.length > 0) {
          this.autopilotMediaSortPending = false;
          this.triggerAutopilotMediaSort();
        }
      });
    });

    // Phase-3 learned-sort rehydration after a pair switch. `reloadForNewPair`
    // clears votes (counts → 0) then reloads them; when the reloaded counts make
    // both classes available again and the user was in `learned` mode, fire one
    // `onLearnedSort`. Tracking both labelset counts re-runs this effect when
    // `loadVotes` lands; the body runs `untracked` so reading
    // `learnedSortAvailable` (which also reads the counts) can't loop it.
    effect(() => {
      this.voteState.labelsetGoodCount;
      this.voteState.labelsetBadCount;
      untracked(() => {
        if (!this.pendingRehydrateLearned) return;
        if (this.sortState.sortMode === 'learned' && this.voteState.learnedSortAvailable) {
          this.pendingRehydrateLearned = false;
          this.onLearnedSort(false);
        }
      });
    });
  }

  ngOnInit(): void {
    this.autopilotStateService.clear();
    this.embedderCaps.ensureLoaded();
    this.voteState.clear();
    this.layoutRef().nativeElement.style.setProperty('--left-width', `${this.leftWidth()}px`);
    this.layoutRef().nativeElement.style.setProperty('--right-width', `${this.rightWidth()}px`);
    this.pendingSnapOnLoad = true;
    this.mediaState.loadMedias();
    this.voteState.loadVotes();
    this.loadSettings();
    this.startStatusPolling();
    this.datasetsRegistryApi.getStatus().pipe(takeUntil(this.pairScope$)).subscribe({
      next: (status) => { this.datasetName.set(status.display_name || ''); },
    });

    this.activeContext.modelId$
      .pipe(takeUntil(this.destroy$))
      .subscribe((modelId) => this.refreshTrainableModelName(modelId));
    this.refreshTrainableModelName(this.activeContext.modelId);
    this.seedInclusion();

    // Reload data when the active pair changes via the top-bar switcher.
    // Skip the first emission; `ngOnInit` above already triggered the
    // initial loads.
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

    this.autopilotStateService.state$
      .pipe(pairwise(), takeUntil(this.destroy$))
      .subscribe(([prev, curr]) => {
        if (prev.phase === curr.phase) return;
        this.autopilotExhausted.set(curr.phase === 'exhausted');
        if (curr.phase === 'good') {
          this.sortState.setSelectMode('top');
          if (curr.retrainMode) {
            this.sortState.setSortMode('learned');
            this.onLearnedSort(false);
          }
        }
        else if (curr.phase === 'bad') {
          this.sortState.setSelectMode('hard');
          if (curr.retrainMode) {
            this.sortState.setSortMode('learned');
            this.onLearnedSort(false);
          }
        }
        else if (curr.phase === 'hard') {
          this.sortState.setSelectMode('hard');
          this.sortState.setSortMode('learned');
          this.onLearnedSort(false);
        }
        else if (curr.phase === 'new') this.sortState.setSelectMode('new');
      });
  }

  ngAfterViewInit(): void {
    setTimeout(() => this.centerPanel()?.init());
  }

  /** Triggered by the top-bar context switcher whenever the active
   *  (dataset, detector) pair changes. Resets the view-local ephemeral
   *  state (sort results, votes cache) and re-runs the same loads that
   *  ngOnInit fires on first entry.
   *
   *  Phase 3 rehydration: if the user's sort mode is `learned` and the
   *  reloaded labelset has both classes, fire one `onLearnedSort` call
   *  after votes land. The server's signature cache short-circuits the
   *  re-fire when the pair has been trained recently (free re-entry),
   *  and starts a fresh job otherwise; either way the user lands on
   *  learned-sorted content without a manual mode toggle. */
  private reloadForNewPair(): void {
    // Supersede the pair we're leaving *first*: every in-flight request scoped
    // to it dies here, before any of the new pair's state is installed, so no
    // late scoring/learned-sort response can `applySortWindow` into the new
    // context. Those subscriptions carry no `finalize`, so the busy flag and
    // the scoring progress feed they own are reset explicitly below.
    this.pairScope$.next();
    this.stopScoringProgressPoll();
    this.currentLearnedSortJobId = null;
    this.sortState.setSortBusy(false);
    this.pendingRehydrateLearned = false;
    this.sortState.setSortResults([], 0);
    this.sortState.setSortStatus('');
    this.sortState.setSortProgress(0, 0);
    this.voteState.clear();
    this.pendingSnapOnLoad = true;
    this.mediaState.loadMedias();
    this.voteState.loadVotes();
    this.datasetsRegistryApi.getStatus().pipe(takeUntil(this.pairScope$)).subscribe({
      next: (status) => { this.datasetName.set(status.display_name || ''); },
    });
    // Re-seed the slider for the detector we just switched to.
    this.seedInclusion();

    // Arm the rehydrate effect: it fires `onLearnedSort` once the reloaded
    // votes land (counts go 0 → available) if the user is still in learned mode.
    this.pendingRehydrateLearned = this.sortState.sortMode === 'learned';
  }

  ngOnDestroy(): void {
    this.stopScoringProgressPoll();
    this.cancelAutoPop('left');
    this.cancelAutoPop('right');
    this.cancelSnapOnLoad();
    if (this.animatePopTimer) clearTimeout(this.animatePopTimer);
    this.pairScope$.next();
    this.pairScope$.complete();
    this.destroy$.next();
    this.destroy$.complete();
    this.voteState.stopPolling();
  }

  // --- Divider drag ---

  /** Min width the left panel can shrink to right now (autopilot-collapsed
   *  state lets the user drag down to a thin sliver). */
  get leftMin(): number {
    return this.autopilotCollapsed() ? this.COLLAPSED_WIDTH : this.LEFT_MIN;
  }

  onLeftWidthChange(width: number): void {
    // Grabbing the divider supersedes any pending icon-size auto-pop.
    this.cancelAutoPop('left');
    if (this.autopilotCollapsed() && width >= this.LEFT_MIN) {
      this.autopilotCollapsed.set(false);
      this.settingsState.update({ hide_autopilot: false }).subscribe();
    }
    this.leftWidth.set(width);
    this.layoutRef().nativeElement.style.setProperty('--left-width', `${width}px`);
  }

  onLeftResizeEnd(width: number): void {
    this.cancelAutoPop('left');
    this.leftWidth.set(width);
    this.popPanelTight('left');
  }

  onRightWidthChange(width: number): void {
    this.cancelAutoPop('right');
    this.rightWidth.set(width);
    this.layoutRef().nativeElement.style.setProperty('--right-width', `${width}px`);
  }

  onRightResizeEnd(width: number): void {
    this.cancelAutoPop('right');
    this.rightWidth.set(width);
    this.popPanelTight('right');
  }

  /** Snap one panel down to the minimum width that still shows its current grid
   *  column count, then clamp and persist. Shared by the divider drag-release
   *  handlers, the icon-size auto-pop, and the on-load snap. The drag/auto-pop
   *  callers animate the snap so every pop looks the same; the on-load caller
   *  passes `animate = false` so the panel simply appears tight instead of
   *  visibly shrinking as the view opens. No-op for the snap step when the panel
   *  isn't in grid mode; the width is still persisted so drag-release always
   *  records where the user left the divider. */
  private popPanelTight(side: 'left' | 'right', animate = true): void {
    const selector = side === 'left' ? 'vt-left-panel' : 'vt-right-panel';
    const panelEl = this.layoutRef().nativeElement.querySelector(selector) as HTMLElement | null;
    const currentWidth = side === 'left' ? this.leftWidth() : this.rightWidth();
    const snapped = panelEl ? snapPanelWidthToGridColumns(panelEl, currentWidth) : null;
    if (snapped !== null) {
      const layoutWidth = this.layoutRef().nativeElement.getBoundingClientRect().width;
      const otherWidth = side === 'left' ? this.rightWidth() : this.leftWidth();
      const max = layoutWidth - this.DIVIDER_TOTAL - this.CENTER_MIN - otherWidth;
      const min = side === 'left' ? this.leftMin : this.RIGHT_MIN;
      const clamped = Math.max(min, Math.min(max, snapped));
      if (clamped !== currentWidth) {
        if (animate) this.animatePop();
        const widthSignal = side === 'left' ? this.leftWidth : this.rightWidth;
        widthSignal.set(clamped);
        this.layoutRef().nativeElement.style.setProperty(`--${side}-width`, `${clamped}px`);
      }
    }
    this.panelState.savePanelPx(side, side === 'left' ? this.leftWidth() : this.rightWidth());
  }

  // --- Snap on load ---

  /** Pending animation-frame ids for the on-load snap poll, one per side, so
   *  ``ngOnDestroy`` can cancel a poll still waiting for the grid to lay out. */
  private snapLoadFrames: Record<'left' | 'right', number | null> = { left: null, right: null };

  /** Snap both panels tight to their grid columns once, after a fresh content
   *  load. On open, ``applyPanelPx`` restores the saved width verbatim — which
   *  may sit wider than the grid needs, leaving a gap between the last thumbnail
   *  column and the panel edge. The divider-drag handlers snap on release, so
   *  the gap "heals" the instant the user touches the divider; this performs the
   *  same snap at load so the user never sees the gap in the first place. */
  private snapPanelsOnLoad(): void {
    this.snapWhenGridReady('left');
    this.snapWhenGridReady('right');
  }

  /** Snap `side` tight once its grid has laid out. ``snapPanelWidthToGridColumns``
   *  needs the panel at its applied width with a real ``--grid-goal-width`` and a
   *  nonzero client width; on first open those land a frame or two after the
   *  medias arrive, so poll a bounded number of animation frames until it can
   *  read a column count, then snap without animating. */
  private snapWhenGridReady(side: 'left' | 'right', attempt = 0): void {
    const MAX_ATTEMPTS = 60;
    const selector = side === 'left' ? 'vt-left-panel' : 'vt-right-panel';
    const panelEl = this.layoutRef().nativeElement.querySelector(selector) as HTMLElement | null;
    const currentWidth = side === 'left' ? this.leftWidth() : this.rightWidth();
    const ready = panelEl != null && snapPanelWidthToGridColumns(panelEl, currentWidth) !== null;
    if (ready) {
      this.snapLoadFrames[side] = null;
      this.popPanelTight(side, false);
      return;
    }
    if (attempt >= MAX_ATTEMPTS) {
      this.snapLoadFrames[side] = null;
      return;
    }
    this.snapLoadFrames[side] = requestAnimationFrame(() => this.snapWhenGridReady(side, attempt + 1));
  }

  private cancelSnapOnLoad(): void {
    for (const side of ['left', 'right'] as const) {
      const frame = this.snapLoadFrames[side];
      if (frame !== null) cancelAnimationFrame(frame);
      this.snapLoadFrames[side] = null;
    }
  }

  // --- Icon-size auto-pop ---

  /** Right-pane grid goal width for the active media type, read from the live
   *  settings (the right pane owns its own size dict, unlike the left which is
   *  mirrored on `panelState`). */
  private currentRightGoalWidth(): number {
    const settings = this.settingsState.settingsSignal();
    const dict = (settings?.grid_icon_size_right ?? null) as Record<string, string> | null;
    return iconSizeToGoalWidth(dict?.[this.panelState.currentMediaType] ?? 'M');
  }

  /** Record the current goal widths as the no-pop baseline for the active media
   *  type, and drop any pending pops. Called on a media-type switch so the
   *  goal-width change a switch carries is absorbed without popping. */
  private captureAutoPopBaseline(): void {
    this.autoPopMediaType = this.panelState.currentMediaType;
    this.lastGoalWidthLeft = this.panelState.gridGoalWidthLeft;
    this.lastGoalWidthRight = this.currentRightGoalWidth();
    this.cancelAutoPop('left');
    this.cancelAutoPop('right');
  }

  /** Compare the new goal width against the baseline for `side`; if it changed
   *  (a real icon-size bump, not the first observation), debounce a re-pop. */
  private maybeScheduleAutoPop(side: 'left' | 'right', goalWidth: number): void {
    const last = side === 'left' ? this.lastGoalWidthLeft : this.lastGoalWidthRight;
    if (side === 'left') this.lastGoalWidthLeft = goalWidth;
    else this.lastGoalWidthRight = goalWidth;
    if (last === null || last === goalWidth) return;
    this.cancelAutoPop(side);
    const timer = setTimeout(() => {
      if (side === 'left') this.leftPopTimer = null;
      else this.rightPopTimer = null;
      this.popPanelTight(side);
    }, this.AUTO_POP_DELAY);
    if (side === 'left') this.leftPopTimer = timer;
    else this.rightPopTimer = timer;
  }

  private cancelAutoPop(side: 'left' | 'right'): void {
    const timer = side === 'left' ? this.leftPopTimer : this.rightPopTimer;
    if (timer) clearTimeout(timer);
    if (side === 'left') this.leftPopTimer = null;
    else this.rightPopTimer = null;
  }

  /** Enable the grid-template-columns transition for one auto-pop, then strip it
   *  so live divider dragging stays instant. */
  private animatePop(): void {
    const el = this.layoutRef().nativeElement;
    el.classList.add('layout--animate-pop');
    if (this.animatePopTimer) clearTimeout(this.animatePopTimer);
    this.animatePopTimer = setTimeout(() => {
      el.classList.remove('layout--animate-pop');
      this.animatePopTimer = null;
    }, this.AUTO_POP_ANIM_MS);
  }

  // --- Data loading ---

  private loadSettings(): void {
    this.settingsState.load();
  }

  private startStatusPolling(): void {
    // adaptivePoll never overlaps GETs (a backend slower than the interval no
    // longer has every /api/labeling-status read cancelled by the next tick,
    // which used to freeze this panel permanently), eases to a heartbeat once
    // the status stops changing, and pauses while the tab is hidden. Poll
    // errors are absorbed inside adaptivePoll, so a transient failure skips a
    // single tick instead of tearing the poll down for the view's lifetime.
    this.statusPolling$ = adaptivePoll(() => this.sortingApi.getLabelingStatus(), {
      fastMs: 2000,
      slowMs: 10000,
    })
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (status) => {
          this.labelingStatus.set(status);
        },
      });
  }

  // --- Sort handlers ---

  onSortModeChange(mode: SortMode): void {
    this.sortState.setSortMode(mode);
    this.autoSelectNext();
  }

  /**
   * Install a (possibly windowed) sort response into the sort state. Handles
   * both `similarity` (text/example sort) and `score` (learned/detector sort)
   * result rows, and carries the window metadata (`total` / `has_more_below` /
   * `sort_token`) so the media-list can page deeper. Below the backend's window
   * threshold the whole ranking arrives and `has_more_below` is false —
   * behaviour is identical to the pre-windowing full-list path.
   */
  private applySortWindow(response: {
    results?: Array<Record<string, unknown>>;
    threshold?: number;
    acq_threshold?: number | null;
    total?: number;
    above_threshold?: number;
    has_more_below?: boolean;
    sort_token?: string;
  }): void {
    const threshold = response.threshold ?? 0;
    const items = (response.results ?? []).map((r) => ({
      id: r['id'] as number,
      score: (r['score'] ?? r['similarity'] ?? 0) as number,
      bestRegion: r['best_region'] as number[] | undefined,
    }));
    this.sortState.setSortWindow({
      items,
      threshold,
      acqThreshold: response.acq_threshold ?? null,
      total: response.total ?? items.length,
      hasMore: response.has_more_below ?? false,
      token: response.sort_token ?? null,
      aboveThreshold: response.above_threshold ?? items.filter((i) => i.score >= threshold).length,
    });
  }

  /**
   * Page in the next window of a windowed ranking (the media-list "Load more"
   * trigger). Fetches from the sort token at the current loaded offset and
   * appends. A failed/expired token just stops paging (the user can re-sort).
   */
  onLoadMore(): void {
    const token = this.sortState.sortToken;
    if (!token || !this.sortState.sortHasMore || this.loadingMoreSort()) return;
    this.loadingMoreSort.set(true);
    const offset = this.sortState.sortOrder?.length ?? 0;
    this.sortingApi
      .getSortPage(token, offset, 200)
      .pipe(takeUntil(this.pairScope$))
      .subscribe({
        next: (page) => {
          const items = (page.results ?? []).map((r) => ({
            id: r['id'] as number,
            score: (r['score'] ?? r['similarity'] ?? 0) as number,
            bestRegion: r['best_region'] as number[] | undefined,
          }));
          this.sortState.appendSortItems(items, page.has_more);
          this.loadingMoreSort.set(false);
        },
        error: () => this.loadingMoreSort.set(false),
      });
  }

  onTextSort(text: string): void {
    this.sortState.setTextQuery(text);
    this.sortState.setSortBusy(true);
    this.sortState.setSortStatus('Sorting…');
    this.sortingApi.sort({ text }).pipe(takeUntil(this.pairScope$)).subscribe({
      next: (response) => {
        this.applySortWindow(response);
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
    if (!this.voteState.learnedSortAvailable) return;
    this.sortState.setSortBusy(true);
    this.sortState.setSortStatus('Training…');
    this.sortingApi.learnedSort().pipe(takeUntil(this.pairScope$)).subscribe({
      next: (response) => {
        if (response.status === 'done') {
          this.applyLearnedSortResult(response, autoSelect);
        } else if (response.status === 'running') {
          this.currentLearnedSortJobId = response.job_id;
          this.pollLearnedSortJob(response.job_id, autoSelect);
        } else {
          this.sortState.setSortBusy(false);
          this.sortState.setSortStatus(response.error || 'Training failed');
        }
      },
      error: () => {
        this.sortState.setSortBusy(false);
        this.sortState.setSortStatus('Training failed');
      },
    });
  }

  private pollLearnedSortJob(jobId: string, autoSelect: boolean): void {
    timer(200, 500)
      .pipe(
        // Pair-scoped: a training job can outlive the pair it was started for,
        // and its result must not be applied to whatever pair is active when it
        // finally settles (see `pairScope$`).
        takeUntil(this.pairScope$),
        switchMap(() => this.sortingApi.getLearnedSortResult(jobId)),
        filter((res) => res.status !== 'running'),
        take(1),
      )
      .subscribe({
        next: (res) => {
          if (res.status === 'done') {
            this.applyLearnedSortResult(res, autoSelect);
          } else if (res.status === 'cancelled') {
            this.currentLearnedSortJobId = null;
            this.sortState.setSortBusy(false);
            this.sortState.setSortStatus('Cancelled');
          } else {
            this.currentLearnedSortJobId = null;
            this.sortState.setSortBusy(false);
            this.sortState.setSortStatus(res.error || 'Training failed');
          }
        },
        error: () => {
          this.currentLearnedSortJobId = null;
          this.sortState.setSortBusy(false);
          this.sortState.setSortStatus('Training failed');
        },
      });
  }

  private applyLearnedSortResult(response: LearnedSortResponse, autoSelect: boolean): void {
    this.applySortWindow(response);
    this.currentLearnedSortJobId = null;
    this.sortState.setSortBusy(false);
    this.sortState.setSortStatus('');
    if (autoSelect) {
      this.autoSelectNext();
    }
  }

  /** Cancel whatever sort run is currently in flight.
   *
   *  - Learned sort: targets the active ``AsyncJob`` by id.
   *  - Load-sort (find-label): trips the shared ``find_progress`` cancel
   *    flag, which the scoring loop polls.
   *  - Text / example sort: no cancellation endpoint; those calls run
   *    synchronously and complete before the user can usefully cancel.
   */
  onSortCancel(): void {
    if (this.currentLearnedSortJobId) {
      const jobId = this.currentLearnedSortJobId;
      this.currentLearnedSortJobId = null;
      this.sortingApi.cancelLearnedSort(jobId).pipe(takeUntil(this.destroy$)).subscribe();
      return;
    }
    if (this.sortState.sortMode === 'load') {
      this.detectorsFindApi.cancelFind().pipe(takeUntil(this.destroy$)).subscribe();
    }
  }

  onLoadSort(): void {
    // Re-sort using existing load sort results when switching back to load mode
  }

  private startScoringProgressPoll(): void {
    this.stopScoringProgressPoll();
    this.scoringProgressPoll$ = this.progressEvents.find$
      .pipe(takeUntil(this.destroy$))
      .subscribe((prog: ProgressEvent) => {
        if (prog.status === 'running') {
          this.sortState.setSortStatus(formatProgressMessage(prog, 'Scoring with detector…'));
          this.sortState.setSortProgress(
            prog.current ?? 0,
            prog.total ?? 0,
            prog.overall ?? null,
            prog.eta_seconds ?? null,
            prog.overall_step_end ?? null,
          );
        }
      });
  }

  private stopScoringProgressPoll(): void {
    if (this.scoringProgressPoll$) {
      this.scoringProgressPoll$.unsubscribe();
      this.scoringProgressPoll$ = null;
    }
  }

  onModelSelected(modelId: string): void {
    if (!modelId) return;
    this.sortState.setSortMode('load');
    this.sortState.setSortBusy(true);
    this.sortState.setSortStatus('Scoring with detector…');
    this.sortState.setSortProgress(0, 0);

    this.startScoringProgressPoll();

    // Pair-scoped: scoring runs for minutes on a large dataset, so a pair switch
    // mid-run must kill this before it ranks the new pair with old scores.
    this.detectorsFindApi.findLabel({ detector_id: modelId }).pipe(takeUntil(this.pairScope$)).subscribe({
      next: (raw) => {
        const response = raw as {
          results: { id: number; score: number; best_region?: number[] }[];
          threshold: number;
          detector_name?: string;
        };
        this.stopScoringProgressPoll();
        this.applySortWindow(response);
        this.sortState.setLoadSortLabel(response.detector_name || 'Detector');
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
    const response = data as {
      results: { id: number; similarity: number; best_region?: number[] }[];
      threshold: number;
    };
    this.sortState.setSortMode('load');
    this.applySortWindow(response);
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
      // The New pick reads the threshold as a sampling position too (it steers
      // the atlas probe by a node's median score), so it takes the acquisition
      // cut alongside the Hard pick.
      .getCoverageAtlasNext(scores, this.sortState.acqThreshold ?? undefined)
      .pipe(takeUntil(this.pairScope$))
      .subscribe({
        next: (response) => {
          if (response.id !== null) {
            this.mediaState.selectMedia(response.id);
          }
          if (typeof response.coverage_level === 'number') {
            this.autopilotStateService.updateDiversityLevel(response.coverage_level);
          }
        },
      });
  }

  // --- Inclusion ---

  /**
   * Seed the slider from the active detector's per-detector inclusion
   * (GET /api/inclusion, which falls back to the user-settings default the
   * first time it's read for a detector). Called on entry and on every
   * detector switch so the slider tracks the detector, not a stale global.
   */
  private seedInclusion(): void {
    this.sortingApi
      .getInclusion()
      .pipe(takeUntil(this.pairScope$))
      .subscribe({ next: (resp) => this.sortState.setInclusion(resp.inclusion) });
  }

  onInclusionChange(value: number): void {
    this.sortState.setInclusion(value);
    this.sortingApi.setInclusion(value).pipe(takeUntil(this.pairScope$)).subscribe();
    this.autoSelectNext();
    if (this.sortState.sortMode === 'learned' && this.voteState.learnedSortAvailable) {
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

  // --- Right-click media context menu ---

  contextMenuOpen = false;
  contextMenuX = 0;
  contextMenuY = 0;
  contextMenuMediaId: number | null = null;
  contextMenuItems: ContextMenuItem[] = [];

  /** Pending crop modal state. When set, the user has chosen a "Crop and …"
   *  action and we have fetched the media bytes; the crop modal renders. */
  readonly cropPending = signal<{
    file: File;
    mediaId: number;
    mediaType: string;
    /** What to do after the crop is confirmed. */
    action: 'sort' | 'seed';
  } | null>(null);

  onMediaContextRequest(event: { id: number; x: number; y: number }): void {
    const media = this.mediaState.mediasSignal().find((m) => m.id === event.id);
    this.contextMenuItems = buildMediaContextMenuItems(media?.media_type ?? '');
    this.contextMenuMediaId = event.id;
    this.contextMenuX = event.x;
    this.contextMenuY = event.y;
    this.contextMenuOpen = true;
  }

  onContextMenuAction(action: string): void {
    const mediaId = this.contextMenuMediaId;
    this.dismissContextMenu();
    if (mediaId == null) return;

    if (action === 'sort') {
      this.runExampleSortById(mediaId);
    } else if (action === 'seed') {
      this.openSeedNewDetector(mediaId);
    } else if (action === 'crop-sort' || action === 'crop-seed') {
      this.openCropOverlay(mediaId, action === 'crop-sort' ? 'sort' : 'seed');
    }
  }

  dismissContextMenu(): void {
    this.contextMenuOpen = false;
    this.contextMenuMediaId = null;
  }

  private runExampleSortById(mediaId: number, cropParams?: Record<string, unknown>): void {
    this.sortState.setSortBusy(true);
    this.sortState.setSortStatus('Sorting by example…');
    this.sortingApi
      .exampleSortById({ media_id: mediaId, crop_params: cropParams })
      .pipe(takeUntil(this.pairScope$))
      .subscribe({
        next: (response) => {
          this.sortState.setSortMode('load');
          this.applySortWindow(response);
          this.sortState.setLoadSortLabel(this.mediaDisplayName(mediaId));
          this.sortState.setSortBusy(false);
          this.sortState.setSortStatus('');
          this.autoSelectNext();
        },
        error: (err) => {
          this.sortState.setSortBusy(false);
          this.sortState.setSortStatus('Example sort failed');
          this.toast.error({ message: err?.error?.message || 'Example sort failed' });
        },
      });
  }

  private openSeedNewDetector(mediaId: number, cropParams?: Record<string, unknown>): void {
    const media = this.mediaState.mediasSignal().find((m) => m.id === mediaId);
    this.newThingFlows.openNewDetector({
      defaultMediaType: media?.media_type ?? '',
      datasetEmbedder: media?.embedder ?? '',
      seedMediaId: mediaId,
      seedCropParams: cropParams,
    });
  }

  private openCropOverlay(mediaId: number, action: 'sort' | 'seed'): void {
    const media = this.mediaState.getMedia(mediaId);
    if (!media) return;
    const mediaType = media.media_type;
    const url =
      mediaType === 'audio'
        ? this.activeContext.mediaUrl(`/api/medias/${mediaId}/audio`)
        : this.activeContext.mediaUrl(`/api/medias/${mediaId}/image`);

    this.sortState.setSortBusy(true);
    this.sortState.setSortStatus('Loading media for crop…');
    fetch(url, { credentials: 'same-origin' })
      .then((r) => {
        if (!r.ok) throw new Error(`Failed to fetch media (${r.status})`);
        return r.blob();
      })
      .then((blob) => {
        const name = media.filename || `media_${mediaId}`;
        const file = new File([blob], name, { type: blob.type });
        this.cropPending.set({ file, mediaId, mediaType, action });
        this.sortState.setSortBusy(false);
        this.sortState.setSortStatus('');
      })
      .catch((err) => {
        this.sortState.setSortBusy(false);
        this.sortState.setSortStatus('');
        this.toast.error({ message: err?.message || 'Failed to load media for crop' });
      });
  }

  onCropConfirmed(result: MediaCropResult): void {
    const pending = this.cropPending();
    this.cropPending.set(null);
    if (!pending) return;
    const cropParams = result.cropParams as Record<string, unknown> | undefined;
    if (pending.action === 'sort') {
      this.runExampleSortById(pending.mediaId, cropParams);
    } else {
      this.openSeedNewDetector(pending.mediaId, cropParams);
    }
  }

  onCropCancelled(): void {
    this.cropPending.set(null);
  }

  private refreshTrainableModelName(modelId: string): void {
    if (!modelId) {
      this.trainableModelName.set(null);
      return;
    }
    this.detectorsRegistryApi.getRegistry().pipe(takeUntil(this.destroy$)).subscribe({
      next: (resp) => {
        const entry = resp.detectors.find((m: DetectorRegistryEntry) => m.id === modelId);
        this.trainableModelName.set(entry?.name || null);
      },
      error: () => {
        this.trainableModelName.set(null);
      },
    });
  }

  onHoverVote(event: { id: number; vote: 'good' | 'bad' }): void {
    this.voteState
      .submitToggleVoteAndRecord(event.id, event.vote, this.mediaDisplayName(event.id))
      .pipe(takeUntil(this.pairScope$))
      .subscribe({
        next: () => {
          this.onMediaVoted(event);
        },
      });
  }

  private mediaDisplayName(id: number): string {
    const m = this.mediaState.getMedia(id);
    return m?.filename || m?.origin_name || `#${id}`;
  }

  onMediaVoted(event: { id: number; vote: 'good' | 'bad' }): void {
    // Local vote state is already reconciled from the POST response inside
    // submitToggleVote; loadVotes() only refreshes derived counters.
    this.voteState.loadVotes();
    // In train mode the right pane's Good/Bad piles are sourced from the
    // labelset (not the cid-based vote signals), and the labelset only repaints
    // on its 1500ms poll — so a just-cast vote takes up to that long to land in
    // a pile. Kick an immediate labelset refresh so the new label appears as
    // soon as the server has it, instead of waiting for the next poll tick.
    // No-op when no detector is being trained (refresh() bails on a null model).
    this.labelsetState.refresh();
    this.autoSelectNext(event.id);
    if (this.sortState.sortMode === 'learned' && this.voteState.learnedSortAvailable) {
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

  /**
   * Whether the active dataset's embedder can embed text queries. Drives the
   * Text-sort gate and the Autopilot availability check. Defaults to ``true``
   * until medias / the embedder registry have loaded so we never hide a
   * working feature on missing metadata.
   */
  get textSupported(): boolean {
    const medias = this.mediaState.mediasSignal();
    if (medias.length === 0) return true;
    const first = medias[0];
    const names = first.embedders ?? (first.embedder ? [first.embedder] : []);
    return this.embedderCaps.supportsTextAny(names);
  }

  /**
   * True when Autopilot has no way to seed its first sort: the dataset's
   * embedder can't search by text, the detector carries no media-example seed,
   * and there aren't yet enough labels for Learn sort. Bound into the
   * left-panel to disable the Autopilot tab; it re-enables automatically once
   * the user labels a good and a bad (Learn sort becomes available).
   */
  get autopilotDisabled(): boolean {
    if (this.textSupported) return false;
    if (this.labelSession.mediaExampleFilenames.length > 0) return false;
    return !this.voteState.learnedSortAvailable;
  }

  onAutopilotStart(): void {
    // Initialize re-sort tracking
    this.resortVoteCount = 0;
    this.resortNextThreshold = this.resortInterval;

    const state = this.autopilotStateService.state;
    const phase = state.phase;

    // For phases beyond 'good', the phase-transition subscription already set
    // the correct selectMode and (for 'hard') triggered a learned sort.
    // Only override selectMode for the initial 'good' phase.
    if (phase === 'good') {
      this.sortState.setSelectMode('top');
    }

    // Retrain mode: the subscription already set sortMode='learned' and
    // kicked off learned sort for whatever phase we're in.  Nothing more to do.
    if (state.retrainMode) {
      return;
    }

    // For 'hard' and later phases the subscription already triggered learned
    // sort; no text/media sort needed.  For 'good' and 'bad' phases, kick off
    // the text/media sort so the user has results to vote on.
    if (phase === 'good' || phase === 'bad') {
      const textQuery = this.labelSession.textQuery;
      const hasMediaExamples = this.labelSession.mediaExampleFilenames.length > 0;
      if (textQuery) {
        // Defer until both medias and the embedder registry are loaded so the
        // no-text check in `triggerAutopilotTextSort` is reliable.
        if (this.mediaState.mediasSignal().length > 0 && this.embedderCaps.infos() !== null) {
          this.triggerAutopilotTextSort();
        } else {
          this.autopilotTextSortPending = true;
        }
      } else if (hasMediaExamples) {
        if (this.mediaState.mediasSignal().length > 0) {
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
    if (!textQuery) return;
    // No-text dataset, text-hint-only detector: the dataset's embedder can't
    // embed the query, so don't fire a sort that's guaranteed to fail. The
    // left-panel disables the Autopilot tab (see `autopilotDisabled`) and the
    // user labels manually until Learn sort re-enables Autopilot.
    if (!this.textSupported) return;
    this.onTextSort(textQuery);
  }

  private triggerAutopilotMediaSort(): void {
    // Every media example seeds the sort: plural examples rank the haystack
    // against the centroid of their embeddings, so the "good" phase surfaces
    // items resembling what the examples have in common.
    const filenames = this.labelSession.mediaExampleFilenames;
    if (filenames.length > 0) {
      this.sortState.setSortBusy(true);
      this.sortState.setSortStatus(filenames.length > 1 ? 'Sorting by examples…' : 'Sorting by example…');
      this.sortingApi.exampleSortServer({ filenames }).pipe(takeUntil(this.pairScope$)).subscribe({
        next: (response) => {
          this.applySortWindow(response);
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
    // Retrain mode uses learned sort instead of text/example; there is no
    // example to swap, so the resort prompt is irrelevant.
    if (this.autopilotStateService.state.retrainMode) return;
    // Eagerly check phase transition so we don't show the prompt after the user
    // has already found enough greens (the panel's ngOnChanges may not have run yet).
    this.autopilotStateService.checkPhaseTransition(
      this.voteState.goodVotes.size, this.voteState.badVotes.size,
      this.mediaState.mediasSignal().length,
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
      this.showResortPrompt.set(true);
    }
  }

  onResortKeep(): void {
    this.showResortPrompt.set(false);
    // Multiply threshold by 1.5 for next prompt
    this.resortNextThreshold = Math.round(this.resortNextThreshold * 1.5);
    this.resortVoteCount = 0;
  }

  onResortNewExample(result: ResortResult): void {
    this.showResortPrompt.set(false);
    this.resortVoteCount = 0;
    // Reset threshold back to the base interval
    this.resortNextThreshold = this.resortInterval;

    if (result.type === 'text') {
      this.labelSession.textQuery = result.value;
      this.labelSession.mediaExample = '';
      this.labelSession.examples = [{ type: 'text', value: result.value }];
      this.sortState.setSelectMode('top');
      this.triggerAutopilotTextSort();
    } else {
      // The re-sort prompt swaps in a single fresh example, replacing any
      // multi-example seed stack the detector started with.
      this.labelSession.mediaExample = result.value;
      this.labelSession.textQuery = '';
      this.labelSession.examples = [{ type: 'media', value: result.value }];
      this.sortState.setSelectMode('top');
      this.triggerAutopilotMediaSort();
    }
  }

  onResortClosed(): void {
    // Treat closing the modal as "keep"
    this.onResortKeep();
  }

  onAutopilotRefocus(): void {
    this.autoSelectNext();
  }

  onAutopilotToggleCollapse(): void {
    const newVal = !this.autopilotCollapsed();
    this.setAutopilotCollapsed(newVal);
    this.settingsState.update({ hide_autopilot: newVal }).subscribe();
  }

  private setAutopilotCollapsed(collapsed: boolean): void {
    this.autopilotCollapsed.set(collapsed);
    if (collapsed) {
      this.savedLeftWidth = this.leftWidth();
      this.leftWidth.set(this.COLLAPSED_WIDTH);
    } else {
      this.leftWidth.set(this.savedLeftWidth);
    }
    this.layoutRef().nativeElement.style.setProperty('--left-width', `${this.leftWidth()}px`);
  }

  onAutopilotEnabledChange(enabled: boolean): void {
    this.autopilotEnabled.set(enabled);
    this.settingsState.update({ autopilot_enabled: enabled }).subscribe();
  }

  onAutopilotStop(): void {
    const state = this.autopilotStateService.state;
    const phase = state.phase;
    const isMediaBased = !!this.labelSession.mediaExample && !this.labelSession.textQuery;
    // Retrain mode never used text/example sort, so stopping shouldn't switch
    // the UI back to it; keep learned sort selected for every phase.
    const earlySortMode: SortMode = state.retrainMode
      ? 'learned'
      : (isMediaBased ? 'load' : 'text');

    // Map autopilot phase to the same Sort + Select that autopilot was using.
    if (phase === 'good') {
      this.sortState.setSortMode(earlySortMode);
      this.sortState.setSelectMode('top');
    } else if (phase === 'bad') {
      this.sortState.setSortMode(earlySortMode);
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
    this.showResortPrompt.set(false);
    // Drop any deferred seed sort that hasn't fired yet (e.g. we stopped before
    // medias finished loading) so it can't fire after autopilot is gone.
    this.autopilotTextSortPending = false;
    this.autopilotMediaSortPending = false;
  }

  // --- Panel width helpers ---

  /** Apply the panel widths saved for the active media type, clamping against
   *  the current layout bounds. Called when the media type changes or when
   *  fresh per-media-type settings come in. */
  private applyPanelPx(): void {
    const layoutWidth = this.layoutRef().nativeElement.getBoundingClientRect().width || 1200;
    const leftPx = this.panelState.getPanelPx('left');
    if (leftPx != null && !this.autopilotCollapsed()) {
      const leftMax = layoutWidth - this.DIVIDER_TOTAL - this.CENTER_MIN - this.rightWidth();
      this.leftWidth.set(Math.max(this.LEFT_MIN, Math.min(leftMax, leftPx)));
      this.layoutRef().nativeElement.style.setProperty('--left-width', `${this.leftWidth()}px`);
    }
    const rightPx = this.panelState.getPanelPx('right');
    if (rightPx != null) {
      const rightMax = layoutWidth - this.DIVIDER_TOTAL - this.CENTER_MIN - this.leftWidth();
      this.rightWidth.set(Math.max(this.RIGHT_MIN, Math.min(rightMax, rightPx)));
      this.layoutRef().nativeElement.style.setProperty('--right-width', `${this.rightWidth()}px`);
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
    } else if (this.sortState.selectMode === 'hard' && this.sortState.acqThreshold !== null) {
      // The *acquisition* cut, not the decision line: this reads the threshold
      // as a rank position, which is why it wants one further up the ranking
      // than the one the user is shown (#2876).
      const threshold = this.sortState.acqThreshold!;
      // Find the index where the threshold falls in the sorted (descending) list.
      // This is the first position whose score is at or below the threshold.
      let thresholdIndex = sortOrder.length;
      for (let i = 0; i < sortOrder.length; i++) {
        if (sortOrder[i].score <= threshold) {
          thresholdIndex = i;
          break;
        }
      }
      // Pick the unlabeled item whose index is closest to the threshold index.
      // This avoids biasing toward one side when scores cluster unevenly.
      let best: SortedItem | null = null;
      let bestDist = Infinity;
      for (let i = 0; i < sortOrder.length; i++) {
        if (isVoted(sortOrder[i].id)) continue;
        const dist = Math.abs(i - thresholdIndex);
        if (dist < bestDist) {
          bestDist = dist;
          best = sortOrder[i];
        }
      }
      if (best) this.mediaState.selectMedia(best.id);
    } else if (this.sortState.selectMode === 'new') {
      this.fetchDiversityNext();
    }
  }
}
