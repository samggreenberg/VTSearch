import { AfterViewInit, ChangeDetectionStrategy, Component, DestroyRef, effect, ElementRef, inject, OnDestroy, OnInit, signal, untracked, viewChild } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';

import { Subscription, pairwise } from 'rxjs';
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
import { PairScopeService } from '../../services/pair-scope.service';
import { SortRunnerService } from '../../services/sort-runner.service';
import { adaptivePoll } from '../../services/adaptive-poll';
import { DetectorsFindApiService } from '../../services/detectors-find-api.service';
import { DetectorsRegistryApiService } from '../../services/detectors-registry-api.service';
import { LabelSessionService } from '../../services/label-session.service';
import { MediaStateService } from '../../services/media-state.service';
import { VoteStateService } from '../../services/vote-state.service';
import { LabelsetStateService } from '../../services/labelset-state.service';
import { SortStateService, SortMode, SelectMode } from '../../services/sort-state.service';
import { SettingsStateService } from '../../services/settings-state.service';
import { AutopilotStateService } from '../../services/autopilot-state.service';
import { EmbedderCapabilityService } from '../../services/embedder-capability.service';
import { ActiveContextService } from '../../services/active-context.service';
import { DetectorRegistryEntry } from '../../generated/api-client/models/detector-registry-entry';
import { ProgressModalComponent, ProgressMetric } from '../modals/progress-modal/progress-modal.component';
import { ResortPromptModalComponent, ResortResult } from '../modals/resort-prompt-modal/resort-prompt-modal.component';
import type { LabelingStatusResponse } from '../../generated/api-client/models/labeling-status-response';
import { snapPanelWidthToGridColumns, iconSizeToGoalWidth } from '../../utils/grid-icon-size';
import { PanelResizeDirective } from '../../directives/panel-resize.directive';
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
  providers: [LabelViewPanelStateService, PairScopeService, SortRunnerService],
  templateUrl: './label-view.component.html',
  styleUrl: './label-view.component.scss',
})
export class LabelViewComponent implements OnInit, AfterViewInit, OnDestroy {
  private sortingApi = inject(SortingApiService);
  private detectorsFindApi = inject(DetectorsFindApiService);
  private detectorsRegistryApi = inject(DetectorsRegistryApiService);
  private labelSession = inject(LabelSessionService);
  mediaState = inject(MediaStateService);
  voteState = inject(VoteStateService);
  private labelsetState = inject(LabelsetStateService);
  sortState = inject(SortStateService);
  private settingsState = inject(SettingsStateService);
  private autopilotStateService = inject(AutopilotStateService);
  private embedderCaps = inject(EmbedderCapabilityService);
  private activeContext = inject(ActiveContextService);
  private newThingFlows = inject(NewThingFlowsService);
  private toast = inject(ToastService);
  panelState = inject(LabelViewPanelStateService);
  /** Component-provided. Public: the header binds `pairScope.datasetName()`. */
  readonly pairScope = inject(PairScopeService);
  /** Component-provided sort orchestration (#3428): the sorts, and the
   *  `autoSelectNext` each one ends on. Private — the template's surface is the
   *  one-line handlers below, which forward to it. */
  private readonly sortRunner = inject(SortRunnerService);

  readonly layoutRef = viewChild.required<ElementRef<HTMLElement>>('layout');
  readonly centerPanel = viewChild(CenterPanelComponent);

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
  /** True while a windowed-sort "Load more" page fetch is in flight. Aliased
   *  from the runner that owns it, so the template binding is unchanged. */
  readonly loadingMoreSort = this.sortRunner.loadingMoreSort;
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
  private readonly destroyRef = inject(DestroyRef);
  // NOTE: a subscription started from `modelId$` (which emits *before*
  // `pair$`) must stay on component teardown — `takeUntilDestroyed` — rather
  // than the pair scope, or `reloadForNewPair`'s teardown would kill the
  // request it just issued for the new pair.
  private statusPolling$: Subscription | null = null;
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
  /** Armed on each pair reload; consumed by the first ranking that lands for
   *  the new pair, which the centre viewer is then seeded from. See the effect
   *  in the constructor for why the pair change cannot just auto-select itself.
   */
  private pendingSelectOnPairChange = false;

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
        // `panelState` reads the settings signal directly (its prefs are
        // `computed`s), so there is nothing to hydrate here — this branch only
        // reacts to the new values.
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

    // Seed the centre viewer for the new pair, once the pair change produces a
    // ranking to seed it from.
    //
    // The pair change clears the selection, because a media id from the pair we
    // left means nothing under the new one (`PairScopeService.clearPairState`,
    // #3489). Something has to put an item back, and on this path nothing did:
    // *entry* seeds the centre through Autopilot's activation sort
    // (`triggerAutopilotTextSort` -> `onTextSort` -> `autoSelectNext`), which a
    // switch never re-runs, and every re-rank a switch *does* fire passes
    // `autoSelect: false` — correctly, since those same calls also run
    // underneath a user who is mid-labelling, where moving them off the item
    // they are looking at is the bug. So the seed is armed by the pair change
    // itself and consumed here, once, by whichever re-rank happens to land
    // first (the learned-sort rehydration below, an Autopilot phase change, a
    // text sort).
    //
    // Deliberately silent when no ranking ever arrives: switching to a pair the
    // detector has no labelset for leaves the centre on its placeholder, which
    // is exactly where a fresh entry to that same pair leaves it.
    effect(() => {
      const order = this.sortState.sortOrder;
      untracked(() => {
        if (!this.pendingSelectOnPairChange) return;
        if (!order || order.length === 0) return;
        this.pendingSelectOnPairChange = false;
        // A re-rank that auto-selected on its own (or a click that beat us to
        // it) already owns the centre; never move the user off it.
        if (this.mediaState.selectedId() === null) this.sortRunner.autoSelectNext();
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
          this.sortRunner.onLearnedSort(false);
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
    this.loadTrainingVotes();
    this.loadSettings();
    this.startStatusPolling();
    this.pairScope.loadDatasetName();

    this.activeContext.modelId$
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((modelId) => this.refreshTrainableModelName(modelId));
    this.refreshTrainableModelName(this.activeContext.modelId);
    this.pairScope.seedInclusion();

    // Reload data when the active pair changes via the top-bar switcher.
    // Skip the first emission; `ngOnInit` above already triggered the
    // initial loads.
    let firstPair = true;
    this.activeContext.pair$
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe(() => {
        if (firstPair) {
          firstPair = false;
          return;
        }
        this.reloadForNewPair();
      });

    this.autopilotStateService.state$
      .pipe(pairwise(), takeUntilDestroyed(this.destroyRef))
      .subscribe(([prev, curr]) => {
        if (prev.phase === curr.phase) return;
        this.autopilotExhausted.set(curr.phase === 'exhausted');
        if (curr.phase === 'good') {
          this.sortState.setSelectMode('top');
          if (curr.retrainMode) {
            this.sortState.setSortMode('learned');
            this.sortRunner.onLearnedSort(false);
          }
        }
        else if (curr.phase === 'bad') {
          this.sortState.setSelectMode('hard');
          if (curr.retrainMode) {
            this.sortState.setSortMode('learned');
            this.sortRunner.onLearnedSort(false);
          }
        }
        else if (curr.phase === 'hard') {
          this.sortState.setSelectMode('hard');
          this.sortState.setSortMode('learned');
          this.sortRunner.onLearnedSort(false);
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
    // Supersede → quiesce → clear → reload, in that order and enforced there;
    // see `PairScopeService.resetForNewPair`.
    this.pairScope.resetForNewPair(() => {
      // Train's scoring and learned-sort subscriptions carry no `finalize`, so
      // the busy flag, the job id and the progress feed they own are reset here
      // — after the supersede, so nothing can re-set them.
      this.sortRunner.quiesce();
      this.pendingRehydrateLearned = false;
      // Read by the medias effect when the reload below lands.
      this.pendingSnapOnLoad = true;
      // Read by the seed effect when the new pair's first ranking lands.
      this.pendingSelectOnPairChange = true;
    });
    this.loadTrainingVotes();

    // Arm the rehydrate effect: it fires `onLearnedSort` once the reloaded
    // votes land (counts go 0 → available) if the user is still in learned mode.
    this.pendingRehydrateLearned = this.sortState.sortMode === 'learned';
  }

  /**
   * Load the votes this window trains from, ending any live Find session first.
   *
   * Find fills the *same* per-detector vote dicts as training does, but with
   * the detector's own call for every item in the dataset — a presumption, not
   * a human decision. Read as votes they make the whole collection look
   * labeled, which lands Autopilot in a terminal phase the moment the user
   * arrives, and the server-side find-mode guard keeps every vote cast here out
   * of the labelset (#3212). Entering the Train window is the statement that
   * this is training, so it says so before it reads the votes.
   *
   * The load runs either way: a failed hand-off is no reason to leave the
   * window with no votes at all.
   */
  private loadTrainingVotes(): void {
    this.detectorsFindApi
      .endFindSession()
      .pipe(this.pairScope.scoped())
      .subscribe({
        next: () => this.voteState.loadVotes(),
        error: () => this.voteState.loadVotes(),
      });
  }

  ngOnDestroy(): void {
    this.sortState.stopFindProgressTracking();
    this.cancelAutoPop('left');
    this.cancelAutoPop('right');
    this.cancelSnapOnLoad();
    if (this.animatePopTimer) clearTimeout(this.animatePopTimer);
    // `pairScope` is component-provided, so Angular fires its scope on destroy.
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

  /** Right-pane grid goal width for the active media type. The right pane owns
   *  its own size key (`grid_icon_size_right`); the left one lives on
   *  `panelState`. Both are `perMediaType` prefs over the settings signal. */
  private readonly gridIconSizeRight = this.settingsState.perMediaType<string>(
    'grid_icon_size_right',
    this.panelState.mediaType,
    { fallback: 'M' },
  );

  private currentRightGoalWidth(): number {
    return iconSizeToGoalWidth(this.gridIconSizeRight.value());
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
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (status) => {
          this.labelingStatus.set(status);
        },
      });
  }

  // --- Sort handlers ---
  //
  // The sorts themselves live on `SortRunnerService` (component-provided, so it
  // can use the same pair scope this view does). What is left here is the
  // template's event surface: one line each, no logic.

  onSortModeChange(mode: SortMode): void {
    this.sortRunner.onSortModeChange(mode);
  }

  onLoadMore(): void {
    this.sortRunner.onLoadMore();
  }

  onTextSort(text: string): void {
    this.sortRunner.onTextSort(text);
  }

  onLearnedSort(autoSelect = true): void {
    this.sortRunner.onLearnedSort(autoSelect);
  }

  onSortCancel(): void {
    this.sortRunner.onSortCancel();
  }

  onLoadSort(): void {
    // Re-sort using existing load sort results when switching back to load mode
  }

  onModelSelected(modelId: string): void {
    this.sortRunner.onModelSelected(modelId);
  }

  onExampleSortStarted(data: unknown): void {
    this.sortRunner.onExampleSortStarted(data);
  }

  // --- Select mode ---

  onSelectModeChange(mode: SelectMode): void {
    this.sortRunner.onSelectModeChange(mode);
  }

  // --- Inclusion ---

  onInclusionChange(value: number): void {
    this.sortRunner.onInclusionChange(value);
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
    this.sortRunner.runExampleSortById(mediaId, this.mediaDisplayName(mediaId), cropParams);
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
    this.detectorsRegistryApi.getRegistry().pipe(takeUntilDestroyed(this.destroyRef)).subscribe({
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
      .pipe(this.pairScope.scoped())
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
    this.sortRunner.autoSelectNext(event.id);
    if (this.sortState.sortMode === 'learned' && this.voteState.learnedSortAvailable) {
      this.sortRunner.scheduleLearnedSort(false);
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
        this.sortRunner.autoSelectNext();
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
    this.sortRunner.onTextSort(textQuery);
  }

  private triggerAutopilotMediaSort(): void {
    this.sortRunner.exampleSortByFilenames(this.labelSession.mediaExampleFilenames);
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
    this.sortRunner.autoSelectNext();
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


}
