import { ChangeDetectionStrategy, Component, computed, effect, ElementRef, HostListener, inject, NgZone, OnDestroy, OnInit, signal, untracked, viewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router } from '@angular/router';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import {
  BrowseCanvasComponent,
  BrowseContextMenuEvent,
  HexHoverEvent,
} from '../browse-canvas/browse-canvas.component';
import type { BrowseGraphicsMode } from '../browse-canvas/render-perf';
import { BrowseHoverPreviewComponent, NowPlaying } from '../browse-hover-preview/browse-hover-preview.component';
import { BrowseBinPopupComponent } from '../browse-bin-popup/browse-bin-popup.component';
import { BrowseLegendComponent } from '../browse-legend/browse-legend.component';
import { BrowseSelectionPanelComponent } from '../browse-selection-panel/browse-selection-panel.component';
import { BrowseMinimapComponent } from '../browse-minimap/browse-minimap.component';
import { ProgressBarComponent } from '../progress-bar/progress-bar.component';
import { IconComponent } from '../icon/icon.component';
import { NoFocusStealDirective } from '../../directives/no-focus-steal.directive';
import { ProjectionApiService } from '../../services/projection-api.service';
import { TileCacheService } from '../../services/tile-cache.service';
import { ActiveContextService } from '../../services/active-context.service';
import { DatasetsRegistryApiService } from '../../services/datasets-registry-api.service';
import { DetectorsRegistryApiService } from '../../services/detectors-registry-api.service';
import { SettingsStateService } from '../../services/settings-state.service';
import { BrowseViewportService } from '../../services/browse-viewport.service';
import { BrowseSelectionService } from '../../services/browse-selection.service';
import { BrowseSubsetService } from '../../services/browse-subset.service';
import { MediasApiService } from '../../services/medias-api.service';
import { VtDialogService } from '../../services/dialog.service';
import { ToastService } from '../../services/toast.service';
import { MediaTypeCapabilityService } from '../../services/media-type-capability.service';
import {
  BROWSE_COLORMAP_IDS,
  DEFAULT_THUMBNAIL_BORDER,
  MAX_THUMBNAIL_BORDER,
  type BrowseColormapId,
} from '../browse-canvas/hex-render.util';
import type { ProjectionMeta, RegionLabelPayload } from '../../models/projection.models';
import { snapPanelWidthToGridColumns } from '../../utils/grid-icon-size';
import { progressBarState } from '../../utils/format-progress';
import { shortcutsBlocked } from '../../utils/keyboard-shortcuts';
import type { AppSettings } from '../../generated/api-client/models/app-settings';
import type { SettingsUpdate } from '../../generated/api-client/models/settings-update';
import { apiErrorMessage } from '../../utils/api-error';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-browse-view',
  standalone: true,
  imports: [
    CommonModule,
    BrowseCanvasComponent,
    BrowseHoverPreviewComponent,
    BrowseLegendComponent,
    BrowseSelectionPanelComponent,
    BrowseMinimapComponent,
    ProgressBarComponent,
    IconComponent,
    BrowseBinPopupComponent,
    NoFocusStealDirective,
  ],
  // Scoped per browse view so the canvas, its minimap, and the selection panel
  // share one viewport channel and one selection set without leaking across
  // other instances of the view.
  providers: [BrowseViewportService, BrowseSelectionService],
  templateUrl: './browse-view.component.html',
  styleUrl: './browse-view.component.scss',
})
export class BrowseViewComponent implements OnInit, OnDestroy {
  private projectionApi = inject(ProjectionApiService);
  private tileCache = inject(TileCacheService);
  private activeContext = inject(ActiveContextService);
  private datasetsRegistryApi = inject(DatasetsRegistryApiService);
  private detectorsRegistryApi = inject(DetectorsRegistryApiService);
  private settingsState = inject(SettingsStateService);
  private route = inject(ActivatedRoute);
  private router = inject(Router);
  private browseSubset = inject(BrowseSubsetService);
  private selection = inject(BrowseSelectionService);
  private mediasApi = inject(MediasApiService);
  private dialog = inject(VtDialogService);
  private toast = inject(ToastService);
  private mediaTypeCaps = inject(MediaTypeCapabilityService);
  private ngZone = inject(NgZone);

  private readonly canvas = viewChild(BrowseCanvasComponent);
  /** The active bin-details popup — the docked panel or the floating window,
   *  whichever is mounted (they're mutually exclusive). Used to snap the details
   *  divider tight to its grid columns on release. */
  private readonly binPopup = viewChild(BrowseBinPopupComponent);
  /** The 3-column grid (canvas | divider | side panel); the divider drag
   *  measures against its box. Only present in the ``ready`` state. */
  private readonly content = viewChild<ElementRef<HTMLElement>>('content');

  readonly meta = signal<ProjectionMeta | null>(null);
  // Written from raw/async callbacks (dataset-status subscribe, projection
  // build poller, settings effect), so a signal so those writes repaint the
  // canvas/minimap inputs and the info row under zoneless.
  readonly mediaType = signal('');
  /**
   * Preview-audio volume (0–1) for the Browse view's hover/bin-popup playback,
   * seeded from and written back to the shared ``volume`` setting so it stays in
   * lockstep with the Find view's player. Only surfaced (as a toolbar mute +
   * slider) for audio datasets — the only media type Browse plays sound for.
   */
  readonly volume = signal(1);
  /** Last non-zero level, so the mute toggle can restore where the user was. */
  private preMuteVolume = 1;
  hoverEvent: HexHoverEvent | null = null;
  /** The clip currently auditioning from a canvas-bin or bin-popup hover, or
   *  ``null`` when silent. Drives the top-left now-playing waveform. */
  readonly nowPlaying = signal<NowPlaying | null>(null);
  /**
   * Top of the density scale for the legend: the densest cell currently in
   * view, pushed up from the canvas. The legend labels the yellow end with it.
   */
  densityMax = 1;
  readonly status = signal<'loading' | 'building' | 'ready' | 'error' | 'done'>('loading');
  /**
   * Whether the canvas content is uncovered. The canvas is mounted (so it can
   * lay out and fetch) as soon as {@link status} is ``ready``, but a cover stays
   * over it until the canvas signals its opening view is fully painted — the
   * fit ran and the top view's thumbnails decoded — so the user is shown the
   * finished view instead of a thumbnail-less grid that then fills in. Reset to
   * ``false`` each time a *fresh* canvas mounts (see {@link enterReady}); an
   * in-place refresh of an already-ready canvas leaves it untouched.
   */
  readonly revealed = signal(false);
  readonly errorMessage = signal('');
  readonly buildProgress = signal(0);
  readonly buildTotal = signal(0);
  readonly buildMessage = signal('');
  /** Whole-job build position: which coarse phase (arranging → tiling →
   *  naming regions) is running, and the stitched 0..1 fraction across all of
   *  them, both reported by the projection meta. */
  readonly buildStep = signal<number | null>(null);
  readonly buildTotalSteps = signal<number | null>(null);
  readonly buildOverall = signal<number | null>(null);

  /** `<vt-progress-bar>` inputs for the build state, preferring the whole-job
   *  ``overall`` fraction so the bar fills once across the build rather than
   *  restarting at each phase. */
  readonly buildBar = computed(() =>
    progressBarState({
      current: this.buildProgress(),
      total: this.buildTotal(),
      overall: this.buildOverall(),
    }),
  );

  /** Phase line under the bar: `"Step 2 of 3 · building pyramid"`. */
  readonly buildDetail = computed(() => {
    const step = this.buildStep();
    const totalSteps = this.buildTotalSteps();
    const phase = step != null && totalSteps != null && totalSteps > 1 ? `Step ${step} of ${totalSteps}` : '';
    return [phase, this.buildMessage()].filter(Boolean).join(' · ');
  });

  /**
   * Discrete thumbnail-size multipliers, one per named icon size (XS…XL),
   * applied to the canvas's default target bin radius. ``M`` (index 2, scale 1)
   * is the default. The bigger/smaller buttons step through these and the
   * Settings → Browser tab picks one by name. Growing the size makes the *same
   * bins* use more pixels (the view zooms in lock-step, so the binning — which
   * vectors land in a hex — is unchanged); it is not the "Zoom" control, which
   * instead re-bins a smaller region more finely. The persisted per-media value
   * is the size *label* (see {@link ICON_SIZES}), not the multiplier.
   */
  private readonly HEX_SCALES = [0.5, 0.75, 1, 1.6, 2.5, 4, 6.25, 10, 16];
  /** Named icon sizes, index-aligned with {@link HEX_SCALES}. The top steps
   * (4XL/5XL) render a cell close to the full media, so some users use them to
   * inspect (essentially) the whole image rather than a thumbnail. */
  static readonly ICON_SIZES = ['XS', 'S', 'M', 'L', 'XL', '2XL', '3XL', '4XL', '5XL'] as const;
  readonly hexScaleIndex = signal(2);

  /**
   * Density colormap preset for the flat (non-thumbnail) shading, mirrored
   * from the per-media ``browse_colormap`` setting and passed to the canvas
   * and minimap. ``auto`` follows the theme (Ocean in light, Heat in dark).
   */
  readonly colormap = signal<BrowseColormapId>('auto');

  /**
   * Width (CSS px) of the colormap-coloured border drawn around multi-item
   * pile thumbnails, mirrored from the per-media ``browse_thumbnail_border``
   * setting and passed to the canvas. Only image/video datasets paint
   * thumbnails, so it has no visible effect for other media types.
   */
  readonly thumbnailBorder = signal(DEFAULT_THUMBNAIL_BORDER);

  /** Last settings snapshot, kept so per-media browser prefs can be
   *  re-resolved when the active media type becomes known after load. */
  private lastSettings: AppSettings | null = null;

  /**
   * Region-select mode: the GUI parallel to the Shift+drag hotkey. When on, a
   * plain left-drag on the canvas rubber-bands a selection marquee instead of
   * panning. The toolbar button surfaces (and toggles) it so the gesture is
   * discoverable; Shift+drag keeps working whether or not this is on. Not
   * persisted — it's a transient interaction mode, reset on each visit.
   */
  marqueeMode = false;

  /**
   * Whether Shift is currently held. Shift+drag draws a region marquee, so while
   * Shift is down we momentarily engage the region-select affordance — the
   * toolbar button lights up and the canvas cursor switches to a crosshair — the
   * same way holding Shift for region voting previews that mode. Reset on window
   * blur so alt-tabbing away doesn't strand the view in the engaged state.
   */
  readonly shiftHeld = signal(false);
  private keyDownHandler?: (e: KeyboardEvent) => void;
  private keyUpHandler?: (e: KeyboardEvent) => void;
  private blurHandler?: () => void;

  /**
   * Subset mode: browse an ephemeral UMAP fit over just a handful of media
   * (the positives of a Find run) instead of the full dataset. Set from the
   * `?subset=1` query param plus a handoff from {@link BrowseSubsetService}.
   * `subsetIds` is kept on the component so re-resolving the projection (e.g.
   * a bin-shape switch) re-sends the same ids without a fresh handoff.
   */
  subset = false;
  subsetIds: number[] = [];

  /**
   * Width (CSS px) of the docked side panel (selection list + legend +
   * overview minimap), mirrored from the ``browse_panel_width`` setting and
   * driven by the draggable divider between the canvas and the panel.
   */
  readonly panelWidth = signal(300);
  /** Clamp + divider geometry, mirroring the Find view's panel dividers. */
  /** Absolute smallest the panel may become, used before the panel's content
   *  has laid out (so {@link panelMin} can't measure yet) and as a hard floor
   *  under the dynamic, content-derived minimum. */
  private static readonly PANEL_FLOOR = 96;
  private static readonly PANEL_MAX = 800;
  private static readonly CANVAS_MIN = 200;
  private static readonly DIVIDER_WIDTH = 8;
  /** Once the saved width has been applied to the live panel, the panel width
   *  is owned by the user's divider drags: later settings pushes (our own save
   *  round-trip, an unrelated per-media pref change, …) must not reset it, or
   *  the panel "pops back" to the stored width right after a drag. */
  private panelWidthInitialized = false;
  private dragging = false;

  /**
   * Docked bin-details mode for the active media type, mirrored from the
   * per-media ``bin_details_docked`` setting. When on (the default for a
   * media type with no saved choice), the bin details render as a persistent
   * left panel (large item + metadata on top, member grid below) instead of
   * the floating right-click popup; the panel doubles as the audio
   * now-playing display, replacing the toolbar's small waveform widget.
   * Flipped by the dock button on the floating window and the pop-out button
   * on the panel, both of which persist the choice.
   */
  readonly detailsDocked = signal(true);

  /**
   * Whether the user has dismissed the (docked) details panel for now. The
   * panel's X clears the bin it's showing; pressed again on an already-empty
   * panel it hides the panel altogether, handing its width back to the canvas.
   * Right-clicking a bin brings the panel straight back with that bin in it
   * (see {@link onCanvasContextMenu}), so this is a transient, per-visit
   * dismissal rather than a remembered presentation choice — the docked /
   * floating choice stays with {@link detailsDocked}.
   */
  readonly detailsPanelHidden = signal(false);

  /** Whether the docked details panel actually renders: docked *and* not
   *  dismissed. Everything that keys off the panel's presence (the grid
   *  template, the panel element, the toolbar's now-playing waveform the panel
   *  otherwise replaces) reads this rather than {@link detailsDocked}. */
  readonly detailsPanelShown = computed(() => this.detailsDocked() && !this.detailsPanelHidden());

  /**
   * Width (CSS px) of the docked bin-details panel, mirrored from the
   * ``browse_details_panel_width`` setting and driven by the draggable divider
   * between the panel and the canvas.
   */
  readonly detailsPanelWidth = signal(340);
  /** Smallest the details panel may shrink to (matches the backend clamp). */
  private static readonly DETAILS_FLOOR = 220;
  /** Mirrors {@link panelWidthInitialized} for the details panel. */
  private detailsWidthInitialized = false;
  private draggingDetails = false;
  private boundDetailsMove = this.onDetailsMouseMove.bind(this);
  private boundDetailsUp = this.onDetailsMouseUp.bind(this);

  /**
   * True while the docked details panel's *horizontal* row divider (between the
   * focused-item/metadata row and the member grid, inside the panel) is being
   * dragged. That divider resizes the whole panel — the focused item is square,
   * so a taller item means a wider panel — by mapping the vertical drag onto the
   * same {@link detailsPanelWidth} the panel↔canvas divider drives. A signal so
   * the drag-cue class round-trips to the panel under zoneless.
   */
  readonly draggingDetailsRow = signal(false);
  private detailsRowStartY = 0;
  private detailsRowStartWidth = 0;
  private boundDetailsRowMove = this.onDetailsRowMouseMove.bind(this);
  private boundDetailsRowUp = this.onDetailsRowMouseUp.bind(this);
  /** Dynamic minimum snapshotted at drag start, so the per-move handler reuses
   *  it instead of re-measuring the DOM (and thrashing layout) on every move —
   *  the thumbnail size and sort-control width can't change mid-drag anyway. */
  private dragMin = BrowseViewComponent.PANEL_FLOOR;
  private boundPanelMove = this.onPanelMouseMove.bind(this);
  private boundPanelUp = this.onPanelMouseUp.bind(this);

  /**
   * Region signpost labels for the current projection, fetched once per
   * ``projection_id`` (only when the meta advertises ``has_labels``) and passed
   * to the canvas. Empty for datasets with no computed labels — the signpost
   * toggle button disables itself then.
   */
  readonly labels = signal<RegionLabelPayload[]>([]);

  /**
   * Whether the canvas draws the region signposts — mirrored from the
   * per-media ``browse_signposts`` setting (default on) and toggled by the
   * signpost button in the canvas toolbar, which persists the choice back.
   */
  readonly signposts = signal(true);

  /**
   * Canvas rendering effort — the ``browse_graphics`` setting, mirrored here
   * and passed to the canvas. Unlike the other browse prefs this is a single
   * scalar rather than a per-media-type map: it describes the client's
   * rendering capability, not anything about the data. ``auto`` (the default)
   * lets the canvas detect a software-rendering browser for itself.
   */
  readonly graphics = signal<BrowseGraphicsMode>('auto');

  /** The ``projection_id`` the current {@link labels} were fetched for (or
   *  requested for — set before the request so a poll can't double-fetch). */
  private labelsFetchedFor = '';

  /**
   * How many +/- button clicks (and wheel notches) cross one pyramid level —
   * the per-media ``browse_mouse_zooms_per_level`` setting, mirrored here and
   * passed to the canvas so the wheel matches the buttons. A pyramid level spans
   * a full 2x of zoom, so the per-step width factor is ``2 ** (1 / n)``: 1 ⇒ 2x
   * (one click per level), 2 ⇒ √2 (the default, two clicks per level), 3 ⇒ ∛2.
   * Clamped to 1..3; falls back to 2 when unset for the active media type.
   */
  readonly zoomsPerLevel = signal(2);

  /**
   * Per-click zoom step for the on-screen +/- buttons, derived from
   * {@link zoomsPerLevel}. Matches the wheel's per-notch factor so the two
   * gestures stay in lock-step; button zoom anchors at the viewport centre
   * (no cursor to zoom toward).
   */
  private get zoomButtonFactor(): number {
    return Math.pow(2, 1 / this.zoomsPerLevel());
  }

  /** Bin-popup state: open flag, the viewport anchor it opens at, and the
   *  member ids of the bin the user right-clicked. Right-clicking a bin pops a
   *  scrollable list of its items (hear on hover, click to select) instead of an
   *  action menu; right-clicking empty space closes it. */
  contextMenuOpen = false;
  contextMenuX = 0;
  contextMenuY = 0;
  contextMembers: number[] = [];
  /** Representative (centroid) id of the right-clicked bin — the popup opens on
   *  it and scrolls its 1-D member list to it, so the detail view starts on the
   *  same image whose thumbnail the user clicked. */
  contextRepId: number | null = null;
  /** Canvas bounds the popup clamps inside, so it stays on the canvas rather
   *  than spilling onto the side panel or past the canvas edges. */
  contextBounds: DOMRect | null = null;

  private destroy$ = new Subject<void>();
  private polling = false;
  private pollTimer: ReturnType<typeof setTimeout> | null = null;
  private pollErrors = 0;
  private static readonly MAX_POLL_ERRORS = 5;

  /**
   * Gate for the *first* projection load. Held until the per-media display
   * prefs (cell size, colormap) are known, which needs two async facts: the
   * saved settings and the dataset's media type (those prefs are keyed
   * per-media). Loading sooner would fit the canvas at the default ``M`` cell
   * size and then re-frame once the saved size arrives — a visible jump. We'd
   * rather hold on the loading spinner until the prefs are in. Once true, later
   * reloads (pair change) go through the normal paths.
   */
  private initialLoadStarted = false;
  /** Set once the dataset-status fetch — which yields the media type the
   *  per-media display prefs are keyed on — has resolved or failed. */
  private statusResolved = false;

  constructor() {
    effect(() => {
      const settings = this.settingsState.settingsSignal();
      // Also track the settings load error so this effect re-runs (and the
      // gated initial load can proceed with defaults) if settings never load,
      // rather than stranding the view on the loading spinner forever.
      const settingsErrored = this.settingsState.error() != null;
      if (settings) {
        this.lastSettings = settings;
        const vol = settings.volume ?? 1;
        this.volume.set(vol);
        if (vol > 0) this.preMuteVolume = vol;
        if (settings.browse_panel_width != null && !this.panelWidthInitialized) {
          this.panelWidth.set(
            this.clamp(
              settings.browse_panel_width,
              this.panelMin(),
              BrowseViewComponent.PANEL_MAX,
            ),
          );
          this.panelWidthInitialized = true;
        }
        if (settings.browse_details_panel_width != null && !this.detailsWidthInitialized) {
          this.detailsPanelWidth.set(
            this.clamp(
              settings.browse_details_panel_width,
              BrowseViewComponent.DETAILS_FLOOR,
              BrowseViewComponent.PANEL_MAX,
            ),
          );
          this.detailsWidthInitialized = true;
        }
        this.applyBrowsePrefsForMediaType();
      }
      if (settings || settingsErrored) this.maybeStartInitialLoad();
    });
  }

  /**
   * Start the first projection load once — but only when the per-media display
   * prefs are known: the saved settings must be in (or their load have failed)
   * AND the dataset's media type resolved. See {@link initialLoadStarted}.
   */
  private maybeStartInitialLoad(): void {
    if (this.initialLoadStarted) return;
    const settingsIn = this.lastSettings != null || this.settingsState.error() != null;
    if (!settingsIn || !this.statusResolved) return;
    this.initialLoadStarted = true;
    this.loadProjection();
  }

  ngOnInit(): void {
    this.setupShiftListeners();
    this.settingsState.load();
    // Data-drive the thumbnail-type set (bin shape, grayscale pinning, per-item
    // thumbnails in the popup/selection panel) from the served has_thumbnail.
    this.mediaTypeCaps.ensureLoaded();

    // Subset mode: the Find view handed off a set of positive ids to project
    // on their own. Detect it from the query param + the in-memory handoff.
    this.subset = this.route.snapshot.queryParamMap.get('subset') === '1';
    if (this.subset) {
      const handoff = this.browseSubset.take();
      if (handoff && handoff.ids.length > 0) {
        this.subsetIds = handoff.ids;
      } else {
        // No handoff (e.g. a hard reload): the ephemeral subset is gone.
        this.status.set('error');
        this.errorMessage.set(
          'This map has expired. Re-run Find and click Browse to rebuild it.',
        );
        return;
      }
    }
    this.tileCache.setSubset(this.subset);

    this.datasetsRegistryApi
      .getStatus()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (status) => {
          this.mediaType.set(status.media_type || '');
          this.applyBrowsePrefsForMediaType();
          this.statusResolved = true;
          this.maybeStartInitialLoad();
        },
        error: () => {
          // Couldn't read the dataset status (media type unknown). Proceed with
          // the default lattice rather than hang on the loading spinner.
          this.statusResolved = true;
          this.maybeStartInitialLoad();
        },
      });

    // The full-dataset projection re-resolves when the active pair changes via
    // the top bar. A subset projection is tied to the ids that produced it, so
    // ignore pair changes in subset mode. The first (replayed) emission lands
    // before settings/status are in, so it routes through the gated initial
    // load; genuine later changes reload directly.
    if (!this.subset) {
      this.activeContext.pair$
        .pipe(takeUntil(this.destroy$))
        .subscribe(() => {
          if (this.initialLoadStarted) {
            this.loadProjection();
          } else {
            this.maybeStartInitialLoad();
          }
        });
    }
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
    if (this.pollTimer) clearTimeout(this.pollTimer);
    document.removeEventListener('mousemove', this.boundPanelMove);
    document.removeEventListener('mouseup', this.boundPanelUp);
    document.removeEventListener('mousemove', this.boundDetailsMove);
    document.removeEventListener('mouseup', this.boundDetailsUp);
    document.removeEventListener('mousemove', this.boundDetailsRowMove);
    document.removeEventListener('mouseup', this.boundDetailsRowUp);
    this.removeShiftListeners();
    this.tileCache.clear();
    this.releaseEphemeralPositivesContext();
  }

  /**
   * Track the Shift key at the window level so {@link regionDrawActive} reflects
   * it. The canvas detects `event.shiftKey` directly when a drag starts, so this
   * is purely for the affordance (button + cursor); the blur reset mirrors the
   * image viewer's region-draw handling.
   */
  private setupShiftListeners(): void {
    this.keyDownHandler = (e: KeyboardEvent) => {
      if (e.key === 'Shift') this.shiftHeld.set(true);
    };
    this.keyUpHandler = (e: KeyboardEvent) => {
      if (e.key === 'Shift') this.shiftHeld.set(false);
    };
    this.blurHandler = () => this.shiftHeld.set(false);
    window.addEventListener('keydown', this.keyDownHandler);
    window.addEventListener('keyup', this.keyUpHandler);
    window.addEventListener('blur', this.blurHandler);
  }

  private removeShiftListeners(): void {
    if (this.keyDownHandler) window.removeEventListener('keydown', this.keyDownHandler);
    if (this.keyUpHandler) window.removeEventListener('keyup', this.keyUpHandler);
    if (this.blurHandler) window.removeEventListener('blur', this.blurHandler);
  }

  /**
   * Region-select armed: either the GUI toggle is on, or Shift is held. Drives
   * both the toolbar button's engaged state and the canvas crosshair cursor, so
   * holding Shift previews the Shift+drag region gesture.
   */
  get regionDrawActive(): boolean {
    return this.marqueeMode || this.shiftHeld();
  }

  /** Free a detector-positives browse context (`__detpos__<detectorId>`) when
   *  leaving the view, so its in-memory vectors + preview bytes aren't leaked. */
  private releaseEphemeralPositivesContext(): void {
    const datasetId = this.route.snapshot.paramMap.get('datasetId') || '';
    const prefix = '__detpos__';
    if (!datasetId.startsWith(prefix)) return;
    const detectorId = datasetId.slice(prefix.length);
    // Clear the active pair if it still points at this throwaway context, so
    // dashboard requests don't keep tagging a released id (unless the user
    // already switched to another dataset, in which case leave that alone).
    if (this.activeContext.datasetId === datasetId) {
      this.activeContext.setActive('', '');
    }
    this.detectorsRegistryApi.releasePositivesBrowse(detectorId).subscribe({
      error: () => {
        /* best-effort cleanup; the context is harmless if it lingers */
      },
    });
  }

  onHexHover(event: HexHoverEvent | null): void {
    this.hoverEvent = event;
  }

  /** A canvas-bin or bin-popup hover started/stopped auditioning a clip;
   *  update the shared top-left now-playing indicator. */
  onNowPlaying(event: NowPlaying | null): void {
    this.nowPlaying.set(event);
  }

  onDensityMaxChanged(max: number): void {
    this.densityMax = max;
  }

  /** Noun for the item-count chip — "positives" for a Find-subset browse. */
  get countNoun(): string {
    return this.subset ? 'positives' : 'items';
  }

  get atMinHexSize(): boolean {
    return this.hexScaleIndex() === 0;
  }

  get atMaxHexSize(): boolean {
    return this.hexScaleIndex() === this.HEX_SCALES.length - 1;
  }

  /** Target bin radius (CSS px) for the current named thumbnail size. Also
   *  passed to the bin popup so its preview pane tracks the on-canvas hover
   *  size. */
  get thumbnailRadius(): number {
    return BrowseCanvasComponent.DEFAULT_TARGET_RADIUS * this.HEX_SCALES[this.hexScaleIndex()];
  }

  /** Grow (+1) or shrink (-1) the on-screen thumbnail size, clamped to the
   *  range, and persist the new size as the per-media ``browse_icon_size``.
   *  Reframes the canvas so the same bins render bigger/smaller (the region
   *  shrinks/grows) without changing the binning. */
  bumpHexSize(delta: 1 | -1): void {
    const next = Math.max(
      0,
      Math.min(this.HEX_SCALES.length - 1, this.hexScaleIndex() + delta),
    );
    if (next === this.hexScaleIndex()) return;
    this.hexScaleIndex.set(next);
    this.canvas()?.setThumbnailRadius(this.thumbnailRadius, true);
    this.persistBrowsePref('browse_icon_size', BrowseViewComponent.ICON_SIZES[next]);
  }

  /**
   * Re-resolve the per-media browser preferences (colormap, icon size) for the
   * active media type from the last settings snapshot. Called when settings
   * arrive and again once the media type is known (it lands after the initial
   * settings push), so a saved-per-type choice is applied even though the two
   * facts arrive out of order. (The bin shape is not a preference — it is fixed
   * by media type and reported by the projection meta.)
   */
  private applyBrowsePrefsForMediaType(): void {
    const s = this.lastSettings;
    if (!s) return;
    const mt = this.mediaType();

    // Thumbnail media (image/video) are pinned to grayscale so the colourful
    // density presets never tint real thumbnails; the saved per-type value is
    // ignored for them (the Settings UI hides the picker to match).
    if (this.mediaTypeCaps.usesThumbnails(mt)) {
      this.colormap.set('gray');
    } else {
      const cmap = mt ? this.perMediaValue(s.browse_colormap, mt) : '';
      this.colormap.set(
        cmap && (BROWSE_COLORMAP_IDS as readonly string[]).includes(cmap)
          ? (cmap as BrowseColormapId)
          : 'auto',
      );
    }

    const sizeLabel = mt ? this.perMediaValue(s.browse_icon_size, mt) : '';
    const sizeIdx = (BrowseViewComponent.ICON_SIZES as readonly string[]).indexOf(sizeLabel);
    this.hexScaleIndex.set(sizeIdx >= 0 ? sizeIdx : 2);
    // Seed the saved size as the overview granularity (no reframe): a settings
    // change re-bins at the current framing rather than zooming the viewport,
    // and on first load the initial fit picks the matching level. The query is
    // read untracked: this runs inside the constructor's settings effect, and
    // tracking it there would re-run that effect on every canvas mount/unmount,
    // which the decorator @ViewChild it replaces never did.
    untracked(this.canvas)?.setThumbnailRadius(this.thumbnailRadius, false);

    const borderMap = s.browse_thumbnail_border as { [key: string]: number } | undefined;
    const rawBorder = mt && borderMap ? borderMap[mt] : undefined;
    this.thumbnailBorder.set(
      rawBorder == null
        ? DEFAULT_THUMBNAIL_BORDER
        : Math.max(0, Math.min(MAX_THUMBNAIL_BORDER, rawBorder)),
    );

    const zoomsMap = s.browse_mouse_zooms_per_level as { [key: string]: number } | undefined;
    const rawZooms = mt && zoomsMap ? zoomsMap[mt] : undefined;
    this.zoomsPerLevel.set(
      rawZooms == null ? 2 : Math.max(1, Math.min(3, Math.round(rawZooms))),
    );

    const signMap = s.browse_signposts as { [key: string]: boolean } | undefined;
    const rawSigns = mt && signMap ? signMap[mt] : undefined;
    this.signposts.set(rawSigns == null ? true : rawSigns);

    // Global (not per-media): the client's rendering capability.
    const rawGraphics = s.browse_graphics;
    this.graphics.set(
      rawGraphics === 'full' || rawGraphics === 'reduced' ? rawGraphics : 'auto',
    );

    const dockMap = s.bin_details_docked as { [key: string]: boolean } | undefined;
    const dockedValue = mt && dockMap ? dockMap[mt] : undefined;
    this.detailsDocked.set(dockedValue === undefined ? true : dockedValue);
  }

  /** Read a ``{media_type: value}`` setting for *mt*, or ``''`` when unset. */
  private perMediaValue(map: { [key: string]: string } | undefined, mt: string): string {
    if (!map || !mt) return '';
    return map[mt] ?? '';
  }

  /** Persist a per-media browser preference, merging into the current map so
   *  other media types' choices are preserved, and update the local snapshot
   *  so subsequent reads stay consistent before the PUT round-trips. */
  private persistBrowsePref(
    key: 'browse_colormap' | 'browse_icon_size',
    value: string,
  ): void {
    const mt = this.mediaType();
    if (!mt) return;
    const existing = (this.lastSettings?.[key] as { [k: string]: string } | undefined) || {};
    const next = { ...existing, [mt]: value };
    if (this.lastSettings) {
      (this.lastSettings as Record<string, unknown>)[key] = next;
    }
    this.settingsState.update({ [key]: next } as SettingsUpdate).subscribe();
  }

  private clamp(value: number, lo: number, hi: number): number {
    return Math.max(lo, Math.min(hi, value));
  }

  /** Largest the side panel can grow to while leaving the canvas its minimum. */
  private panelMax(): number {
    const layoutWidth = this.content()?.nativeElement.getBoundingClientRect().width ?? 0;
    const fit = layoutWidth - BrowseViewComponent.DIVIDER_WIDTH - BrowseViewComponent.CANVAS_MIN;
    return Math.min(BrowseViewComponent.PANEL_MAX, Math.max(this.panelMin(), fit));
  }

  /**
   * Smallest the side panel may shrink to, measured from the live panel content
   * so it tracks the current thumbnail size and the sort control's real width.
   *
   * The floor is the larger of two needs, exactly as requested: room for a
   * single thumbnail column (so the user can always pick one image), and room
   * for the "Sort: [select]" control to sit on its own row. Whichever is wider
   * wins — tiny thumbnails floor on the sort control; thumbnails larger than the
   * sort control floor on the thumbnail. Falls back to {@link PANEL_FLOOR}
   * before the content has laid out.
   */
  private panelMin(): number {
    // Untracked: reachable from the constructor's settings effect, which must
    // not pick up the view query as a dependency (see setThumbnailRadius above).
    const root = untracked(this.content)?.nativeElement;
    if (!root) return BrowseViewComponent.PANEL_FLOOR;
    const current = this.panelWidth();

    let oneColumn = BrowseViewComponent.PANEL_FLOOR;
    const list = root.querySelector('.bsp-list') as HTMLElement | null;
    if (list) {
      const goal = parseFloat(getComputedStyle(list).getPropertyValue('--grid-goal-width')) || 80;
      const style = getComputedStyle(list);
      const padH = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
      const bounding = list.getBoundingClientRect().width;
      const scrollbar = Math.max(0, bounding - list.clientWidth);
      // Chrome between the panel edge and the grid element (0 today, but measured
      // so any future wrapper padding is accounted for automatically).
      const chrome = Math.max(0, current - bounding);
      oneColumn = Math.ceil(goal + padH + scrollbar + chrome) + 1;
    }

    let sortRow = 0;
    const sort = root.querySelector('.bsp-sort') as HTMLElement | null;
    if (sort) {
      const toolbar = sort.closest('.bsp-toolbar') as HTMLElement | null;
      let toolbarChromeH = 0;
      if (toolbar) {
        const tstyle = getComputedStyle(toolbar);
        const tpad = parseFloat(tstyle.paddingLeft) + parseFloat(tstyle.paddingRight);
        const panelChrome = Math.max(0, current - toolbar.getBoundingClientRect().width);
        toolbarChromeH = tpad + panelChrome;
      }
      sortRow = Math.ceil(sort.getBoundingClientRect().width + toolbarChromeH) + 1;
    }

    return Math.max(BrowseViewComponent.PANEL_FLOOR, oneColumn, sortRow);
  }

  // --- Side-panel divider drag (mirrors the Find view's panel dividers) ----

  onDividerMouseDown(event: MouseEvent): void {
    event.preventDefault();
    this.dragging = true;
    this.dragMin = this.panelMin();
    this.ngZone.runOutsideAngular(() => {
      document.addEventListener('mousemove', this.boundPanelMove);
      document.addEventListener('mouseup', this.boundPanelUp);
    });
  }

  private onPanelMouseMove(event: MouseEvent): void {
    const content = this.content();
    if (!this.dragging || !content) return;
    const rect = content.nativeElement.getBoundingClientRect();
    // The panel is on the right, so its width grows as the cursor moves left.
    const fit = rect.width - BrowseViewComponent.DIVIDER_WIDTH - BrowseViewComponent.CANVAS_MIN;
    const max = Math.min(BrowseViewComponent.PANEL_MAX, Math.max(this.dragMin, fit));
    const width = this.clamp(rect.right - event.clientX, this.dragMin, max);
    // `panelWidth` is a signal read in the template's `--browse-panel-width`
    // binding, so the `.set()` schedules CD on its own — no `ngZone.run`.
    this.panelWidth.set(width);
  }

  private onPanelMouseUp(): void {
    if (!this.dragging) return;
    this.dragging = false;
    document.removeEventListener('mousemove', this.boundPanelMove);
    document.removeEventListener('mouseup', this.boundPanelUp);
    // Pop tight to the column count the user dragged to: snap away any trailing
    // empty strip so releasing never leaves a ragged half-column, then re-clamp.
    // The snap only ever shrinks toward the current columns — it can't pop the
    // panel back out wider — and the dynamic floor keeps at least one column and
    // the sort control on screen.
    const side = this.content()?.nativeElement.querySelector('.browse-side') as HTMLElement | null;
    const snapped = side ? snapPanelWidthToGridColumns(side, this.panelWidth()) : null;
    if (snapped !== null) {
      this.panelWidth.set(this.clamp(snapped, this.panelMin(), this.panelMax()));
    }
    // The user now owns the width; block the settings round-trip from resetting it.
    this.panelWidthInitialized = true;
    this.settingsState.update({ browse_panel_width: Math.round(this.panelWidth()) }).subscribe();
  }

  // --- Docked-details divider drag (left side; mirrors the right divider) ---

  onDetailsDividerMouseDown(event: MouseEvent): void {
    event.preventDefault();
    this.draggingDetails = true;
    this.ngZone.runOutsideAngular(() => {
      document.addEventListener('mousemove', this.boundDetailsMove);
      document.addEventListener('mouseup', this.boundDetailsUp);
    });
  }

  /** Largest the details panel can grow to while leaving the canvas its minimum
   *  after both dividers and the right (selection) panel. Shared by both the
   *  panel↔canvas divider drag and the in-panel row divider drag. */
  private detailsMax(rect: DOMRect): number {
    const fit =
      rect.width -
      2 * BrowseViewComponent.DIVIDER_WIDTH -
      BrowseViewComponent.CANVAS_MIN -
      this.panelWidth();
    return Math.min(
      BrowseViewComponent.PANEL_MAX,
      Math.max(BrowseViewComponent.DETAILS_FLOOR, fit),
    );
  }

  private onDetailsMouseMove(event: MouseEvent): void {
    const content = this.content();
    if (!this.draggingDetails || !content) return;
    const rect = content.nativeElement.getBoundingClientRect();
    // The details panel is on the left, so its width grows as the cursor moves
    // right.
    const width = this.clamp(
      event.clientX - rect.left,
      BrowseViewComponent.DETAILS_FLOOR,
      this.detailsMax(rect),
    );
    this.detailsPanelWidth.set(width);
  }

  private onDetailsMouseUp(): void {
    if (!this.draggingDetails) return;
    this.draggingDetails = false;
    document.removeEventListener('mousemove', this.boundDetailsMove);
    document.removeEventListener('mouseup', this.boundDetailsUp);
    this.finishDetailsDrag();
  }

  /** Snap + persist the settled details-panel width. Shared by both details
   *  dividers on release. */
  private finishDetailsDrag(): void {
    // Pop tight to the column count the user dragged to: snap away any trailing
    // empty strip so releasing never leaves a ragged half-column (mirrors the
    // right panel's snap). The docked panel reports the minimum width that still
    // shows its current columns; it only ever shrinks, so clamp just guards the
    // floor.
    const snapped = this.binPopup()?.snappedPanelWidth() ?? null;
    if (snapped !== null) {
      this.detailsPanelWidth.set(
        this.clamp(snapped, BrowseViewComponent.DETAILS_FLOOR, this.detailsPanelWidth()),
      );
    }
    // The user now owns the width; block the settings round-trip from resetting it.
    this.detailsWidthInitialized = true;
    this.settingsState
      .update({ browse_details_panel_width: Math.round(this.detailsPanelWidth()) })
      .subscribe();
  }

  // --- Docked-details in-panel row divider drag ----------------------------
  // A horizontal divider inside the docked panel, between the focused-item /
  // metadata row and the member grid. It resizes the *panel width*, not a row
  // height: the focused item is square, so dragging the divider down (a taller
  // item) is the same as widening the panel. Mapping the vertical delta 1:1 onto
  // the panel↔canvas width lets the user grow the item's vertical space directly
  // instead of reaching for the side divider and reasoning about width→height.

  onDetailsRowDividerMouseDown(event: MouseEvent): void {
    event.preventDefault();
    this.draggingDetailsRow.set(true);
    this.detailsRowStartY = event.clientY;
    this.detailsRowStartWidth = this.detailsPanelWidth();
    this.ngZone.runOutsideAngular(() => {
      document.addEventListener('mousemove', this.boundDetailsRowMove);
      document.addEventListener('mouseup', this.boundDetailsRowUp);
    });
  }

  private onDetailsRowMouseMove(event: MouseEvent): void {
    const content = this.content();
    if (!this.draggingDetailsRow() || !content) return;
    const rect = content.nativeElement.getBoundingClientRect();
    // Down (positive dy) grows the square item's height, hence the panel width;
    // up shrinks it. 1:1 because the item's height tracks the panel width.
    const dy = event.clientY - this.detailsRowStartY;
    const width = this.clamp(
      this.detailsRowStartWidth + dy,
      BrowseViewComponent.DETAILS_FLOOR,
      this.detailsMax(rect),
    );
    this.detailsPanelWidth.set(width);
  }

  private onDetailsRowMouseUp(): void {
    if (!this.draggingDetailsRow()) return;
    this.draggingDetailsRow.set(false);
    document.removeEventListener('mousemove', this.boundDetailsRowMove);
    document.removeEventListener('mouseup', this.boundDetailsRowUp);
    this.finishDetailsDrag();
  }

  /** Toggle region-select mode (drag-to-marquee without holding Shift). */
  toggleMarqueeMode(): void {
    this.marqueeMode = !this.marqueeMode;
  }

  /** Whether the current projection has any signpost labels to show. Gates the
   *  signpost toggle button: with no labels the toggle would do nothing. */
  get hasLabels(): boolean {
    return this.labels().length > 0;
  }

  /** Toggle the region-signpost layer and persist the choice as the per-media
   *  ``browse_signposts`` setting, so the map comes back the same way. */
  toggleSignposts(): void {
    const next = !this.signposts();
    this.signposts.set(next);
    const mt = this.mediaType();
    if (!mt) return;
    const existing =
      (this.lastSettings?.browse_signposts as { [k: string]: boolean } | undefined) || {};
    const map = { ...existing, [mt]: next };
    if (this.lastSettings) {
      (this.lastSettings as Record<string, unknown>)['browse_signposts'] = map;
    }
    this.settingsState.update({ browse_signposts: map } as SettingsUpdate).subscribe();
  }

  /**
   * Fetch the region signposts for a freshly-ready projection, once per
   * ``projection_id``. Metas without ``has_labels`` (no labeler has run, or an
   * older server) clear the signs instead of fetching a guaranteed-empty list.
   * A failure just leaves the map unsigned — the signs are optional decoration,
   * never worth an error state.
   */
  private syncLabels(meta: ProjectionMeta): void {
    if (!meta.projection_id || meta.point_count === 0) return;
    if (meta.projection_id === this.labelsFetchedFor) return;
    this.labelsFetchedFor = meta.projection_id;
    this.labels.set([]);
    if (!meta.has_labels) return;
    this.projectionApi
      .getLabels(this.subset)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (resp) => {
          // Guard against a stale response landing after the projection moved on.
          if (resp.projection_id && resp.projection_id !== this.meta()?.projection_id) return;
          this.labels.set(resp.labels ?? []);
        },
        error: () => {
          /* leave the map unsigned */
        },
      });
  }

  /**
   * Keyboard shortcuts for the browse canvas: arrow keys pan, ``+``/``-`` zoom
   * (mirroring the on-screen buttons), and Ctrl/Cmd-A selects every bin fully in
   * view. Suppressed while the projection isn't ready, while the bin-details
   * popup is open (it has its own keys — see {@link BrowseBinPopupComponent}),
   * and while typing or behind a modal ({@link shortcutsBlocked}).
   */
  @HostListener('document:keydown', ['$event'])
  onKeydown(event: KeyboardEvent): void {
    if (this.status() !== 'ready' || !this.revealed() || this.contextMenuOpen) return;
    if (shortcutsBlocked()) return;

    // Ctrl/Cmd-A: fully select every bin whose silhouette sits entirely in view.
    if ((event.ctrlKey || event.metaKey) && !event.altKey && (event.key === 'a' || event.key === 'A')) {
      event.preventDefault();
      this.canvas()?.selectAllInView();
      return;
    }
    // The remaining shortcuts take no modifiers.
    if (event.ctrlKey || event.metaKey || event.altKey) return;

    switch (event.key) {
      case 'ArrowUp':
        event.preventDefault();
        this.canvas()?.panByKey(0, -1);
        break;
      case 'ArrowDown':
        event.preventDefault();
        this.canvas()?.panByKey(0, 1);
        break;
      case 'ArrowLeft':
        event.preventDefault();
        this.canvas()?.panByKey(-1, 0);
        break;
      case 'ArrowRight':
        event.preventDefault();
        this.canvas()?.panByKey(1, 0);
        break;
      case '+':
      case '=':
        event.preventDefault();
        this.zoomIn();
        break;
      case '-':
      case '_':
        event.preventDefault();
        this.zoomOut();
        break;
    }
  }

  /** Zoom in one step (narrower span, cells keep their display size). */
  zoomIn(): void {
    this.canvas()?.zoomBy(this.zoomButtonFactor);
  }

  /** Zoom out one step. */
  zoomOut(): void {
    this.canvas()?.zoomBy(1 / this.zoomButtonFactor);
  }

  /** Choose a zoom and pan so the current data just fits in view. */
  zoomToFit(): void {
    this.canvas()?.zoomToFit();
  }

  // --- Preview-audio volume (audio datasets only) -------------------------

  /** Live drag of the volume slider: update the level (and the playing preview,
   *  via the ``[volume]`` inputs) on every ``input`` without hammering the
   *  settings API — the write is deferred to {@link onVolumeCommit} on release. */
  onVolumeInput(event: Event): void {
    const v = this.clampVolume(+(event.target as HTMLInputElement).value);
    this.volume.set(v);
    if (v > 0) this.preMuteVolume = v;
  }

  /** Slider released (``change``): persist the settled level once. */
  onVolumeCommit(event: Event): void {
    const v = this.clampVolume(+(event.target as HTMLInputElement).value);
    this.setVolume(v);
  }

  /** Mute toggle: drop to 0 (remembering where we were) or restore the last
   *  non-zero level. Persists immediately since it's a discrete action. */
  toggleMute(): void {
    if (this.volume() > 0) {
      this.preMuteVolume = this.volume();
      this.setVolume(0);
    } else {
      this.setVolume(this.preMuteVolume > 0 ? this.preMuteVolume : 1);
    }
  }

  private setVolume(v: number): void {
    const clamped = this.clampVolume(v);
    this.volume.set(clamped);
    if (clamped > 0) this.preMuteVolume = clamped;
    this.settingsState.update({ volume: clamped }).subscribe();
  }

  private clampVolume(v: number): number {
    if (!Number.isFinite(v)) return this.volume();
    return Math.max(0, Math.min(1, v));
  }

  // --- Right-click bin popup ----------------------------------------------

  /** Right-click on the canvas: pop the bin's item list at the cursor (or,
   *  docked, show it in the left details panel) when it landed on a bin; clear
   *  the open details when it hit empty space. */
  onCanvasContextMenu(event: BrowseContextMenuEvent): void {
    if (event.members.length === 0) {
      this.dismissContextMenu();
      return;
    }
    this.contextMembers = event.members;
    this.contextRepId = event.repId;
    if (this.detailsDocked()) {
      // A right-click on a bin always wants the details visible, so it undoes a
      // previous dismissal (the panel's X on an empty panel) — the panel comes
      // back, showing this bin. It comes back *docked*: dismissing it never
      // changed the docked/floating choice.
      this.detailsPanelHidden.set(false);
      // Docked: the persistent left panel shows the bin — no floating window.
      // Keep the canvas's pinned enlarge (the canvas pinned it on right-click)
      // so the chosen bin stays enlarged while it's open in the panel, exactly
      // like the floating window. It's released on dismiss (the panel's X or an
      // empty-space right-click), or re-pinned when another bin is right-clicked.
      this.contextMenuOpen = false;
      return;
    }
    this.contextMenuX = event.clientX;
    this.contextMenuY = event.clientY;
    this.contextBounds = event.bounds;
    this.contextMenuOpen = true;
  }

  /**
   * The docked panel's X. With a bin showing it clears the bin (leaving the
   * empty panel and its hint); pressed on an already-empty panel there is
   * nothing left to clear, so it hides the panel itself and gives the width
   * back to the canvas. Right-clicking any bin brings it back
   * ({@link onCanvasContextMenu}).
   */
  onDetailsDismiss(): void {
    if (this.contextMembers.length === 0) {
      this.detailsPanelHidden.set(true);
      return;
    }
    this.dismissContextMenu();
  }

  dismissContextMenu(): void {
    this.contextMenuOpen = false;
    if (this.detailsDocked()) {
      // Docked: dismissal (the panel's X, or an empty-space right-click)
      // clears the shown bin; the panel itself stays, showing its empty hint.
      this.contextMembers = [];
      this.contextRepId = null;
    }
    // Release the canvas's pinned enlarge so live hover resumes on the bin now
    // under the cursor.
    this.canvas()?.unpinCell();
  }

  /** The floating window's dock button: remember the docked presentation for
   *  this media type; the open bin carries over into the panel (it reads the
   *  same ``contextMembers``). */
  onDockRequested(): void {
    this.contextMenuOpen = false;
    // Docking is an explicit ask for the panel, so it clears any earlier
    // dismissal — otherwise the window would vanish into a hidden panel.
    this.detailsPanelHidden.set(false);
    // Keep the bin pinned enlarged as it moves into the docked panel — docked
    // bins stay enlarged too now, so there's nothing to release here.
    this.persistBinDetailsDocked(true);
  }

  /** The docked panel's pop-out button: remember the floating presentation and
   *  re-open the current bin as a floating window over the canvas. */
  onPopOutRequested(): void {
    this.persistBinDetailsDocked(false);
    if (this.contextMembers.length === 0) return;
    // No summon point (the click was in the panel header) — anchor the window
    // over the upper-left of the canvas; its own clamping keeps it on-screen.
    const main = this.content()?.nativeElement.querySelector('.browse-main');
    const rect = main ? main.getBoundingClientRect() : null;
    this.contextBounds = rect;
    this.contextMenuX = rect ? rect.left + rect.width / 4 : 100;
    this.contextMenuY = rect ? rect.top + rect.height / 4 : 100;
    this.contextMenuOpen = true;
  }

  /** Persist the docked/floating choice per media type (``bin_details_docked``)
   *  and apply it locally, keeping the settings snapshot consistent. */
  private persistBinDetailsDocked(value: boolean): void {
    const mt = this.mediaType();
    if (!mt) return;
    this.detailsDocked.set(value);
    const existing =
      (this.lastSettings?.bin_details_docked as { [k: string]: boolean } | undefined) || {};
    const map = { ...existing, [mt]: value };
    if (this.lastSettings) {
      (this.lastSettings as Record<string, unknown>)['bin_details_docked'] = map;
    }
    this.settingsState.update({ bin_details_docked: map } as SettingsUpdate).subscribe();
  }

  /**
   * Issue the right build request for the current mode: a subset build (UMAP
   * over just the handed-off ids) when in subset mode, else the full-dataset
   * build.
   */
  private buildRequest() {
    return this.subset
      ? this.projectionApi.buildSubset(this.subsetIds)
      : this.projectionApi.build();
  }

  onBuild(): void {
    if (this.subset && this.subsetIds.length === 0) {
      // Nothing to rebuild (e.g. Retry after the handoff expired).
      this.status.set('error');
      this.errorMessage.set(
        'This map has expired. Re-run Find and click Browse to rebuild it.',
      );
      return;
    }
    this.enterBuilding();
    this.buildRequest()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (resp) => {
          if (resp.status === 'ready') {
            // Already built/persisted — re-read meta so the canvas renders.
            this.loadProjection();
            return;
          }
          this.pollBuildStatus();
        },
        error: (err) => {
          this.status.set('error');
          this.errorMessage.set(
            apiErrorMessage(err, 'Failed to build the map'),
          );
        },
      });
  }

  /**
   * Re-fit UMAP over the items currently on screen and replace the layout with
   * a fresh 2-D arrangement. Unlike the bin-shape toggle (which re-bins the
   * frozen layout) or the Remove-from-Good cull (which re-bins minus points),
   * this forces a new UMAP fit: useful after culling items from a subset (so
   * the survivors re-spread) or just to reshuffle into a new arrangement. In
   * full-dataset mode it re-projects every item and overwrites the persisted
   * canonical layout; in subset mode it re-projects the current ``subsetIds``.
   * The fit runs in the background, so the view shows build progress meanwhile.
   */
  onReproject(): void {
    if (this.subset && this.subsetIds.length === 0) return;
    // The same items come back in new positions, and selection is id-based —
    // arm the one-shot survive mark so the canvas keeps it when the fresh
    // projection id lands, instead of treating the re-fit as a new projection.
    this.selection.markSurviveProjectionChange();
    this.enterBuilding();
    const request$ = this.subset
      ? this.projectionApi.reprojectSubset(this.subsetIds)
      : this.projectionApi.reproject();
    request$.pipe(takeUntil(this.destroy$)).subscribe({
      next: (resp) => {
        if (resp.status === 'ready') {
          // Defensive: a forced build always re-fits, but re-read meta anyway.
          this.loadProjection();
          return;
        }
        this.pollBuildStatus();
      },
      error: (err) => {
        // Disarm: no new projection is coming, so the mark would otherwise
        // linger and wrongly exempt the next genuine projection change.
        this.selection.consumeSurviveProjectionChange();
        this.status.set('error');
        this.errorMessage.set(
          apiErrorMessage(err, 'Failed to build the map'),
        );
      },
    });
  }

  /**
   * Verify the hand-selected items in a Find-positives browse: mark them
   * good/bad in the detector's labels *and* verify them (Find-mode bulk votes
   * land in ``verified_ids``), so they leave the unverified set, then drop them
   * from this browse by re-binning the layout over the reduced id set. Both
   * "Verified Good" (*target* ``good``) and "Verified Bad" (*target* ``bad``)
   * route here. Subset mode only — wired via the selection panel's ``canVerify``
   * affordance, itself gated on ``subset``.
   */
  /**
   * The selection panel's tri-state checkbox asked to select everything in
   * view ([ ]/[-] → [x]). Forward it to the canvas, which owns the viewport;
   * this is the mouse equivalent of the ctrl-A shortcut.
   */
  onSelectAllInView(): void {
    this.canvas()?.selectAllInView();
  }

  onVerify(target: 'good' | 'bad'): void {
    const ids = this.selection.ids();
    if (ids.length === 0) return;
    this.mediasApi
      .voteBulk(ids, target)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => this.dropFromBrowse(ids, target),
        error: () =>
          this.toast.error({ message: 'Failed to verify the selected items.' }),
      });
  }

  /**
   * Drop *removedIds* from the browse after they've been verified as *target*
   * (good/bad). The remaining items keep their exact 2-D positions and bins —
   * the server re-bins the frozen layout in place (no UMAP re-fit), returning
   * the same ``projection_id`` with a bumped ``content_version`` so only the
   * tile cache refreshes. The viewport is left exactly where the user had it:
   * verifying may leave dead space where the culled items used to be, but a
   * surprise zoom-to-fit jump is more disruptive than that dead space. The
   * user can hit Zoom Fit themselves if they want to re-frame.
   */
  private dropFromBrowse(removedIds: number[], target: 'good' | 'bad'): void {
    const removed = new Set(removedIds);
    const remaining = this.subsetIds.filter((id) => !removed.has(id));
    this.selection.clear();
    const label = target === 'good' ? 'Verified Good' : 'Verified Bad';
    this.toast.success({
      message: `Marked ${removedIds.length} item${removedIds.length === 1 ? '' : 's'} ${label}.`,
    });
    this.subsetIds = remaining;
    if (remaining.length === 0) {
      // Nothing left to project; verifying emptied the browse. This is a
      // success, not an error — render the ``done`` state (green, with a
      // Back to Find button) rather than the red error state.
      this.status.set('done');
      return;
    }
    this.projectionApi
      .subsetRemove(removedIds)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (meta) => {
          // Leave the viewport where the user had it; don't yank the camera to
          // re-fit the survivors. They can hit Zoom Fit if they want that.
          this.applyMeta(meta);
        },
        error: () =>
          this.toast.error({
            message: 'Items were verified, but the browse view could not refresh.',
          }),
      });
  }

  /**
   * Return to wherever this browse was launched from: the Find view for a
   * subset (Find-positives) browse, or the dashboard for a full-dataset
   * browse.
   */
  back(): void {
    if (this.subset) {
      this.backToFind();
      return;
    }
    this.router.navigate(['/dashboard']);
  }

  /**
   * Return to the Find view this browse was launched from, preserving the
   * verifications: the flag tells the Find view to skip its automatic
   * re-scoring (which would re-promote the items it re-scores) and just show
   * the updated labels, with the verified items now in the right-panel pile.
   */
  private backToFind(): void {
    const datasetId = this.activeContext.datasetId;
    const detectorId = this.activeContext.modelId;
    this.browseSubset.markReturningToFind();
    if (datasetId && detectorId) {
      this.router.navigate(['/find', datasetId, detectorId]);
    } else {
      this.router.navigate(['/dashboard']);
    }
  }

  /**
   * Enter the ready state, mounting the canvas. When we arrive from a non-ready
   * status a *fresh* canvas is created, so drop the reveal cover and wait for
   * its {@link BrowseCanvasComponent.firstViewReady}; an in-place refresh of an
   * already-ready canvas (e.g. a subset cull re-binning in place) keeps the same
   * canvas, so leave the cover state alone or it would strand a working view
   * (the existing canvas won't re-emit the one-shot signal).
   */
  private enterReady(): void {
    if (this.status() !== 'ready') this.revealed.set(false);
    this.status.set('ready');
  }

  /** The canvas finished painting its opening view (fit + top-view thumbnails);
   *  uncover it. */
  onCanvasFirstView(): void {
    this.revealed.set(true);
  }

  /** Message on the pre-reveal cover: names thumbnails for media that paint
   *  them (the case this cover exists for), the projection otherwise. */
  get preloadMessage(): string {
    return this.mediaTypeCaps.usesThumbnails(this.mediaType()) ? 'Loading thumbnails…' : 'Loading map…';
  }

  /** Enter the building state with a cleared bar (a fresh build starts at 0). */
  private enterBuilding(): void {
    this.status.set('building');
    this.applyBuildProgress(null);
  }

  /** Mirror a meta's build progress onto the bar's signals. */
  private applyBuildProgress(meta: ProjectionMeta | null): void {
    this.buildProgress.set(meta?.current ?? 0);
    this.buildTotal.set(meta?.total ?? 0);
    this.buildMessage.set(meta?.message ?? '');
    this.buildStep.set(meta?.step ?? null);
    this.buildTotalSteps.set(meta?.total_steps ?? null);
    this.buildOverall.set(meta?.overall ?? null);
  }

  private loadProjection(): void {
    this.status.set('loading');
    this.polling = false;
    this.projectionApi
      .getMeta(this.subset)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (meta) => this.applyMeta(meta),
        error: (err) => {
          // The meta endpoint reports "idle" rather than 404/409, but treat a
          // missing projection defensively the same way: build it, don't ask.
          if (err.status === 404 || err.status === 409) {
            this.onBuild();
          } else {
            this.status.set('error');
            this.errorMessage.set(
              apiErrorMessage(err, 'Failed to build the map'),
            );
          }
        },
      });
  }

  /**
   * Route a freshly-fetched meta to the right state, auto-building if absent.
   */
  private applyMeta(meta: ProjectionMeta): void {
    this.meta.set(meta);
    if (meta.media_type) {
      this.mediaType.set(meta.media_type);
      this.applyBrowsePrefsForMediaType();
    }
    this.tileCache.setProjectionId(meta.projection_id);
    this.tileCache.setContentVersion(meta.content_version ?? 0);

    if (meta.point_count > 0) {
      this.syncLabels(meta);
      this.enterReady();
      return;
    }
    if (meta.status === 'error') {
      this.status.set('error');
      this.errorMessage.set(meta.error || 'Failed to build the map');
      return;
    }
    if (meta.status === 'building') {
      // A build is already in flight (e.g. started at ingest); track it.
      this.status.set('building');
      this.applyBuildProgress(meta);
      this.pollBuildStatus();
      return;
    }
    // status === "idle": no projection yet. Build it automatically.
    this.onBuild();
  }

  private pollBuildStatus(): void {
    if (this.polling) return;
    this.polling = true;
    this.pollErrors = 0;
    const poll = (): void => {
      this.projectionApi
        .getMeta(this.subset)
        .pipe(takeUntil(this.destroy$))
        .subscribe({
          next: (meta) => {
            this.pollErrors = 0;
            this.meta.set(meta);
            if (meta.media_type) {
              this.mediaType.set(meta.media_type);
              this.applyBrowsePrefsForMediaType();
            }
            this.tileCache.setProjectionId(meta.projection_id);
            if (meta.point_count > 0) {
              this.polling = false;
              this.syncLabels(meta);
              this.enterReady();
              return;
            }
            if (meta.status === 'error') {
              this.polling = false;
              this.status.set('error');
              this.errorMessage.set(meta.error || 'Failed to build the map');
              return;
            }
            this.applyBuildProgress(meta);
            this.pollTimer = setTimeout(poll, 1000);
          },
          error: () => {
            this.pollErrors += 1;
            // Give up after a run of failures rather than retrying forever.
            if (this.pollErrors >= BrowseViewComponent.MAX_POLL_ERRORS) {
              this.polling = false;
              this.status.set('error');
              this.errorMessage.set(
                'Lost contact with the server while building the map.',
              );
              return;
            }
            // Exponential backoff: 2s, 4s, 8s, … capped at 30s.
            const delay = Math.min(2000 * 2 ** (this.pollErrors - 1), 30000);
            this.pollTimer = setTimeout(poll, delay);
          },
        });
    };
    this.pollTimer = setTimeout(poll, 1000);
  }
}
