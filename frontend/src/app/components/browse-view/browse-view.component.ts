import { ChangeDetectionStrategy, Component, effect, ElementRef, HostListener, inject, NgZone, OnDestroy, OnInit, signal, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, Router } from '@angular/router';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import {
  BrowseCanvasComponent,
  BrowseContextMenuEvent,
  HexHoverEvent,
} from '../browse-canvas/browse-canvas.component';
import { BrowseHoverPreviewComponent } from '../browse-hover-preview/browse-hover-preview.component';
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
import {
  BROWSE_COLORMAP_IDS,
  DEFAULT_THUMBNAIL_BORDER,
  MAX_THUMBNAIL_BORDER,
  usesThumbnails,
  type BrowseColormapId,
} from '../browse-canvas/hex-render.util';
import type { ProjectionMeta } from '../../models/projection.models';
import { snapPanelWidthToGridColumns } from '../../utils/grid-icon-size';
import { shortcutsBlocked } from '../../utils/keyboard-shortcuts';
import type { AppSettings } from '../../generated/api-client/models/app-settings';
import type { SettingsUpdate } from '../../generated/api-client/models/settings-update';

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
  private ngZone = inject(NgZone);

  @ViewChild(BrowseCanvasComponent) private canvas?: BrowseCanvasComponent;
  /** The 3-column grid (canvas | divider | side panel); the divider drag
   *  measures against its box. Only present in the ``ready`` state. */
  @ViewChild('content') private content?: ElementRef<HTMLElement>;

  readonly meta = signal<ProjectionMeta | null>(null);
  // Written from raw/async callbacks (dataset-status subscribe, projection
  // build poller, settings effect), so a signal so those writes repaint the
  // canvas/minimap inputs and the info row under zoneless.
  readonly mediaType = signal('');
  hoverEvent: HexHoverEvent | null = null;
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
  readonly datasetName = signal('');

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
  /** Dynamic minimum snapshotted at drag start, so the per-move handler reuses
   *  it instead of re-measuring the DOM (and thrashing layout) on every move —
   *  the thumbnail size and sort-control width can't change mid-drag anyway. */
  private dragMin = BrowseViewComponent.PANEL_FLOOR;
  private boundPanelMove = this.onPanelMouseMove.bind(this);
  private boundPanelUp = this.onPanelMouseUp.bind(this);

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

    // Subset mode: the Find view handed off a set of positive ids to project
    // on their own. Detect it from the query param + the in-memory handoff.
    this.subset = this.route.snapshot.queryParamMap.get('subset') === '1';
    if (this.subset) {
      const handoff = this.browseSubset.take();
      if (handoff && handoff.ids.length > 0) {
        this.subsetIds = handoff.ids;
        this.datasetName.set(handoff.label);
      } else {
        // No handoff (e.g. a hard reload): the ephemeral subset is gone.
        this.status.set('error');
        this.errorMessage.set(
          'This subset projection has expired. Re-run Find and click Browse to rebuild it.',
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
          if (!this.subset) this.datasetName.set(status.display_name || '');
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
    this.canvas?.setThumbnailRadius(this.thumbnailRadius, true);
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
    if (usesThumbnails(mt)) {
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
    // and on first load the initial fit picks the matching level.
    this.canvas?.setThumbnailRadius(this.thumbnailRadius, false);

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
    const layoutWidth = this.content?.nativeElement.getBoundingClientRect().width ?? 0;
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
    const root = this.content?.nativeElement;
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
    if (!this.dragging || !this.content) return;
    const rect = this.content.nativeElement.getBoundingClientRect();
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
    const side = this.content?.nativeElement.querySelector('.browse-side') as HTMLElement | null;
    const snapped = side ? snapPanelWidthToGridColumns(side, this.panelWidth()) : null;
    if (snapped !== null) {
      this.panelWidth.set(this.clamp(snapped, this.panelMin(), this.panelMax()));
    }
    // The user now owns the width; block the settings round-trip from resetting it.
    this.panelWidthInitialized = true;
    this.settingsState.update({ browse_panel_width: Math.round(this.panelWidth()) }).subscribe();
  }

  /** Toggle region-select mode (drag-to-marquee without holding Shift). */
  toggleMarqueeMode(): void {
    this.marqueeMode = !this.marqueeMode;
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
      this.canvas?.selectAllInView();
      return;
    }
    // The remaining shortcuts take no modifiers.
    if (event.ctrlKey || event.metaKey || event.altKey) return;

    switch (event.key) {
      case 'ArrowUp':
        event.preventDefault();
        this.canvas?.panByKey(0, -1);
        break;
      case 'ArrowDown':
        event.preventDefault();
        this.canvas?.panByKey(0, 1);
        break;
      case 'ArrowLeft':
        event.preventDefault();
        this.canvas?.panByKey(-1, 0);
        break;
      case 'ArrowRight':
        event.preventDefault();
        this.canvas?.panByKey(1, 0);
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
    this.canvas?.zoomBy(this.zoomButtonFactor);
  }

  /** Zoom out one step. */
  zoomOut(): void {
    this.canvas?.zoomBy(1 / this.zoomButtonFactor);
  }

  /** Choose a zoom and pan so the current data just fits in view. */
  zoomToFit(): void {
    this.canvas?.zoomToFit();
  }

  // --- Right-click bin popup ----------------------------------------------

  /** Right-click on the canvas: pop the bin's item list at the cursor when it
   *  landed on a bin; close any open popup when it hit empty space. */
  onCanvasContextMenu(event: BrowseContextMenuEvent): void {
    if (event.members.length === 0) {
      this.dismissContextMenu();
      return;
    }
    this.contextMembers = event.members;
    this.contextRepId = event.repId;
    this.contextMenuX = event.clientX;
    this.contextMenuY = event.clientY;
    this.contextBounds = event.bounds;
    this.contextMenuOpen = true;
  }

  dismissContextMenu(): void {
    this.contextMenuOpen = false;
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
        'This subset projection has expired. Re-run Find and click Browse to rebuild it.',
      );
      return;
    }
    this.status.set('building');
    this.buildProgress.set(0);
    this.buildTotal.set(0);
    this.buildMessage.set('');
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
            err?.error?.message || err?.error?.error || 'Failed to start projection build',
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
    this.status.set('building');
    this.buildProgress.set(0);
    this.buildTotal.set(0);
    this.buildMessage.set('');
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
          err?.error?.message || err?.error?.error || 'Failed to start re-projection',
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
    return usesThumbnails(this.mediaType()) ? 'Loading thumbnails…' : 'Loading projection…';
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
              err?.error?.message || err?.error?.error || 'Failed to load projection',
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
      this.enterReady();
      return;
    }
    if (meta.status === 'error') {
      this.status.set('error');
      this.errorMessage.set(meta.error || 'Projection build failed');
      return;
    }
    if (meta.status === 'building') {
      // A build is already in flight (e.g. started at ingest); track it.
      this.status.set('building');
      this.buildProgress.set(meta.current ?? 0);
      this.buildTotal.set(meta.total ?? 0);
      this.buildMessage.set(meta.message ?? '');
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
              this.enterReady();
              return;
            }
            if (meta.status === 'error') {
              this.polling = false;
              this.status.set('error');
              this.errorMessage.set(meta.error || 'Projection build failed');
              return;
            }
            this.buildProgress.set(meta.current ?? 0);
            this.buildTotal.set(meta.total ?? 0);
            this.buildMessage.set(meta.message ?? '');
            this.pollTimer = setTimeout(poll, 1000);
          },
          error: () => {
            this.pollErrors += 1;
            // Give up after a run of failures rather than retrying forever.
            if (this.pollErrors >= BrowseViewComponent.MAX_POLL_ERRORS) {
              this.polling = false;
              this.status.set('error');
              this.errorMessage.set(
                'Lost contact with the server while building the projection.',
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
