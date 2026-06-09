import { Component, ElementRef, OnInit, OnDestroy, NgZone, ViewChild } from '@angular/core';
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
import { ProjectionApiService } from '../../services/projection-api.service';
import { TileCacheService } from '../../services/tile-cache.service';
import { ActiveContextService } from '../../services/active-context.service';
import { DatasetsRegistryApiService } from '../../services/datasets-registry-api.service';
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
import type { BinShape, ProjectionMeta } from '../../models/projection.models';
import type { AppSettings } from '../../generated/api-client/models/app-settings';
import type { SettingsUpdate } from '../../generated/api-client/models/settings-update';

@Component({
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
  ],
  // Scoped per browse view so the canvas, its minimap, and the selection panel
  // share one viewport channel and one selection set without leaking across
  // other instances of the view.
  providers: [BrowseViewportService, BrowseSelectionService],
  templateUrl: './browse-view.component.html',
  styleUrl: './browse-view.component.scss',
})
export class BrowseViewComponent implements OnInit, OnDestroy {
  @ViewChild(BrowseCanvasComponent) private canvas?: BrowseCanvasComponent;
  /** The 3-column grid (canvas | divider | side panel); the divider drag
   *  measures against its box. Only present in the ``ready`` state. */
  @ViewChild('content') private content?: ElementRef<HTMLElement>;

  meta: ProjectionMeta | null = null;
  mediaType = '';
  hoverEvent: HexHoverEvent | null = null;
  /**
   * Top of the density scale for the legend: the densest cell currently in
   * view, pushed up from the canvas. The legend labels the yellow end with it.
   */
  densityMax = 1;
  status: 'loading' | 'building' | 'ready' | 'error' = 'loading';
  errorMessage = '';
  buildProgress = 0;
  buildTotal = 0;
  buildMessage = '';
  datasetName = '';

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
  private readonly HEX_SCALES = [0.5, 0.75, 1, 1.6, 2.5, 4, 6.25];
  /** Named icon sizes, index-aligned with {@link HEX_SCALES}. */
  static readonly ICON_SIZES = ['XS', 'S', 'M', 'L', 'XL', 'XXL', 'XXXL'] as const;
  hexScaleIndex = 2;

  /**
   * Density colormap preset for the flat (non-thumbnail) shading, mirrored
   * from the per-media ``browse_colormap`` setting and passed to the canvas
   * and minimap. ``auto`` follows the theme (Ocean in light, Heat in dark).
   */
  colormap: BrowseColormapId = 'auto';

  /**
   * Width (CSS px) of the colormap-coloured border drawn around multi-item
   * pile thumbnails, mirrored from the per-media ``browse_thumbnail_border``
   * setting and passed to the canvas. Only image/video datasets paint
   * thumbnails, so it has no visible effect for other media types.
   */
  thumbnailBorder = DEFAULT_THUMBNAIL_BORDER;

  /** Last settings snapshot, kept so per-media browser prefs can be
   *  re-resolved when the active media type becomes known after load. */
  private lastSettings: AppSettings | null = null;

  /**
   * Which lattice the projection is tiled with. Mirrored from the persisted
   * ``browse_bin_shape`` setting and flipped by the hex/square toggle. Switching
   * re-bins the (shared, frozen) UMAP layout — it never re-fits UMAP — and keeps
   * the canvas mounted so pan/zoom survive the switch.
   */
  binShape: BinShape = 'hex';

  /**
   * Region-select mode: the GUI parallel to the Shift+drag hotkey. When on, a
   * plain left-drag on the canvas rubber-bands a selection marquee instead of
   * panning. The toolbar button surfaces (and toggles) it so the gesture is
   * discoverable; Shift+drag keeps working whether or not this is on. Not
   * persisted — it's a transient interaction mode, reset on each visit.
   */
  marqueeMode = false;

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
  panelWidth = 300;
  /** Clamp + divider geometry, mirroring the Find view's panel dividers. */
  private static readonly PANEL_MIN = 260;
  private static readonly PANEL_MAX = 800;
  private static readonly CANVAS_MIN = 200;
  private static readonly DIVIDER_WIDTH = 8;
  private dragging = false;
  private boundPanelMove = this.onPanelMouseMove.bind(this);
  private boundPanelUp = this.onPanelMouseUp.bind(this);

  /**
   * Per-click zoom step for the on-screen +/- buttons. Larger than the wheel's
   * 1.15 per-tick factor so a single click makes a visible difference; button
   * zoom anchors at the viewport centre (no cursor to zoom toward).
   */
  private readonly ZOOM_BUTTON_FACTOR = 1.4;

  /** Bin-popup state: open flag, the viewport anchor it opens at, and the
   *  member ids of the bin the user right-clicked. Right-clicking a bin pops a
   *  scrollable list of its items (hear on hover, click to select) instead of an
   *  action menu; right-clicking empty space closes it. */
  contextMenuOpen = false;
  contextMenuX = 0;
  contextMenuY = 0;
  contextMembers: number[] = [];
  /** Canvas bounds the popup clamps inside, so it stays on the canvas rather
   *  than spilling onto the side panel or past the canvas edges. */
  contextBounds: DOMRect | null = null;

  private destroy$ = new Subject<void>();
  private polling = false;
  private pollTimer: ReturnType<typeof setTimeout> | null = null;
  private pollErrors = 0;
  private static readonly MAX_POLL_ERRORS = 5;

  constructor(
    private projectionApi: ProjectionApiService,
    private tileCache: TileCacheService,
    private activeContext: ActiveContextService,
    private datasetsRegistryApi: DatasetsRegistryApiService,
    private settingsState: SettingsStateService,
    private route: ActivatedRoute,
    private router: Router,
    private browseSubset: BrowseSubsetService,
    private selection: BrowseSelectionService,
    private mediasApi: MediasApiService,
    private dialog: VtDialogService,
    private toast: ToastService,
    private ngZone: NgZone,
  ) {}

  ngOnInit(): void {
    this.settingsState.settings$.pipe(takeUntil(this.destroy$)).subscribe((settings) => {
      if (!settings) return;
      this.lastSettings = settings;
      if (settings.browse_panel_width != null) {
        this.panelWidth = this.clamp(
          settings.browse_panel_width,
          BrowseViewComponent.PANEL_MIN,
          BrowseViewComponent.PANEL_MAX,
        );
      }
      this.applyBrowsePrefsForMediaType();
    });
    this.settingsState.load();

    // Subset mode: the Find view handed off a set of positive ids to project
    // on their own. Detect it from the query param + the in-memory handoff.
    this.subset = this.route.snapshot.queryParamMap.get('subset') === '1';
    if (this.subset) {
      const handoff = this.browseSubset.take();
      if (handoff && handoff.ids.length > 0) {
        this.subsetIds = handoff.ids;
        this.datasetName = handoff.label;
      } else {
        // No handoff (e.g. a hard reload): the ephemeral subset is gone.
        this.status = 'error';
        this.errorMessage =
          'This subset projection has expired. Re-run Find and click Browse to rebuild it.';
        this.tileCache.setBinShape(this.binShape);
        return;
      }
    }
    this.tileCache.setSubset(this.subset);
    this.tileCache.setBinShape(this.binShape);

    this.datasetsRegistryApi
      .getStatus()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (status) => {
          if (!this.subset) this.datasetName = status.display_name || '';
          this.mediaType = status.media_type || '';
          this.applyBrowsePrefsForMediaType();
        },
      });

    this.loadProjection();

    // The full-dataset projection re-resolves when the active pair changes via
    // the top bar. A subset projection is tied to the ids that produced it, so
    // ignore pair changes in subset mode.
    if (!this.subset) {
      this.activeContext.pair$
        .pipe(takeUntil(this.destroy$))
        .subscribe(() => {
          this.loadProjection();
        });
    }
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
    if (this.pollTimer) clearTimeout(this.pollTimer);
    document.removeEventListener('mousemove', this.boundPanelMove);
    document.removeEventListener('mouseup', this.boundPanelUp);
    this.tileCache.clear();
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
    return this.hexScaleIndex === 0;
  }

  get atMaxHexSize(): boolean {
    return this.hexScaleIndex === this.HEX_SCALES.length - 1;
  }

  /** Target bin radius (CSS px) for the current named thumbnail size. */
  private get thumbnailRadius(): number {
    return BrowseCanvasComponent.DEFAULT_TARGET_RADIUS * this.HEX_SCALES[this.hexScaleIndex];
  }

  /** Grow (+1) or shrink (-1) the on-screen thumbnail size, clamped to the
   *  range, and persist the new size as the per-media ``browse_icon_size``.
   *  Reframes the canvas so the same bins render bigger/smaller (the region
   *  shrinks/grows) without changing the binning. */
  bumpHexSize(delta: 1 | -1): void {
    const next = Math.max(
      0,
      Math.min(this.HEX_SCALES.length - 1, this.hexScaleIndex + delta),
    );
    if (next === this.hexScaleIndex) return;
    this.hexScaleIndex = next;
    this.canvas?.setThumbnailRadius(this.thumbnailRadius, true);
    this.persistBrowsePref('browse_icon_size', BrowseViewComponent.ICON_SIZES[next]);
  }

  /**
   * Re-resolve the per-media browser preferences (bin shape, colormap, icon
   * size) for the active media type from the last settings snapshot. Called
   * when settings arrive and again once the media type is known (it lands
   * after the initial settings push), so a saved-per-type choice is applied
   * even though the two facts arrive out of order.
   */
  private applyBrowsePrefsForMediaType(): void {
    const s = this.lastSettings;
    if (!s) return;
    const mt = this.mediaType;

    // Thumbnail media (image/video) are pinned to grayscale so the colourful
    // density presets never tint real thumbnails; the saved per-type value is
    // ignored for them (the Settings UI hides the picker to match).
    if (usesThumbnails(mt)) {
      this.colormap = 'gray';
    } else {
      const cmap = mt ? this.perMediaValue(s.browse_colormap, mt) : '';
      this.colormap =
        cmap && (BROWSE_COLORMAP_IDS as readonly string[]).includes(cmap)
          ? (cmap as BrowseColormapId)
          : 'auto';
    }

    const sizeLabel = mt ? this.perMediaValue(s.browse_icon_size, mt) : '';
    const sizeIdx = (BrowseViewComponent.ICON_SIZES as readonly string[]).indexOf(sizeLabel);
    this.hexScaleIndex = sizeIdx >= 0 ? sizeIdx : 2;
    // Seed the saved size as the overview granularity (no reframe): a settings
    // change re-bins at the current framing rather than zooming the viewport,
    // and on first load the initial fit picks the matching level.
    this.canvas?.setThumbnailRadius(this.thumbnailRadius, false);

    const shape: BinShape = this.perMediaValue(s.browse_bin_shape, mt) === 'square' ? 'square' : 'hex';
    if (shape !== this.binShape) this.switchBinShape(shape, false);

    const borderMap = s.browse_thumbnail_border as { [key: string]: number } | undefined;
    const rawBorder = mt && borderMap ? borderMap[mt] : undefined;
    this.thumbnailBorder =
      rawBorder == null
        ? DEFAULT_THUMBNAIL_BORDER
        : Math.max(0, Math.min(MAX_THUMBNAIL_BORDER, rawBorder));
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
    key: 'browse_bin_shape' | 'browse_colormap' | 'browse_icon_size',
    value: string,
  ): void {
    const mt = this.mediaType;
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
    return Math.min(BrowseViewComponent.PANEL_MAX, Math.max(BrowseViewComponent.PANEL_MIN, fit));
  }

  // --- Side-panel divider drag (mirrors the Find view's panel dividers) ----

  onDividerMouseDown(event: MouseEvent): void {
    event.preventDefault();
    this.dragging = true;
    this.ngZone.runOutsideAngular(() => {
      document.addEventListener('mousemove', this.boundPanelMove);
      document.addEventListener('mouseup', this.boundPanelUp);
    });
  }

  private onPanelMouseMove(event: MouseEvent): void {
    if (!this.dragging || !this.content) return;
    const rect = this.content.nativeElement.getBoundingClientRect();
    // The panel is on the right, so its width grows as the cursor moves left.
    const width = this.clamp(rect.right - event.clientX, BrowseViewComponent.PANEL_MIN, this.panelMax());
    this.ngZone.run(() => {
      this.panelWidth = width;
    });
  }

  private onPanelMouseUp(): void {
    if (!this.dragging) return;
    this.dragging = false;
    document.removeEventListener('mousemove', this.boundPanelMove);
    document.removeEventListener('mouseup', this.boundPanelUp);
    this.settingsState.update({ browse_panel_width: Math.round(this.panelWidth) }).subscribe();
  }

  /** Switch the bin shape from the toggle, persisting the choice. */
  setBinShape(shape: BinShape): void {
    this.switchBinShape(shape, true);
  }

  /**
   * Re-resolve the projection for *shape*. When a projection is already on
   * screen this re-bins in place (canvas stays mounted, pan/zoom preserved);
   * otherwise it falls back to the normal load path. *persist* writes the
   * choice to settings (true for the toggle, false when mirroring settings).
   */
  private switchBinShape(shape: BinShape, persist: boolean): void {
    if (shape === this.binShape) return;
    this.binShape = shape;
    this.tileCache.setBinShape(shape);
    if (persist) this.persistBrowsePref('browse_bin_shape', shape);
    if (this.status === 'ready') {
      this.ensureShape();
    } else {
      this.loadProjection();
    }
  }

  /**
   * Ensure the current bin shape's pyramid exists, then swap in its meta
   * without leaving the ``ready`` state — so the canvas is never torn down and
   * the user's pan/zoom carry across the toggle. The shared UMAP layout is
   * reused, so the build call returns ready after a quick re-bin.
   */
  private ensureShape(): void {
    this.buildRequest()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (resp) => {
          if (resp.status === 'ready') {
            this.projectionApi
              .getMeta(this.binShape, this.subset)
              .pipe(takeUntil(this.destroy$))
              .subscribe({
                next: (meta) => this.applyMeta(meta),
                error: () => this.loadProjection(),
              });
          } else {
            // Rare: this shape needs a full UMAP fit (no shared layout yet).
            this.status = 'building';
            this.buildProgress = 0;
            this.buildTotal = 0;
            this.buildMessage = '';
            this.pollBuildStatus();
          }
        },
        error: (err) => {
          this.status = 'error';
          this.errorMessage =
            err?.error?.message || err?.error?.error || 'Failed to switch bin shape';
        },
      });
  }

  /** Toggle region-select mode (drag-to-marquee without holding Shift). */
  toggleMarqueeMode(): void {
    this.marqueeMode = !this.marqueeMode;
  }

  /** Zoom in one step (narrower span, cells keep their display size). */
  zoomIn(): void {
    this.canvas?.zoomBy(this.ZOOM_BUTTON_FACTOR);
  }

  /** Zoom out one step. */
  zoomOut(): void {
    this.canvas?.zoomBy(1 / this.ZOOM_BUTTON_FACTOR);
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
      ? this.projectionApi.buildSubset(this.binShape, this.subsetIds)
      : this.projectionApi.build(this.binShape);
  }

  onBuild(): void {
    if (this.subset && this.subsetIds.length === 0) {
      // Nothing to rebuild (e.g. Retry after the handoff expired).
      this.status = 'error';
      this.errorMessage =
        'This subset projection has expired. Re-run Find and click Browse to rebuild it.';
      return;
    }
    this.status = 'building';
    this.buildProgress = 0;
    this.buildTotal = 0;
    this.buildMessage = '';
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
          this.status = 'error';
          this.errorMessage =
            err?.error?.message || err?.error?.error || 'Failed to start projection build';
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
    this.status = 'building';
    this.buildProgress = 0;
    this.buildTotal = 0;
    this.buildMessage = '';
    const request$ = this.subset
      ? this.projectionApi.reprojectSubset(this.binShape, this.subsetIds)
      : this.projectionApi.reproject(this.binShape);
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
        this.status = 'error';
        this.errorMessage =
          err?.error?.message || err?.error?.error || 'Failed to start re-projection';
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
      // Nothing left to project; verifying emptied the browse.
      this.status = 'error';
      this.errorMessage = 'All items were verified. Go back to Find to see your results.';
      return;
    }
    this.projectionApi
      .subsetRemove(this.binShape, removedIds)
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

  private loadProjection(): void {
    this.status = 'loading';
    this.polling = false;
    this.projectionApi
      .getMeta(this.binShape, this.subset)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (meta) => this.applyMeta(meta),
        error: (err) => {
          // The meta endpoint reports "idle" rather than 404/409, but treat a
          // missing projection defensively the same way: build it, don't ask.
          if (err.status === 404 || err.status === 409) {
            this.onBuild();
          } else {
            this.status = 'error';
            this.errorMessage =
              err?.error?.message || err?.error?.error || 'Failed to load projection';
          }
        },
      });
  }

  /** Route a freshly-fetched meta to the right state, auto-building if absent. */
  private applyMeta(meta: ProjectionMeta): void {
    this.meta = meta;
    if (meta.media_type) {
      this.mediaType = meta.media_type;
      this.applyBrowsePrefsForMediaType();
    }
    this.tileCache.setProjectionId(meta.projection_id);
    this.tileCache.setContentVersion(meta.content_version ?? 0);

    if (meta.point_count > 0) {
      this.status = 'ready';
      return;
    }
    if (meta.status === 'error') {
      this.status = 'error';
      this.errorMessage = meta.error || 'Projection build failed';
      return;
    }
    if (meta.status === 'building') {
      // A build is already in flight (e.g. started at ingest); track it.
      this.status = 'building';
      this.buildProgress = meta.current ?? 0;
      this.buildTotal = meta.total ?? 0;
      this.buildMessage = meta.message ?? '';
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
        .getMeta(this.binShape, this.subset)
        .pipe(takeUntil(this.destroy$))
        .subscribe({
          next: (meta) => {
            this.pollErrors = 0;
            this.meta = meta;
            if (meta.media_type) {
              this.mediaType = meta.media_type;
              this.applyBrowsePrefsForMediaType();
            }
            this.tileCache.setProjectionId(meta.projection_id);
            if (meta.point_count > 0) {
              this.polling = false;
              this.status = 'ready';
              return;
            }
            if (meta.status === 'error') {
              this.polling = false;
              this.status = 'error';
              this.errorMessage = meta.error || 'Projection build failed';
              return;
            }
            this.buildProgress = meta.current ?? 0;
            this.buildTotal = meta.total ?? 0;
            this.buildMessage = meta.message ?? '';
            this.pollTimer = setTimeout(poll, 1000);
          },
          error: () => {
            this.pollErrors += 1;
            // Give up after a run of failures rather than retrying forever.
            if (this.pollErrors >= BrowseViewComponent.MAX_POLL_ERRORS) {
              this.polling = false;
              this.status = 'error';
              this.errorMessage = 'Lost contact with the server while building the projection.';
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
