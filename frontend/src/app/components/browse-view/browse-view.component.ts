import { Component, OnInit, OnDestroy, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute } from '@angular/router';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { BrowseCanvasComponent, HexHoverEvent } from '../browse-canvas/browse-canvas.component';
import { BrowseHoverPreviewComponent } from '../browse-hover-preview/browse-hover-preview.component';
import {
  BrowseMinimapComponent,
  MINIMAP_MAX_HEIGHT,
  MINIMAP_MAX_WIDTH,
  MINIMAP_MIN_HEIGHT,
  MINIMAP_MIN_WIDTH,
} from '../browse-minimap/browse-minimap.component';
import { ProgressBarComponent } from '../progress-bar/progress-bar.component';
import { IconComponent } from '../icon/icon.component';
import { ProjectionApiService } from '../../services/projection-api.service';
import { TileCacheService } from '../../services/tile-cache.service';
import { ActiveContextService } from '../../services/active-context.service';
import { DatasetsRegistryApiService } from '../../services/datasets-registry-api.service';
import { SettingsStateService } from '../../services/settings-state.service';
import { BrowseViewportService } from '../../services/browse-viewport.service';
import { BrowseSubsetService } from '../../services/browse-subset.service';
import type { BinShape, ProjectionMeta } from '../../models/projection.models';

@Component({
  selector: 'vt-browse-view',
  standalone: true,
  imports: [
    CommonModule,
    BrowseCanvasComponent,
    BrowseHoverPreviewComponent,
    BrowseMinimapComponent,
    ProgressBarComponent,
    IconComponent,
  ],
  // Scoped per browse view so the canvas and its minimap share one viewport
  // channel without leaking across other instances of the view.
  providers: [BrowseViewportService],
  templateUrl: './browse-view.component.html',
  styleUrl: './browse-view.component.scss',
})
export class BrowseViewComponent implements OnInit, OnDestroy {
  @ViewChild(BrowseCanvasComponent) private canvas?: BrowseCanvasComponent;

  meta: ProjectionMeta | null = null;
  mediaType = '';
  hoverEvent: HexHoverEvent | null = null;
  status: 'loading' | 'building' | 'ready' | 'error' = 'loading';
  errorMessage = '';
  buildProgress = 0;
  buildTotal = 0;
  buildMessage = '';
  datasetName = '';

  /**
   * Discrete on-screen size multipliers for the hexes. ``1`` (index 2) is the
   * default fit; the bigger/smaller buttons step through these. This only
   * rescales the rendering — it never changes which vectors land in a hex.
   */
  private readonly HEX_SCALES = [0.5, 0.7, 1, 1.5, 2.2, 3];
  hexScaleIndex = 2;

  /**
   * Which lattice the projection is tiled with. Mirrored from the persisted
   * ``browse_bin_shape`` setting and flipped by the hex/square toggle. Switching
   * re-bins the (shared, frozen) UMAP layout — it never re-fits UMAP — and keeps
   * the canvas mounted so pan/zoom survive the switch.
   */
  binShape: BinShape = 'hex';

  /**
   * Subset mode: browse an ephemeral UMAP fit over just a handful of media
   * (the positives of a Find run) instead of the full dataset. Set from the
   * `?subset=1` query param plus a handoff from {@link BrowseSubsetService}.
   * `subsetIds` is kept on the component so re-resolving the projection (e.g.
   * a bin-shape switch) re-sends the same ids without a fresh handoff.
   */
  subset = false;
  subsetIds: number[] = [];

  /** Overview minimap show/hide + size, mirrored from the settings set. */
  minimapVisible = true;
  minimapWidth = 200;
  minimapHeight = 150;

  /**
   * Per-click zoom step for the on-screen +/- buttons. Larger than the wheel's
   * 1.15 per-tick factor so a single click makes a visible difference; button
   * zoom anchors at the viewport centre (no cursor to zoom toward).
   */
  private readonly ZOOM_BUTTON_FACTOR = 1.4;

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
    private browseSubset: BrowseSubsetService,
  ) {}

  ngOnInit(): void {
    this.settingsState.settings$.pipe(takeUntil(this.destroy$)).subscribe((settings) => {
      if (!settings) return;
      this.minimapVisible = settings.browse_minimap_visible !== false;
      if (settings.browse_minimap_width != null) {
        this.minimapWidth = this.clamp(
          settings.browse_minimap_width,
          MINIMAP_MIN_WIDTH,
          MINIMAP_MAX_WIDTH,
        );
      }
      if (settings.browse_minimap_height != null) {
        this.minimapHeight = this.clamp(
          settings.browse_minimap_height,
          MINIMAP_MIN_HEIGHT,
          MINIMAP_MAX_HEIGHT,
        );
      }
      const shape: BinShape = settings.browse_bin_shape === 'square' ? 'square' : 'hex';
      if (shape !== this.binShape) this.switchBinShape(shape, false);
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
    this.tileCache.clear();
  }

  onHexHover(event: HexHoverEvent | null): void {
    this.hoverEvent = event;
  }

  get hexDisplayScale(): number {
    return this.HEX_SCALES[this.hexScaleIndex];
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

  /** Grow (+1) or shrink (-1) the on-screen hex size, clamped to the range. */
  bumpHexSize(delta: 1 | -1): void {
    this.hexScaleIndex = Math.max(
      0,
      Math.min(this.HEX_SCALES.length - 1, this.hexScaleIndex + delta),
    );
  }

  /** Toggle the overview minimap and persist the choice. */
  setMinimapVisible(visible: boolean): void {
    this.minimapVisible = visible;
    this.settingsState.update({ browse_minimap_visible: visible }).subscribe();
  }

  /** Persist the size the user dragged the minimap to. */
  onMinimapResized(size: { width: number; height: number }): void {
    this.minimapWidth = size.width;
    this.minimapHeight = size.height;
    this.settingsState
      .update({ browse_minimap_width: size.width, browse_minimap_height: size.height })
      .subscribe();
  }

  private clamp(value: number, lo: number, hi: number): number {
    return Math.max(lo, Math.min(hi, value));
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
    if (persist) this.settingsState.update({ browse_bin_shape: shape }).subscribe();
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

  /** Zoom in one step (narrower span, cells keep their display size). */
  zoomIn(): void {
    this.canvas?.zoomBy(this.ZOOM_BUTTON_FACTOR);
  }

  /** Zoom out one step. */
  zoomOut(): void {
    this.canvas?.zoomBy(1 / this.ZOOM_BUTTON_FACTOR);
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
    if (meta.media_type) this.mediaType = meta.media_type;
    this.tileCache.setProjectionId(meta.projection_id);

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
            if (meta.media_type) this.mediaType = meta.media_type;
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
