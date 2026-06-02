import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subject } from 'rxjs';
import { takeUntil } from 'rxjs/operators';
import { BrowseCanvasComponent, HexHoverEvent } from '../browse-canvas/browse-canvas.component';
import { BrowseHoverPreviewComponent } from '../browse-hover-preview/browse-hover-preview.component';
import { ProgressBarComponent } from '../progress-bar/progress-bar.component';
import { ProjectionApiService } from '../../services/projection-api.service';
import { TileCacheService } from '../../services/tile-cache.service';
import { ActiveContextService } from '../../services/active-context.service';
import { DatasetsRegistryApiService } from '../../services/datasets-registry-api.service';
import type { ProjectionMeta } from '../../models/projection.models';

@Component({
  selector: 'vt-browse-view',
  standalone: true,
  imports: [CommonModule, BrowseCanvasComponent, BrowseHoverPreviewComponent, ProgressBarComponent],
  templateUrl: './browse-view.component.html',
  styleUrl: './browse-view.component.scss',
})
export class BrowseViewComponent implements OnInit, OnDestroy {
  meta: ProjectionMeta | null = null;
  mediaType = '';
  hoverEvent: HexHoverEvent | null = null;
  status: 'loading' | 'building' | 'ready' | 'error' = 'loading';
  errorMessage = '';
  buildProgress = 0;
  buildTotal = 0;
  buildMessage = '';
  datasetName = '';

  private destroy$ = new Subject<void>();
  private polling = false;

  constructor(
    private projectionApi: ProjectionApiService,
    private tileCache: TileCacheService,
    private activeContext: ActiveContextService,
    private datasetsRegistryApi: DatasetsRegistryApiService,
  ) {}

  ngOnInit(): void {
    this.datasetsRegistryApi
      .getStatus()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (status) => {
          this.datasetName = status.display_name || '';
          this.mediaType = status.media_type || '';
        },
      });

    this.loadProjection();

    this.activeContext.pair$
      .pipe(takeUntil(this.destroy$))
      .subscribe(() => {
        this.loadProjection();
      });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
    this.tileCache.clear();
  }

  onHexHover(event: HexHoverEvent | null): void {
    this.hoverEvent = event;
  }

  onBuild(): void {
    this.status = 'building';
    this.buildProgress = 0;
    this.buildTotal = 0;
    this.buildMessage = '';
    this.projectionApi
      .build()
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
      .getMeta()
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
    const poll = (): void => {
      this.projectionApi
        .getMeta()
        .pipe(takeUntil(this.destroy$))
        .subscribe({
          next: (meta) => {
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
            setTimeout(poll, 1000);
          },
          error: () => {
            setTimeout(poll, 2000);
          },
        });
    };
    setTimeout(poll, 1000);
  }
}
