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
  status: 'loading' | 'building' | 'ready' | 'empty' | 'error' = 'loading';
  errorMessage = '';
  buildProgress = 0;
  buildTotal = 0;
  datasetName = '';

  private destroy$ = new Subject<void>();

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
    this.projectionApi
      .build()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: () => {
          this.pollBuildStatus();
        },
        error: (err) => {
          this.status = 'error';
          this.errorMessage = err?.error?.error || 'Failed to start projection build';
        },
      });
  }

  private loadProjection(): void {
    this.status = 'loading';
    this.projectionApi
      .getMeta()
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (meta) => {
          this.meta = meta;
          if (meta.media_type) this.mediaType = meta.media_type;
          this.tileCache.setProjectionId(meta.projection_id);
          this.status = meta.point_count > 0 ? 'ready' : 'empty';
        },
        error: (err) => {
          if (err.status === 404 || err.status === 409) {
            this.status = 'empty';
          } else {
            this.status = 'error';
            this.errorMessage = err?.error?.error || 'Failed to load projection';
          }
        },
      });
  }

  private pollBuildStatus(): void {
    const poll = (): void => {
      this.projectionApi
        .getMeta()
        .pipe(takeUntil(this.destroy$))
        .subscribe({
          next: (meta) => {
            this.meta = meta;
            if (meta.media_type) this.mediaType = meta.media_type;
            this.tileCache.setProjectionId(meta.projection_id);
            this.status = meta.point_count > 0 ? 'ready' : 'empty';
          },
          error: () => {
            setTimeout(poll, 2000);
          },
        });
    };
    setTimeout(poll, 3000);
  }
}
