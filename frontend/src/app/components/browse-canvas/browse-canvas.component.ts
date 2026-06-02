import {
  Component,
  ElementRef,
  EventEmitter,
  Input,
  NgZone,
  OnChanges,
  OnDestroy,
  OnInit,
  Output,
  SimpleChanges,
  ViewChild,
} from '@angular/core';
import { Subscription } from 'rxjs';
import { TileCacheService } from '../../services/tile-cache.service';
import type {
  HexCellPayload,
  ProjectionMeta,
  TilePayload,
  ViewTransform,
} from '../../models/projection.models';

const SQRT3 = Math.sqrt(3);
const DEG30 = Math.PI / 6;
const HEX_ANGLES = Array.from({ length: 6 }, (_, i) => (Math.PI / 3) * i - DEG30);

const VIRIDIS: [number, number, number][] = [
  [68, 1, 84],
  [72, 35, 116],
  [64, 67, 135],
  [52, 94, 141],
  [41, 120, 142],
  [33, 145, 140],
  [42, 168, 131],
  [68, 190, 112],
  [94, 201, 98],
  [128, 213, 79],
  [166, 222, 52],
  [199, 227, 33],
  [229, 228, 32],
  [253, 231, 37],
];

export interface HexHoverEvent {
  cell: HexCellPayload;
  screenX: number;
  screenY: number;
}

@Component({
  selector: 'vt-browse-canvas',
  standalone: true,
  templateUrl: './browse-canvas.component.html',
  styleUrl: './browse-canvas.component.scss',
})
export class BrowseCanvasComponent implements OnInit, OnChanges, OnDestroy {
  @ViewChild('canvas', { static: true }) canvasRef!: ElementRef<HTMLCanvasElement>;
  @Input() meta: ProjectionMeta | null = null;
  @Output() hexHover = new EventEmitter<HexHoverEvent | null>();

  private ctx!: CanvasRenderingContext2D;
  private width = 0;
  private height = 0;
  private dpr = 1;

  private transform: ViewTransform = { centerX: 0, centerY: 0, zoom: 1 };
  private activeLevel = 0;
  private maxCount = 1;

  private isPanning = false;
  private panStartX = 0;
  private panStartY = 0;
  private panStartCenterX = 0;
  private panStartCenterY = 0;

  private hoveredCell: HexCellPayload | null = null;
  private hoverDebounceTimer: ReturnType<typeof setTimeout> | null = null;

  private tileLoadSub: Subscription | null = null;
  private rafId = 0;
  private needsRedraw = false;
  private resizeObserver: ResizeObserver | null = null;

  private boundMouseMove = this.onMouseMove.bind(this);
  private boundMouseUp = this.onMouseUp.bind(this);

  constructor(
    private ngZone: NgZone,
    private tileCache: TileCacheService,
  ) {}

  ngOnInit(): void {
    this.ctx = this.canvasRef.nativeElement.getContext('2d')!;

    this.tileLoadSub = this.tileCache.tileLoaded$.subscribe(() => {
      this.requestRedraw();
    });

    this.resizeObserver = new ResizeObserver(() => {
      this.ngZone.runOutsideAngular(() => this.resize());
    });
    this.resizeObserver.observe(this.canvasRef.nativeElement.parentElement!);

    this.ngZone.runOutsideAngular(() => {
      const el = this.canvasRef.nativeElement;
      el.addEventListener('mousedown', this.onMouseDown.bind(this));
      el.addEventListener('wheel', this.onWheel.bind(this), { passive: false });
      el.addEventListener('mousemove', this.onCanvasMouseMove.bind(this));
      el.addEventListener('mouseleave', this.onCanvasMouseLeave.bind(this));
    });
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['meta'] && this.meta) {
      this.tileCache.setProjectionId(this.meta.projection_id);
      this.fitToData();
      this.requestRedraw();
    }
  }

  ngOnDestroy(): void {
    this.tileLoadSub?.unsubscribe();
    this.resizeObserver?.disconnect();
    if (this.rafId) cancelAnimationFrame(this.rafId);
    if (this.hoverDebounceTimer) clearTimeout(this.hoverDebounceTimer);
    document.removeEventListener('mousemove', this.boundMouseMove);
    document.removeEventListener('mouseup', this.boundMouseUp);
  }

  private resize(): void {
    const el = this.canvasRef.nativeElement.parentElement!;
    const rect = el.getBoundingClientRect();
    this.dpr = window.devicePixelRatio || 1;
    this.width = rect.width;
    this.height = rect.height;
    const canvas = this.canvasRef.nativeElement;
    canvas.width = this.width * this.dpr;
    canvas.height = this.height * this.dpr;
    canvas.style.width = `${this.width}px`;
    canvas.style.height = `${this.height}px`;
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.requestRedraw();
  }

  private fitToData(): void {
    if (!this.meta || this.meta.point_count === 0) return;
    const [xmin, ymin, xmax, ymax] = this.meta.bounds;
    const dataW = xmax - xmin || 1;
    const dataH = ymax - ymin || 1;
    const padding = 0.1;
    const padW = dataW * (1 + padding * 2);
    const padH = dataH * (1 + padding * 2);
    const w = this.width || 800;
    const h = this.height || 600;
    this.transform.zoom = Math.min(w / padW, h / padH);
    this.transform.centerX = (xmin + xmax) / 2;
    this.transform.centerY = (ymin + ymax) / 2;
    this.updateActiveLevel();
  }

  private updateActiveLevel(): void {
    if (!this.meta || this.meta.levels.length === 0) return;
    const targetScreenRadius = 28;
    const idealLevel = Math.log2(
      (this.meta.base_radius * this.transform.zoom) / targetScreenRadius,
    );
    this.activeLevel = Math.max(
      0,
      Math.min(this.meta.levels.length - 1, Math.round(idealLevel)),
    );
  }

  private projToScreen(px: number, py: number): [number, number] {
    const sx = (px - this.transform.centerX) * this.transform.zoom + this.width / 2;
    const sy = (py - this.transform.centerY) * this.transform.zoom + this.height / 2;
    return [sx, sy];
  }

  private screenToProj(sx: number, sy: number): [number, number] {
    const px = (sx - this.width / 2) / this.transform.zoom + this.transform.centerX;
    const py = (sy - this.height / 2) / this.transform.zoom + this.transform.centerY;
    return [px, py];
  }

  private getVisibleBounds(): [number, number, number, number] {
    const [xmin, ymin] = this.screenToProj(0, 0);
    const [xmax, ymax] = this.screenToProj(this.width, this.height);
    return [
      Math.min(xmin, xmax),
      Math.min(ymin, ymax),
      Math.max(xmin, xmax),
      Math.max(ymin, ymax),
    ];
  }

  private getVisibleTiles(): { tx: number; ty: number }[] {
    if (!this.meta) return [];
    const level = this.activeLevel;
    const radius = this.meta.base_radius / Math.pow(2, level);
    const dx = radius * SQRT3;
    const dy = radius * 1.5;
    const tileSpan = this.meta.tile_span;
    const tileW = tileSpan * dx;
    const tileH = tileSpan * dy;

    const [vxmin, vymin, vxmax, vymax] = this.getVisibleBounds();
    const txMin = Math.floor(vxmin / tileW - 1);
    const txMax = Math.ceil(vxmax / tileW + 1);
    const tyMin = Math.floor(vymin / tileH - 1);
    const tyMax = Math.ceil(vymax / tileH + 1);

    const tiles: { tx: number; ty: number }[] = [];
    for (let tx = txMin; tx <= txMax; tx++) {
      for (let ty = tyMin; ty <= tyMax; ty++) {
        tiles.push({ tx, ty });
      }
    }
    return tiles;
  }

  private requestRedraw(): void {
    if (this.needsRedraw) return;
    this.needsRedraw = true;
    this.rafId = requestAnimationFrame(() => {
      this.needsRedraw = false;
      this.draw();
    });
  }

  private draw(): void {
    const ctx = this.ctx;
    if (!ctx || this.width === 0) return;

    ctx.clearRect(0, 0, this.width, this.height);
    ctx.fillStyle = this.themeColor('--bg-body');
    ctx.fillRect(0, 0, this.width, this.height);

    if (!this.meta || this.meta.point_count === 0) {
      ctx.fillStyle = this.themeColor('--text-muted');
      ctx.font = '16px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No projection data', this.width / 2, this.height / 2);
      return;
    }

    const level = this.activeLevel;
    const radius = this.meta.base_radius / Math.pow(2, level);
    const screenRadius = radius * this.transform.zoom;

    const visibleTiles = this.getVisibleTiles();
    let allCells: HexCellPayload[] = [];

    for (const { tx, ty } of visibleTiles) {
      const cached = this.tileCache.getCached(level, tx, ty);
      if (cached) {
        allCells = allCells.concat(cached.cells);
      } else {
        this.tileCache.getTile(level, tx, ty)?.subscribe();
      }
    }

    this.maxCount = 1;
    for (const cell of allCells) {
      if (cell.count > this.maxCount) this.maxCount = cell.count;
    }

    for (const cell of allCells) {
      const [sx, sy] = this.projToScreen(cell.cx, cell.cy);
      if (sx < -screenRadius * 2 || sx > this.width + screenRadius * 2) continue;
      if (sy < -screenRadius * 2 || sy > this.height + screenRadius * 2) continue;
      this.drawHex(ctx, sx, sy, screenRadius, cell);
    }

    this.prefetchNeighbors(visibleTiles);
  }

  private drawHex(
    ctx: CanvasRenderingContext2D,
    cx: number,
    cy: number,
    radius: number,
    cell: HexCellPayload,
  ): void {
    const t = Math.log(cell.count) / Math.log(this.maxCount || 2);
    const color = this.viridisColor(Math.max(0, Math.min(1, t)));

    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const x = cx + radius * Math.cos(HEX_ANGLES[i]);
      const y = cy + radius * Math.sin(HEX_ANGLES[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();

    const isHovered =
      this.hoveredCell && this.hoveredCell.q === cell.q && this.hoveredCell.r === cell.r;
    if (isHovered) {
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();
    } else {
      ctx.strokeStyle = this.themeColor('--bg-body');
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }
  }

  private viridisColor(t: number): string {
    const n = VIRIDIS.length - 1;
    const idx = t * n;
    const lo = Math.floor(idx);
    const hi = Math.min(lo + 1, n);
    const frac = idx - lo;
    const r = Math.round(VIRIDIS[lo][0] + (VIRIDIS[hi][0] - VIRIDIS[lo][0]) * frac);
    const g = Math.round(VIRIDIS[lo][1] + (VIRIDIS[hi][1] - VIRIDIS[lo][1]) * frac);
    const b = Math.round(VIRIDIS[lo][2] + (VIRIDIS[hi][2] - VIRIDIS[lo][2]) * frac);
    return `rgb(${r},${g},${b})`;
  }

  private themeColor(varName: string): string {
    return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
  }

  private prefetchNeighbors(visibleTiles: { tx: number; ty: number }[]): void {
    if (!this.meta) return;
    const level = this.activeLevel;
    const seen = new Set(visibleTiles.map((t) => `${t.tx}:${t.ty}`));
    for (const { tx, ty } of visibleTiles) {
      for (const [dx, dy] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
        const nx = tx + dx;
        const ny = ty + dy;
        if (!seen.has(`${nx}:${ny}`)) {
          this.tileCache.prefetch(level, nx, ny);
          seen.add(`${nx}:${ny}`);
        }
      }
    }
    if (level > 0) {
      this.prefetchLevel(level - 1, visibleTiles);
    }
    if (this.meta.levels.length > level + 1) {
      this.prefetchLevel(level + 1, visibleTiles);
    }
  }

  private prefetchLevel(targetLevel: number, sourceTiles: { tx: number; ty: number }[]): void {
    if (!this.meta) return;
    const sourceRadius = this.meta.base_radius / Math.pow(2, this.activeLevel);
    const targetRadius = this.meta.base_radius / Math.pow(2, targetLevel);
    const ratio = sourceRadius / targetRadius;
    const seen = new Set<string>();
    for (const { tx, ty } of sourceTiles) {
      const ttx = Math.floor(tx * ratio);
      const tty = Math.floor(ty * ratio);
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          const key = `${ttx + dx}:${tty + dy}`;
          if (!seen.has(key)) {
            seen.add(key);
            this.tileCache.prefetch(targetLevel, ttx + dx, tty + dy);
          }
        }
      }
    }
  }

  // --- Interaction handlers ---

  private onMouseDown(event: MouseEvent): void {
    if (event.button !== 0) return;
    this.isPanning = true;
    this.panStartX = event.clientX;
    this.panStartY = event.clientY;
    this.panStartCenterX = this.transform.centerX;
    this.panStartCenterY = this.transform.centerY;
    document.addEventListener('mousemove', this.boundMouseMove);
    document.addEventListener('mouseup', this.boundMouseUp);
  }

  private onMouseMove(event: MouseEvent): void {
    if (!this.isPanning) return;
    const dx = event.clientX - this.panStartX;
    const dy = event.clientY - this.panStartY;
    this.transform.centerX = this.panStartCenterX - dx / this.transform.zoom;
    this.transform.centerY = this.panStartCenterY - dy / this.transform.zoom;
    this.requestRedraw();
  }

  private onMouseUp(): void {
    this.isPanning = false;
    document.removeEventListener('mousemove', this.boundMouseMove);
    document.removeEventListener('mouseup', this.boundMouseUp);
  }

  private onWheel(event: WheelEvent): void {
    event.preventDefault();
    const rect = this.canvasRef.nativeElement.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;

    const [projX, projY] = this.screenToProj(mx, my);
    const factor = event.deltaY < 0 ? 1.15 : 1 / 1.15;
    const newZoom = Math.max(0.01, Math.min(100000, this.transform.zoom * factor));

    this.transform.centerX = projX - (mx - this.width / 2) / newZoom;
    this.transform.centerY = projY - (my - this.height / 2) / newZoom;
    this.transform.zoom = newZoom;

    this.updateActiveLevel();
    this.requestRedraw();
  }

  private onCanvasMouseMove(event: MouseEvent): void {
    if (this.isPanning) return;
    const rect = this.canvasRef.nativeElement.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;

    if (this.hoverDebounceTimer) clearTimeout(this.hoverDebounceTimer);

    this.hoverDebounceTimer = setTimeout(() => {
      const hit = this.hitTest(mx, my);
      const prevQ = this.hoveredCell?.q;
      const prevR = this.hoveredCell?.r;
      this.hoveredCell = hit;
      if (hit) {
        if (hit.q !== prevQ || hit.r !== prevR) {
          const [sx, sy] = this.projToScreen(hit.cx, hit.cy);
          this.ngZone.run(() => {
            this.hexHover.emit({
              cell: hit,
              screenX: sx + rect.left,
              screenY: sy + rect.top,
            });
          });
          this.requestRedraw();
        }
      } else if (prevQ != null) {
        this.ngZone.run(() => this.hexHover.emit(null));
        this.requestRedraw();
      }
    }, 30);
  }

  private onCanvasMouseLeave(): void {
    if (this.hoverDebounceTimer) clearTimeout(this.hoverDebounceTimer);
    if (this.hoveredCell) {
      this.hoveredCell = null;
      this.ngZone.run(() => this.hexHover.emit(null));
      this.requestRedraw();
    }
  }

  private hitTest(sx: number, sy: number): HexCellPayload | null {
    if (!this.meta) return null;
    const [px, py] = this.screenToProj(sx, sy);
    const level = this.activeLevel;
    const radius = this.meta.base_radius / Math.pow(2, level);
    const dx = radius * SQRT3;
    const dy = radius * 1.5;
    const tileSpan = this.meta.tile_span;

    const qEst = Math.round((px / dx) * 2);
    const rEst = Math.round(py / dy);
    const txEst = Math.floor((qEst / 2) / tileSpan);
    const tyEst = Math.floor(rEst / tileSpan);

    let best: HexCellPayload | null = null;
    let bestDist = Infinity;

    for (let dtx = -1; dtx <= 1; dtx++) {
      for (let dty = -1; dty <= 1; dty++) {
        const tile = this.tileCache.getCached(level, txEst + dtx, tyEst + dty);
        if (!tile) continue;
        for (const cell of tile.cells) {
          const cdx = cell.cx - px;
          const cdy = cell.cy - py;
          const dist = cdx * cdx + cdy * cdy;
          if (dist < bestDist) {
            bestDist = dist;
            best = cell;
          }
        }
      }
    }

    if (best && bestDist < radius * radius) {
      return best;
    }
    return null;
  }
}
