import { Component, Input, OnDestroy, OnInit } from '@angular/core';
import {
  gradientStops,
  resolveColormap,
  rgbString,
  type BrowseColormapId,
  type CanvasTheme,
} from '../browse-canvas/hex-render.util';

interface LegendTick {
  /** Fractional height up the ramp, 0 (bottom) → 1 (top). */
  t: number;
  /** Item count this height maps to, formatted for display. */
  label: string;
}

/**
 * Vertical color key for the browse canvas. The canvas shades each cell by its
 * item count: a one-item cell gets the colormap's dedicated ``single`` colour
 * (drawn as a dot), and multi-item cells use the density ``ramp``, renormalized
 * to the densest cell currently in view — so the ramp's top is {@link maxCount}
 * items, log-spaced down. Because that top moves with pan/zoom, the legend
 * can't show fixed numbers; it shows the *current* mapping and re-labels as
 * ``maxCount`` changes.
 *
 * The ramp and singleton colours are resolved from the active {@link colormap}
 * against the live theme (mirroring the canvas), so the key always matches what
 * the canvas is painting — Heat, Ocean, or Grayscale, light or dark.
 */
@Component({
  selector: 'vt-browse-legend',
  standalone: true,
  template: `
    <div class="browse-legend" role="img" [attr.aria-label]="ariaLabel">
      <span class="browse-legend-title">{{ title }}</span>
      @if (ticks.length > 0) {
        <div class="browse-legend-body">
          <div class="browse-legend-bar" [style.background]="barGradient"></div>
          <div class="browse-legend-ticks">
            @for (tick of ticks; track tick.t) {
              <span class="browse-legend-tick" [style.bottom.%]="tick.t * 100">{{
                tick.label
              }}</span>
            }
          </div>
        </div>
      }
      <div class="browse-legend-single">
        <span class="browse-legend-dot" [style.background]="singleColor"></span>
        <span class="browse-legend-tick">1</span>
      </div>
    </div>
  `,
  styleUrl: './browse-legend.component.scss',
})
export class BrowseLegendComponent implements OnInit, OnDestroy {
  /** Heading above the swatch — what the color encodes. */
  @Input() title = 'Items per cell';
  /** Item count the top of the ramp currently maps to. */
  @Input() maxCount = 1;
  /** Active density colormap preset; resolved against the live theme. */
  @Input() colormap: BrowseColormapId = 'auto';

  private readonly numberFormat = new Intl.NumberFormat();
  /** Live theme, refreshed by a ``data-theme`` observer so the key repaints. */
  private theme: CanvasTheme = 'dark';
  private themeObserver: MutationObserver | null = null;

  ngOnInit(): void {
    this.theme = this.readTheme();
    this.themeObserver = new MutationObserver(() => (this.theme = this.readTheme()));
    this.themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme'],
    });
  }

  ngOnDestroy(): void {
    this.themeObserver?.disconnect();
  }

  /** The effective theme from the document, defaulting to dark (no ``system``). */
  private readTheme(): CanvasTheme {
    const t = document.documentElement.getAttribute('data-theme');
    return t === 'light' || t === 'highviz' ? t : 'dark';
  }

  /** Colormap resolved for the current preset + theme (same as the canvas). */
  private get resolved() {
    return resolveColormap(this.colormap, this.theme);
  }

  /** Vertical gradient with the dense (last ramp stop) end at the top. */
  get barGradient(): string {
    return `linear-gradient(to top, ${gradientStops(this.resolved.ramp)})`;
  }

  /** The one-item cell colour, shown as a dot. */
  get singleColor(): string {
    return rgbString(this.resolved.single);
  }

  /**
   * Tick labels for the ramp (multi-item cells, count ≥ 2). The canvas maps a
   * count to a height via `t = ln(count) / ln(maxCount)`, so a height `t` reads
   * back as `count = maxCount^t` and each label sits exactly at the color that
   * count is painted with. Sampled top→bottom, deduped, and dropping anything
   * that rounds below 2 (those are singletons, shown by the dot instead).
   */
  get ticks(): LegendTick[] {
    const max = Math.max(Math.round(this.maxCount), 1);
    if (max < 2) return [];
    const seen = new Set<number>();
    const out: LegendTick[] = [];
    for (const t of [1, 0.75, 0.5, 0.25]) {
      const count = Math.round(Math.pow(max, t));
      if (count < 2 || seen.has(count)) continue;
      seen.add(count);
      out.push({ t, label: this.numberFormat.format(count) });
    }
    return out;
  }

  get ariaLabel(): string {
    const max = Math.max(Math.round(this.maxCount), 1);
    return max < 2
      ? `${this.title}: every cell in view holds 1 item`
      : `${this.title}: a dot is 1 item; the ramp runs up to ${this.numberFormat.format(
          max,
        )} items at its brightest`;
  }
}
