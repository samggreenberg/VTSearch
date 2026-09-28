import {
  afterRenderEffect,
  ChangeDetectionStrategy,
  Component,
  effect,
  ElementRef,
  inject,
  input,
  signal,
} from '@angular/core';

/** The subset of `DOMRect` the geometry needs; lets specs pass plain objects. */
export interface RectLike {
  left: number;
  top: number;
  right: number;
  bottom: number;
  width: number;
  height: number;
}

/** SVG path data for one arrow, in the host's own pixel coordinates. */
export interface PointerArrowGeometry {
  shaft: string;
  head: string;
}

/** Clearance (px) left between the arrow and each element it connects. */
const GAP = 8;
/** Length (px) of each arrowhead wing. */
const HEAD_LEN = 10;
/** Half-angle of the arrowhead. */
const HEAD_ANGLE = Math.PI / 6;

const fmt = (n: number): string => (Math.round(n * 10) / 10).toString();

/**
 * Geometry of an arrow from the `from` box to the `to` box, both given in
 * viewport coordinates (as `getBoundingClientRect()` returns them) and
 * translated into coordinates relative to `origin` (the arrow's host box).
 *
 * The arrow always *arrives vertically*, tip at the middle of the target's
 * near edge, which is the shape both of today's callers want: a Dashboard `+`
 * button sits above its empty-state message, and the Train button sits below
 * the Detectors panel. When the target lies clear of the source horizontally,
 * the shaft leaves the source's nearer side and bends into the target (a
 * quarter curve); when the two overlap horizontally it drops straight out of
 * the source's top or bottom edge as a gentle S. Returns `null` when either
 * box is unrendered (zero-size) or the boxes overlap vertically, since there
 * is no sensible vertical arrival then.
 */
export function pointerArrowGeometry(
  from: RectLike,
  to: RectLike,
  origin: Pick<RectLike, 'left' | 'top'>,
): PointerArrowGeometry | null {
  if ((from.width === 0 && from.height === 0) || (to.width === 0 && to.height === 0)) return null;
  const below = to.top >= from.bottom;
  const above = to.bottom <= from.top;
  if (!below && !above) return null;

  const x = (vx: number): number => vx - origin.left;
  const y = (vy: number): number => vy - origin.top;
  // +1 when the arrow travels down the page, -1 when it travels up.
  const dir = below ? 1 : -1;
  const endX = x((to.left + to.right) / 2);
  const endY = below ? y(to.top) - GAP : y(to.bottom) + GAP;

  let shaft: string;
  if (endX > x(from.right) + GAP || endX < x(from.left) - GAP) {
    // Target clears the source sideways: leave from the nearer side,
    // level with the source's middle, and bend into the target.
    const startX = endX > x(from.right) ? x(from.right) + GAP : x(from.left) - GAP;
    const startY = y((from.top + from.bottom) / 2);
    shaft = `M ${fmt(startX)} ${fmt(startY)} Q ${fmt(endX)} ${fmt(startY)} ${fmt(endX)} ${fmt(endY)}`;
  } else {
    const startX = x((from.left + from.right) / 2);
    const startY = below ? y(from.bottom) + GAP : y(from.top) - GAP;
    const midY = (startY + endY) / 2;
    shaft =
      `M ${fmt(startX)} ${fmt(startY)} C ${fmt(startX)} ${fmt(midY)} ` +
      `${fmt(endX)} ${fmt(midY)} ${fmt(endX)} ${fmt(endY)}`;
  }

  // Both shaft shapes end on a vertical tangent, so the head is a plain
  // chevron opening back along the direction of travel.
  const wingDx = HEAD_LEN * Math.sin(HEAD_ANGLE);
  const wingY = endY - dir * HEAD_LEN * Math.cos(HEAD_ANGLE);
  const head =
    `M ${fmt(endX - wingDx)} ${fmt(wingY)} L ${fmt(endX)} ${fmt(endY)} ` +
    `L ${fmt(endX + wingDx)} ${fmt(wingY)}`;
  return { shaft, head };
}

/**
 * A curved "look here" arrow drawn from one element to another, used by the
 * Dashboard's first-run hints (issue #4227) to connect a message ("No datasets
 * yet…", "Click Train…") to the control it talks about.
 *
 * The host absolutely fills its nearest positioned ancestor and ignores the
 * pointer, so the caller only has to drop `<vt-pointer-arrow>` somewhere under
 * a positioned container that encloses both endpoints. Positions are measured
 * after render and re-measured whenever either endpoint, the source's parent,
 * or the host itself changes size.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-pointer-arrow',
  standalone: true,
  templateUrl: './pointer-arrow.component.html',
  styleUrl: './pointer-arrow.component.scss',
  host: { 'aria-hidden': 'true' },
})
export class PointerArrowComponent {
  readonly from = input.required<HTMLElement>();
  readonly to = input.required<HTMLElement>();

  /** Written from the after-render hook and the ResizeObserver callback
   *  (neither is a template event), so a signal so the path repaints. */
  readonly geometry = signal<PointerArrowGeometry | null>(null);

  private readonly host: HTMLElement = inject(ElementRef).nativeElement;

  constructor() {
    afterRenderEffect({ read: () => this.measure(this.from(), this.to()) });
    effect((onCleanup) => {
      const from = this.from();
      const to = this.to();
      if (typeof ResizeObserver === 'undefined') return;
      const observer = new ResizeObserver(() => this.measure(from, to));
      for (const el of [this.host, from, from.parentElement, to]) {
        if (el) observer.observe(el);
      }
      onCleanup(() => observer.disconnect());
    });
  }

  private measure(from: HTMLElement, to: HTMLElement): void {
    this.geometry.set(
      pointerArrowGeometry(
        from.getBoundingClientRect(),
        to.getBoundingClientRect(),
        this.host.getBoundingClientRect(),
      ),
    );
  }
}
