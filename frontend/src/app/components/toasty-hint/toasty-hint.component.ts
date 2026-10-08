import {
  afterRenderEffect,
  ChangeDetectionStrategy,
  Component,
  computed,
  effect,
  ElementRef,
  inject,
  input,
  signal,
} from '@angular/core';
import { HintId, HintsService } from '../../services/hints.service';

/** The subset of `DOMRect` the placement needs; lets specs pass plain objects. */
export interface RectLike {
  left: number;
  top: number;
  right: number;
  bottom: number;
  width: number;
  height: number;
}

/** Which side of the control the hint sits on. */
export type ToastyHintSide = 'below' | 'above';

/** Which way the speech bubble reaches from Toasty, who stands at the control. */
export type ToastyHintReach = 'left' | 'right';

/** A point in the host's containing block, in px. */
export interface ToastyHintPoint {
  x: number;
  y: number;
}

/** Clearance (px) between the control and Toasty. */
const GAP = 4;

/**
 * Where Toasty stands: centred on the anchor control, just below or above it.
 * `anchor` is in viewport coordinates (as `getBoundingClientRect()` returns
 * it) and the result is relative to `origin`, the box the hint is absolutely
 * positioned in. Returns `null` while the anchor is unrendered (zero-size).
 */
export function toastyHintPoint(
  anchor: RectLike,
  origin: Pick<RectLike, 'left' | 'top'>,
  side: ToastyHintSide,
): ToastyHintPoint | null {
  if (anchor.width === 0 && anchor.height === 0) return null;
  return {
    x: (anchor.left + anchor.right) / 2 - origin.left,
    y: (side === 'below' ? anchor.bottom + GAP : anchor.top - GAP) - origin.top,
  };
}

/**
 * One of King Toasty's hints (#4680): Toasty stands beside the control a new
 * user should click next, with a speech bubble saying what to do and why. The
 * bubble reaches away from the control, so it never covers it. The projected
 * content is what Toasty says.
 *
 * Each bubble carries "Hide this hint" and "Hide all hints" boxes, kept in the
 * user's settings through {@link HintsService}; a hint the user has hidden
 * renders nothing. When to show a hint at all is the caller's call: wrap it in
 * an `@if` that goes false the moment the user takes the step.
 *
 * The host positions itself absolutely in its nearest positioned ancestor, so
 * drop it anywhere under a positioned container that encloses the anchor. It
 * is measured after render and again whenever the anchor, the container or the
 * hint itself changes size.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-toasty-hint',
  standalone: true,
  templateUrl: './toasty-hint.component.html',
  styleUrl: './toasty-hint.component.scss',
  host: {
    role: 'note',
    'aria-label': 'Hint from Toasty',
    '[class.toasty-hint--above]': "side() === 'above'",
    '[class.toasty-hint--reach-right]': "reach() === 'right'",
    '[class.toasty-hint--off]': '!shown()',
    '[style.left.px]': 'point()?.x',
    '[style.top.px]': 'point()?.y',
    '[style.visibility]': "point() ? null : 'hidden'",
  },
})
export class ToastyHintComponent {
  readonly hintId = input.required<HintId>();
  /** The control the hint is about. */
  readonly anchor = input.required<HTMLElement>();
  readonly side = input<ToastyHintSide>('below');
  readonly reach = input<ToastyHintReach>('left');
  /** Toasty's picture: searching, magnifier in hand, unless a hint asks for
   *  another pose. */
  readonly image = input('toasty-search.png');

  private readonly hints = inject(HintsService);
  readonly shown = computed(() => this.hints.isShown(this.hintId()));

  /** Written from the after-render hook and the ResizeObserver callback
   *  (neither is a template event), so a signal so the host moves. */
  readonly point = signal<ToastyHintPoint | null>(null);

  private readonly host: HTMLElement = inject(ElementRef).nativeElement;

  constructor() {
    afterRenderEffect({ read: () => this.measure(this.anchor(), this.side(), this.shown()) });
    effect((onCleanup) => {
      const anchor = this.anchor();
      const side = this.side();
      if (typeof ResizeObserver === 'undefined') return;
      const observer = new ResizeObserver(() => this.measure(anchor, side, this.shown()));
      for (const el of [this.host, this.container(), anchor]) observer.observe(el);
      onCleanup(() => observer.disconnect());
    });
  }

  hideThis(): void {
    this.hints.hide(this.hintId());
  }

  hideAll(): void {
    this.hints.hideAll();
  }

  /** The box the host is absolutely positioned in. jsdom has no layout, so no
   *  `offsetParent`; the parent element stands in there. */
  private container(): HTMLElement {
    return (this.host.offsetParent as HTMLElement | null) ?? this.host.parentElement ?? this.host;
  }

  private measure(anchor: HTMLElement, side: ToastyHintSide, shown: boolean): void {
    if (!shown) return;
    const box = this.container();
    const rect = box.getBoundingClientRect();
    // The containing block is the padding box: step in past the border, and
    // count anything the container has scrolled.
    const origin = {
      left: rect.left + box.clientLeft - box.scrollLeft,
      top: rect.top + box.clientTop - box.scrollTop,
    };
    this.point.set(toastyHintPoint(anchor.getBoundingClientRect(), origin, side));
  }
}
