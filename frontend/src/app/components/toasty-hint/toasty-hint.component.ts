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

/** Where the hint goes, in its containing block's px, and which way it reaches. */
export interface ToastyHintPlacement {
  left: number;
  top: number;
  reach: ToastyHintReach;
}

/** Clearance (px) between the control and Toasty. */
const GAP = 4;

/**
 * Place a hint: Toasty centred on the anchor control, just below or above it,
 * with the bubble reaching the preferred way. When the bubble would run past
 * the container's edge and reaching the other way would not, it reaches the
 * other way; when neither fits, it is pushed back inside.
 *
 * `anchor` is in viewport coordinates (as `getBoundingClientRect()` returns
 * it), `origin` is the container's padding-box corner in the same coordinates,
 * `bounds` the container's size, `hint` the hint's own size and `toasty` the
 * width of Toasty's picture. Returns `null` while the anchor is unrendered.
 */
export function toastyHintPlacement(
  anchor: RectLike,
  origin: Pick<RectLike, 'left' | 'top'>,
  bounds: { width: number },
  hint: { width: number; height: number },
  toasty: number,
  side: ToastyHintSide,
  reach: ToastyHintReach,
): ToastyHintPlacement | null {
  if (anchor.width === 0 && anchor.height === 0) return null;
  const x = (anchor.left + anchor.right) / 2 - origin.left;
  const top =
    side === 'below' ? anchor.bottom + GAP - origin.top : anchor.top - GAP - origin.top - hint.height;
  const leftFor = (r: ToastyHintReach): number => (r === 'left' ? x + toasty / 2 - hint.width : x - toasty / 2);
  const fits = (left: number): boolean => left >= 0 && left + hint.width <= bounds.width;
  const other: ToastyHintReach = reach === 'left' ? 'right' : 'left';
  const chosen = !fits(leftFor(reach)) && fits(leftFor(other)) ? other : reach;
  const left = Math.min(Math.max(leftFor(chosen), 0), Math.max(bounds.width - hint.width, 0));
  return { left, top, reach: chosen };
}

/**
 * One of King Toasty's hints (#4680): Toasty stands beside the control a user
 * should click next, with a speech bubble saying what to do and why. The bubble
 * reaches away from the control, so it never covers it. The projected content
 * is what Toasty says.
 *
 * Each bubble carries "Hide this hint" and "Hide all hints" boxes, kept in the
 * user's settings through {@link HintsService}; a hint the user has hidden
 * renders nothing. When to show a hint at all is the caller's call: wrap it in
 * an `@if` that goes false the moment the user takes the step.
 *
 * The host positions itself absolutely in its nearest positioned ancestor, so
 * drop it anywhere under a positioned container; the anchor may sit outside
 * it, as long as the container does not clip where the hint lands. It is
 * measured after render and again whenever the anchor, the container or the
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
    '[class.toasty-hint--reach-right]': "(placement()?.reach ?? reach()) === 'right'",
    '[class.toasty-hint--off]': '!shown()',
    '[style.left.px]': 'placement()?.left',
    '[style.top.px]': 'placement()?.top',
    '[style.visibility]': "placement() ? null : 'hidden'",
  },
})
export class ToastyHintComponent {
  readonly hintId = input.required<HintId>();
  /** The control the hint is about. */
  readonly anchor = input.required<HTMLElement>();
  readonly side = input<ToastyHintSide>('below');
  /** Which way the bubble would rather reach; it flips if that runs off the edge. */
  readonly reach = input<ToastyHintReach>('left');
  /** Toasty's picture: the logo, unless a hint asks for another pose. */
  readonly image = input('logo.png');

  private readonly hints = inject(HintsService);
  readonly shown = computed(() => this.hints.isShown(this.hintId()));

  /** Written from the after-render hook and the ResizeObserver callback
   *  (neither is a template event), so a signal so the host moves. */
  readonly placement = signal<ToastyHintPlacement | null>(null);

  private readonly host: HTMLElement = inject(ElementRef).nativeElement;

  constructor() {
    afterRenderEffect({
      read: () => this.measure(this.anchor(), this.side(), this.reach(), this.shown()),
    });
    effect((onCleanup) => {
      const anchor = this.anchor();
      const side = this.side();
      const reach = this.reach();
      if (typeof ResizeObserver === 'undefined') return;
      const observer = new ResizeObserver(() => this.measure(anchor, side, reach, this.shown()));
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

  private measure(anchor: HTMLElement, side: ToastyHintSide, reach: ToastyHintReach, shown: boolean): void {
    if (!shown) return;
    const box = this.container();
    const rect = box.getBoundingClientRect();
    // The containing block is the padding box: step in past the border, and
    // count anything the container has scrolled.
    const origin = {
      left: rect.left + box.clientLeft - box.scrollLeft,
      top: rect.top + box.clientTop - box.scrollTop,
    };
    const toasty = this.host.querySelector('img')?.offsetWidth ?? 0;
    this.placement.set(
      toastyHintPlacement(
        anchor.getBoundingClientRect(),
        origin,
        { width: box.clientWidth || rect.width },
        { width: this.host.offsetWidth, height: this.host.offsetHeight },
        toasty,
        side,
        reach,
      ),
    );
  }
}
