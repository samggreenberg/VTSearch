import {
  afterRenderEffect,
  booleanAttribute,
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
import { ModalComponent, topOpenModal } from '../modal/modal.component';
import { TOASTY_FACES, ToastyFace } from '../../utils/toasty-faces';

/** The subset of `DOMRect` the placement needs; lets specs pass plain objects. */
export interface RectLike {
  left: number;
  top: number;
  right: number;
  bottom: number;
  width: number;
  height: number;
}

/**
 * Where the hint sits: under the control, over it, or `beside` it, at its top
 * corner with the bubble reaching out sideways. `beside` is for a dialog box,
 * which fills the window and leaves no room under it; `below` falls back to it
 * on its own when the hint would run off the bottom (see
 * {@link toastyHintPlacement}).
 */
export type ToastyHintSide = 'below' | 'above' | 'beside';

/** Which way the speech bubble reaches from Toasty, who stands at the control. */
export type ToastyHintReach = 'left' | 'right';

/** Where the hint goes, in its containing block's px, and which way it reaches. */
export interface ToastyHintPlacement {
  left: number;
  top: number;
  side: ToastyHintSide;
  reach: ToastyHintReach;
}

/** Clearance (px) between the control and Toasty. */
const GAP = 4;

/** The face Toasty pulls while the pointer is over him (#5462): startled out
 *  of a happy or sad face, cheered out of a surprised one. */
export const HOVER_FACE: Readonly<Record<ToastyFace, ToastyFace>> = {
  happy: 'surprised',
  sad: 'surprised',
  surprised: 'happy',
};

/** Where a drag of the hint started: the pointer, and the hint's offset then. */
interface DragStart {
  pointerId: number;
  x: number;
  y: number;
  from: { x: number; y: number };
}

/**
 * Place a hint: Toasty centred on the anchor control, just below or above it,
 * with the bubble reaching the preferred way; or, `beside`, Toasty at the
 * control's top corner on the side the bubble reaches, the bubble reaching on
 * outward. When the bubble would run past the container's edge and reaching
 * the other way would not, it reaches the other way; when neither fits, it is
 * pushed back inside. A hint `below` a control that would run off the bottom
 * of the container (a dialog box filling the window) stands `beside` it
 * instead, when the container's height is known.
 *
 * `anchor` is in viewport coordinates (as `getBoundingClientRect()` returns
 * it), `origin` is the container's padding-box corner in the same coordinates,
 * `bounds` the container's size, `hint` the hint's own size and `toasty` the
 * width of Toasty's picture. Returns `null` while the anchor is unrendered.
 */
export function toastyHintPlacement(
  anchor: RectLike,
  origin: Pick<RectLike, 'left' | 'top'>,
  bounds: { width: number; height?: number },
  hint: { width: number; height: number },
  toasty: number,
  side: ToastyHintSide,
  reach: ToastyHintReach,
): ToastyHintPlacement | null {
  if (anchor.width === 0 && anchor.height === 0) return null;
  const x = (anchor.left + anchor.right) / 2 - origin.left;
  let top = side === 'above' ? anchor.top - GAP - origin.top - hint.height : anchor.bottom + GAP - origin.top;
  if (side === 'below' && bounds.height !== undefined && bounds.height > 0 && top + hint.height > bounds.height) {
    side = 'beside';
  }
  if (side === 'beside') top = Math.max(anchor.top - origin.top, 0);
  const leftFor = (r: ToastyHintReach): number => {
    if (side === 'beside') {
      return r === 'right' ? anchor.right + GAP - origin.left : anchor.left - GAP - origin.left - hint.width;
    }
    return r === 'left' ? x + toasty / 2 - hint.width : x - toasty / 2;
  };
  const fits = (left: number): boolean => left >= 0 && left + hint.width <= bounds.width;
  const other: ToastyHintReach = reach === 'left' ? 'right' : 'left';
  const chosen = !fits(leftFor(reach)) && fits(leftFor(other)) ? other : reach;
  const left = Math.min(Math.max(leftFor(chosen), 0), Math.max(bounds.width - hint.width, 0));
  return { left, top, side, reach: chosen };
}

/**
 * One of King Toasty's hints (#4680): Toasty stands beside the control a user
 * should click next, with a speech bubble saying what to do and why. The bubble
 * reaches away from the control, so it never covers it. The projected content
 * is what Toasty says.
 *
 * Each bubble carries "Hide this hint" and "Hide all hints" boxes, kept in the
 * user's settings through {@link HintsService}, unless it is not `hideable`;
 * a hint the user has hidden renders nothing. When to show a hint at all is
 * the caller's call: wrap it in an `@if` that goes false the moment the user
 * takes the step.
 *
 * At most one hint is on screen at a time (#4807): a hint inside a dialog
 * (any `vt-modal` it is projected into, however deep) shows only while that
 * dialog is the topmost one open, and a hint on a view shows only while no
 * dialog is open. A hint `aboveDialogs` (the offline banner's, #5149) shows
 * whatever is open, and {@link HintsService} keeps every other hint off while
 * it is up. Inside a dialog the `anchor` may be left out: the hint is then
 * about the dialog box itself, standing below it (or beside it when the box
 * fills the window) with the bubble reaching right.
 *
 * The host positions itself absolutely in its nearest positioned ancestor, so
 * drop it anywhere under a positioned container; the anchor may sit outside
 * it, as long as the container does not clip where the hint lands. It is
 * measured after render and again whenever the anchor, the container or the
 * hint itself changes size.
 *
 * The user can drag Toasty or his bubble out of the way (#5462), by anything
 * but the Hide boxes. Where it was dragged is not kept: the offset lives on
 * this instance and goes with it, so the next hint stands where it belongs.
 * Pointing at Toasty changes his face ({@link HOVER_FACE}).
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
    '[class.toasty-hint--above]': "(placement()?.side ?? side()) === 'above'",
    '[class.toasty-hint--reach-right]': "(placement()?.reach ?? reachWay()) === 'right'",
    '[class.toasty-hint--off]': '!shown()',
    '[style.left.px]': 'placement()?.left',
    '[style.top.px]': 'placement()?.top',
    '[style.visibility]': "placement() ? null : 'hidden'",
    '[style.translate]': 'dragTranslate()',
    '[class.toasty-hint--dragging]': 'dragging()',
    '(pointerdown)': 'startDrag($event)',
    '(pointermove)': 'drag($event)',
    '(pointerup)': 'endDrag($event)',
    '(pointercancel)': 'endDrag($event)',
  },
})
export class ToastyHintComponent {
  readonly hintId = input.required<HintId>();
  /** The control the hint is about. Inside a dialog it may be left out, and
   *  the hint is about the dialog box. */
  readonly anchor = input<HTMLElement>();
  readonly side = input<ToastyHintSide>('below');
  /** Which way the bubble would rather reach; it flips if that runs off the
   *  edge. Left on a view, right under a dialog box, unless set. */
  readonly reach = input<ToastyHintReach>();
  /** Toasty's face: happy for a next step, surprised when something needs
   *  fixing, sad when something went wrong. */
  readonly face = input<ToastyFace>('happy');
  /** The pointer is over Toasty, who then pulls his {@link HOVER_FACE}. */
  readonly hovered = signal(false);
  readonly faceSrc = computed(() => TOASTY_FACES[this.hovered() ? HOVER_FACE[this.face()] : this.face()]);
  /** Whether the bubble offers "Hide this hint" and "Hide all hints". Off for
   *  a hint shown while the server is unreachable, where a hide can't be saved. */
  readonly hideable = input(true, { transform: booleanAttribute });
  /** Shows over any dialog, rather than only while none is open: for a hint
   *  on a surface that floats over them all (the offline banner). */
  readonly aboveDialogs = input(false, { transform: booleanAttribute });

  private readonly hints = inject(HintsService);
  /** The dialog this hint is projected into, or `null` on a view. */
  private readonly modal = inject(ModalComponent, { optional: true });

  /** What the hint stands at: the anchor, else the enclosing dialog's box. */
  readonly target = computed<HTMLElement | null>(
    () => this.anchor() ?? this.modal?.box()?.nativeElement ?? null,
  );
  readonly reachWay = computed<ToastyHintReach>(() => this.reach() ?? (this.modal ? 'right' : 'left'));
  readonly shown = computed(
    () => this.hints.isShown(this.hintId()) && (this.aboveDialogs() || topOpenModal() === this.modal),
  );

  /** Written from the after-render hook and the ResizeObserver callback
   *  (neither is a template event), so a signal so the host moves. */
  readonly placement = signal<ToastyHintPlacement | null>(null);

  /** How far (px) the user has dragged the hint from its placement. */
  readonly dragOffset = signal({ x: 0, y: 0 });
  readonly dragTranslate = computed(() => {
    const { x, y } = this.dragOffset();
    return x === 0 && y === 0 ? null : `${x}px ${y}px`;
  });
  private readonly dragStart = signal<DragStart | null>(null);
  readonly dragging = computed(() => this.dragStart() !== null);

  private readonly host: HTMLElement = inject(ElementRef).nativeElement;

  constructor() {
    afterRenderEffect({
      read: () => this.measure(this.target(), this.side(), this.reachWay(), this.shown()),
    });
    effect((onCleanup) => {
      const anchor = this.target();
      const side = this.side();
      const reach = this.reachWay();
      if (typeof ResizeObserver === 'undefined') return;
      const observer = new ResizeObserver(() => this.measure(anchor, side, reach, this.shown()));
      for (const el of [this.host, this.container(), ...(anchor ? [anchor] : [])]) observer.observe(el);
      onCleanup(() => observer.disconnect());
    });
  }

  hideThis(): void {
    this.hints.hide(this.hintId());
  }

  hideAll(): void {
    this.hints.hideAll();
  }

  /** A primary press on Toasty or his bubble, outside the Hide boxes, starts
   *  a drag. Only those two take the pointer at all (the host passes it
   *  through), and the press is captured on whichever it landed on, so the
   *  drag keeps up with a pointer that outruns it. */
  startDrag(event: PointerEvent): void {
    const target = event.target as Element | null;
    if (event.button !== 0 || !target || target.closest('.toasty-hint__options')) return;
    const handle = target.closest('.toasty-hint__toasty, .toasty-hint__bubble');
    if (!handle) return;
    // Optional: jsdom has no pointer capture.
    handle.setPointerCapture?.(event.pointerId);
    this.dragStart.set({ pointerId: event.pointerId, x: event.clientX, y: event.clientY, from: this.dragOffset() });
  }

  drag(event: PointerEvent): void {
    const start = this.dragStart();
    if (!start || event.pointerId !== start.pointerId) return;
    this.dragOffset.set({ x: start.from.x + event.clientX - start.x, y: start.from.y + event.clientY - start.y });
  }

  endDrag(event: PointerEvent): void {
    if (this.dragStart()?.pointerId === event.pointerId) this.dragStart.set(null);
  }

  /** The box the host is absolutely positioned in. jsdom has no layout, so no
   *  `offsetParent`; the parent element stands in there. */
  private container(): HTMLElement {
    return (this.host.offsetParent as HTMLElement | null) ?? this.host.parentElement ?? this.host;
  }

  private measure(
    anchor: HTMLElement | null,
    side: ToastyHintSide,
    reach: ToastyHintReach,
    shown: boolean,
  ): void {
    if (!shown || !anchor) return;
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
        { width: box.clientWidth || rect.width, height: box.clientHeight || rect.height },
        { width: this.host.offsetWidth, height: this.host.offsetHeight },
        toasty,
        side,
        reach,
      ),
    );
  }
}
