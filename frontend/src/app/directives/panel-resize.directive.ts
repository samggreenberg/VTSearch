import { Directive, ElementRef, HostBinding, HostListener, NgZone, OnDestroy, inject, input, output } from '@angular/core';

/**
 * Handles a vertical-divider drag for a panel flanking a layout container.
 *
 * The parent owns the panel widths; this directive translates mouse motion on
 * the divider into a stream of width values for the side it is bound to,
 * clamped to the available space. The drag listeners are attached outside
 * Angular so the mousemove handler does no per-pixel framework work, and are
 * torn down both on mouseup and on destroy (a component unmounted mid-drag
 * must not leave `document` listeners behind).
 *
 * ## When to use this
 *
 * The shape it covers is: **an absolute pointer position measured as a
 * distance from one edge of a layout rect, producing one width.** If that is
 * your divider, bind it and delete your handlers.
 *
 * It deliberately does **not** cover three shapes that look similar but are
 * not, and stretching it to fit them would cost more than it saves:
 *
 * - **Delta drags** that track `startX`/`startWidth` from mousedown rather
 *   than an absolute edge distance (`browse-bin-popup`'s metadata divider —
 *   also inverted, and on PointerEvents).
 * - **Cross-axis drags**, where vertical motion drives a *width*
 *   (`browse-view`'s docked-details row divider maps `dy` 1:1 onto the panel
 *   width, because the focused item is square).
 * - **Two-dimensional resizes** (`browse-minimap`'s corner handle, which
 *   drives width *and* height from a bottom-right anchor).
 *
 * ## The change-detection contract (read before adopting)
 *
 * The app is zoneless, so `NgZone.runOutsideAngular` here is a no-op kept for
 * intent rather than effect — see `docs/FRONTEND.md` §5, which is explicit
 * that `NgZone` is not what makes any of this work. What actually repaints the
 * panel is the parent's **template-bound** `(widthChange)`/`(resizeEnd)`
 * listeners: invoking a template listener notifies the scheduler, so every
 * emission costs one change-detection pass.
 *
 * That cost is the reason this directive is not a universal win. A parent that
 * already writes a signal per mousemove pays it either way and should adopt.
 * A parent that deliberately avoids per-pixel change detection — `find-view`
 * sets a `--left-width` CSS custom property imperatively and binds nothing —
 * would *gain* a change-detection pass per mousemove by adopting, on its
 * heaviest view. Check which of the two you are before wiring this up.
 *
 * Likewise, prefer constants or cached signals for the clamping inputs: they
 * are re-evaluated on every change-detection cycle, so binding a DOM-measuring
 * expression (`getBoundingClientRect`, `getComputedStyle`) to `[minWidth]`
 * turns a drag into sustained layout thrash.
 *
 * Inputs:
 *   - `[vtPanelResize]`: `'left'` or `'right'`; controls which edge of the
 *     layout the new width is measured from.
 *   - `[layoutEl]`: the layout container whose bounding rect defines the
 *     drag bounds.
 *   - `[minWidth]` / `[maxWidth]` / `[opposingWidth]` / `[centerMin]` /
 *     `[dividerTotal]`: clamping inputs. Width is constrained to
 *     `[minWidth, min(maxWidth, layoutWidth - dividerTotal - centerMin - opposingWidth)]`,
 *     with `minWidth` winning if that range inverts.
 *
 * Outputs:
 *   - `(widthChange)` fires on every mousemove with the new width.
 *   - `(resizeEnd)` fires once on mouseup with the final width, after which
 *     the parent typically snaps to a grid-column boundary and persists.
 */
@Directive({
  selector: '[vtPanelResize]',
  standalone: true,
})
export class PanelResizeDirective implements OnDestroy {
  private host = inject<ElementRef<HTMLElement>>(ElementRef);
  private ngZone = inject(NgZone);

  readonly side = input<'left' | 'right'>('left', { alias: "vtPanelResize" });
  readonly layoutEl = input.required<HTMLElement>();
  readonly minWidth = input(100);
  /** Absolute cap, applied on top of the space-derived maximum. Defaults to
   *  unbounded, so a parent with no cap of its own can leave it unset. */
  readonly maxWidth = input(Number.POSITIVE_INFINITY);
  readonly opposingWidth = input(0);
  readonly centerMin = input(100);
  readonly dividerTotal = input(16);

  readonly widthChange = output<number>();
  readonly resizeEnd = output<number>();

  @HostBinding('class.dragging') dragging = false;

  private boundMove = this.onMouseMove.bind(this);
  private boundUp = this.onMouseUp.bind(this);
  private lastWidth = 0;

  @HostListener('mousedown', ['$event'])
  onMouseDown(event: MouseEvent): void {
    event.preventDefault();
    this.dragging = true;
    // Seed `lastWidth` with the current width so a mousedown→mouseup with no
    // intervening move emits the panel's existing width on `resizeEnd`, not the
    // 0 it would otherwise carry — which the parent would clamp/snap straight to
    // the minimum, jumping the panel on a stray click of the divider.
    this.lastWidth = this.computeWidth(event);
    this.ngZone.runOutsideAngular(() => {
      document.addEventListener('mousemove', this.boundMove);
      document.addEventListener('mouseup', this.boundUp);
    });
  }

  ngOnDestroy(): void {
    document.removeEventListener('mousemove', this.boundMove);
    document.removeEventListener('mouseup', this.boundUp);
  }

  /** Translate a pointer position into a clamped width for this side. */
  private computeWidth(event: MouseEvent): number {
    const rect = this.layoutEl().getBoundingClientRect();
    const raw = this.side() === 'left' ? event.clientX - rect.left : rect.right - event.clientX;
    const fit = rect.width - this.dividerTotal() - this.centerMin() - this.opposingWidth();
    const max = Math.min(this.maxWidth(), fit);
    return Math.max(this.minWidth(), Math.min(max, raw));
  }

  private onMouseMove(event: MouseEvent): void {
    if (!this.dragging) return;
    const width = this.computeWidth(event);
    this.lastWidth = width;
    this.widthChange.emit(width);
  }

  private onMouseUp(): void {
    this.dragging = false;
    document.removeEventListener('mousemove', this.boundMove);
    document.removeEventListener('mouseup', this.boundUp);
    this.resizeEnd.emit(this.lastWidth);
  }
}
