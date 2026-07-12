import { Directive, ElementRef, HostBinding, HostListener, NgZone, OnDestroy, inject, input, output } from '@angular/core';

/**
 * Handles a vertical-divider drag for a flanking panel inside `vt-label-view`.
 *
 * The parent owns the panel widths; this directive translates mouse motion on
 * the divider into a stream of width values for the side it is bound to,
 * clamped to the available space. The drag listeners run outside Angular so
 * the mousemove handler does no per-pixel framework work; the parent's bound
 * `(widthChange)`/`(resizeEnd)` template listeners schedule change detection
 * when each value is emitted (under zoneless `NgZone.run` would be a no-op).
 *
 * Inputs:
 *   - `[vtPanelResize]`: `'left'` or `'right'`; controls which edge of the
 *     layout the new width is measured from.
 *   - `[layoutEl]`: the layout container whose bounding rect defines the
 *     drag bounds.
 *   - `[minWidth]` / `[opposingWidth]` / `[centerMin]` / `[dividerTotal]`:
 *     clamping inputs. Width is constrained to
 *     `[minWidth, layoutWidth - dividerTotal - centerMin - opposingWidth]`.
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
    const max = rect.width - this.dividerTotal() - this.centerMin() - this.opposingWidth();
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
