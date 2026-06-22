import { Directive, ElementRef, HostBinding, HostListener, NgZone, OnDestroy, inject, input, output } from '@angular/core';

/**
 * Handles a vertical-divider drag for a flanking panel inside `vt-label-view`.
 *
 * The parent owns the panel widths; this directive translates mouse motion on
 * the divider into a stream of width values for the side it is bound to,
 * clamped to the available space. Listeners run outside Angular while the user
 * is dragging; only the per-move emission re-enters the zone, so the
 * mousemove handler does not trigger a full change-detection pass on every
 * pixel.
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
    this.ngZone.runOutsideAngular(() => {
      document.addEventListener('mousemove', this.boundMove);
      document.addEventListener('mouseup', this.boundUp);
    });
  }

  ngOnDestroy(): void {
    document.removeEventListener('mousemove', this.boundMove);
    document.removeEventListener('mouseup', this.boundUp);
  }

  private onMouseMove(event: MouseEvent): void {
    if (!this.dragging) return;
    const rect = this.layoutEl().getBoundingClientRect();
    const raw = this.side() === 'left' ? event.clientX - rect.left : rect.right - event.clientX;
    const max = rect.width - this.dividerTotal() - this.centerMin() - this.opposingWidth();
    const width = Math.max(this.minWidth(), Math.min(max, raw));
    this.lastWidth = width;
    this.ngZone.run(() => this.widthChange.emit(width));
  }

  private onMouseUp(): void {
    this.dragging = false;
    document.removeEventListener('mousemove', this.boundMove);
    document.removeEventListener('mouseup', this.boundUp);
    this.ngZone.run(() => this.resizeEnd.emit(this.lastWidth));
  }
}
