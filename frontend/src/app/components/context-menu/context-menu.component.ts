
import { ChangeDetectionStrategy, Component, ElementRef, HostListener, OnDestroy, OnInit, inject, input, output } from '@angular/core';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';

export interface ContextMenuItem {
  id: string;
  label: string;
  title?: string;
  disabled?: boolean;
  /** Raw inline SVG markup rendered as the item's leading icon. Trusted: only
   *  app-authored constants are passed here, never user content. */
  iconSvg?: string;
}

/**
 * A standard right-click context menu: positioned at a fixed viewport point,
 * dismissed by clicking elsewhere, right-clicking elsewhere, or pressing
 * Escape.  Mousing over an item highlights it; left-clicking an enabled item
 * emits its ``id`` via ``actionSelected``.  Shared by media items
 * (`vt-label-view`) and the dashboard grids.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-context-menu',
  standalone: true,
  imports: [],
  templateUrl: './context-menu.component.html',
  styleUrl: './context-menu.component.scss',
})
export class ContextMenuComponent implements OnInit, OnDestroy {
  private host = inject<ElementRef<HTMLElement>>(ElementRef);
  private sanitizer = inject(DomSanitizer);

  readonly x = input.required<number>();
  readonly y = input.required<number>();
  readonly items = input<ContextMenuItem[]>([]);

  readonly actionSelected = output<string>();
  readonly dismissed = output<void>();

  private iconCache = new Map<string, SafeHtml>();

  /** Sanitise (once per distinct SVG string) the item icon for `[innerHTML]`. */
  safeIcon(svg: string): SafeHtml {
    const hit = this.iconCache.get(svg);
    if (hit !== undefined) return hit;
    const safe = this.sanitizer.bypassSecurityTrustHtml(svg);
    this.iconCache.set(svg, safe);
    return safe;
  }

  /**
   * Dismiss on any click / right-click outside the menu. These run in the
   * *capture* phase (note the `true` flag) rather than via a bubble-phase
   * `@HostListener('document:click')`, because the element being clicked may
   * `stopPropagation()` before the event bubbles to `document` — e.g. a
   * dashboard card's ⋯ overflow button stops the click so it doesn't select
   * the row. A bubble-phase listener never fires in that case, so an
   * already-open menu would linger when the user clicked a *different* card's
   * ⋯ button. Capture sees every event before any descendant can swallow it.
   *
   * The listeners are plain DOM listeners (not Angular `@HostListener`s) so
   * they can opt into capture. They fire outside Angular, but the consumer's
   * `(dismissed)` template binding still schedules change detection on emit
   * under zoneless (the same mechanism `panel-resize.directive` relies on).
   */
  private readonly onDocumentPointer = (event: Event): void => {
    if (!this.host.nativeElement.contains(event.target as Node)) {
      this.dismissed.emit();
    }
  };

  ngOnInit(): void {
    document.addEventListener('click', this.onDocumentPointer, true);
    document.addEventListener('contextmenu', this.onDocumentPointer, true);
  }

  ngOnDestroy(): void {
    document.removeEventListener('click', this.onDocumentPointer, true);
    document.removeEventListener('contextmenu', this.onDocumentPointer, true);
  }

  @HostListener('document:keydown.escape')
  onEscape(): void {
    this.dismissed.emit();
  }

  /** Swallow right-clicks landing on the menu itself so they neither pop a
   *  nested native menu nor bubble back to the element that opened us. */
  onMenuContextMenu(event: MouseEvent): void {
    event.preventDefault();
    event.stopPropagation();
  }

  onItemClick(event: MouseEvent, item: ContextMenuItem): void {
    event.stopPropagation();
    if (item.disabled) return;
    this.actionSelected.emit(item.id);
  }
}
