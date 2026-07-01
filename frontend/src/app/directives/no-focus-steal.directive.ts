import { Directive, ElementRef, HostListener, inject } from '@angular/core';

/**
 * Stops descendant `<button>`s from stealing keyboard focus on a mouse press.
 *
 * A native button focuses itself on `mousedown`. Next to a drag-interactive
 * surface — the VTSBrowse canvas — that focus is pure nuisance: the
 * last-clicked toolbar button keeps DOM focus, paints a focus ring the moment
 * the user presses a modifier (Shift promotes `:focus` to `:focus-visible`),
 * and answers a Space/Enter that was meant for the canvas. These are icon and
 * label controls that fire an action and have nothing to keep focus *for*.
 *
 * Calling `preventDefault()` on the `mousedown` suppresses the focus transfer
 * (and the start of any text selection/drag) while leaving the click itself
 * intact — the button still fires, but focus stays where it was, over the
 * canvas. Keyboard users are unaffected: Tab focus never routes through
 * `mousedown`, so the buttons remain reachable and operable by keyboard.
 *
 * Put the directive on a container (e.g. a toolbar) rather than each button:
 * it delegates via the event target, so buttons added later are covered too.
 * Only presses that land on a `<button>` inside the host are intercepted.
 */
@Directive({
  selector: '[vtNoFocusSteal]',
  standalone: true,
})
export class NoFocusStealDirective {
  private host = inject<ElementRef<HTMLElement>>(ElementRef);

  @HostListener('mousedown', ['$event'])
  onMouseDown(event: MouseEvent): void {
    const target = event.target as HTMLElement | null;
    const button = target?.closest('button');
    if (button && this.host.nativeElement.contains(button)) {
      event.preventDefault();
    }
  }
}
