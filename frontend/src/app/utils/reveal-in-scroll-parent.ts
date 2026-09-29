/**
 * Scroll the nearest ancestor that actually scrolls, and nothing else, just far
 * enough to bring *el* into view: the down-scroll stops once the element's
 * bottom shows, or its top reaches the top of the box when it is too tall to
 * fit, so the start of a tall block is what lands in view.
 *
 * Not `scrollIntoView`: that walks every scroll container up to the viewport,
 * including `overflow: hidden` ones no reader can scroll. A dialog's
 * `.modal-content` is one, and a target taller than `.modal-body` drags it too,
 * sliding the header out of the dialog.
 *
 * A no-op where nothing scrolls (and under jsdom, which lays nothing out).
 */
export function revealInScrollParent(el: HTMLElement): void {
  const box = scrollParent(el);
  if (!box) return;
  const target = el.getBoundingClientRect();
  const view = box.getBoundingClientRect();
  if (target.top < view.top) {
    box.scrollTop -= view.top - target.top;
  } else if (target.bottom > view.bottom) {
    box.scrollTop += Math.min(target.bottom - view.bottom, target.top - view.top);
  }
}

/** Nearest ancestor with `overflow-y: auto | scroll` whose content overflows. */
function scrollParent(el: HTMLElement): HTMLElement | null {
  for (let node = el.parentElement; node; node = node.parentElement) {
    const overflowY = getComputedStyle(node).overflowY;
    if ((overflowY === 'auto' || overflowY === 'scroll') && node.scrollHeight > node.clientHeight) {
      return node;
    }
  }
  return null;
}
