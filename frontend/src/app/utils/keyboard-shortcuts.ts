/**
 * Shared guards for document-level keyboard shortcuts.
 *
 * Mirrors the checks {@link KeyboardService} applies to the Train/Find
 * shortcuts so the VTSBrowse shortcuts (handled locally by the browse view and
 * its bin-details popup) suppress in the same situations: while the user is
 * typing in a text field, and while a modal dialog is open over the page.
 */

/** True when ``el`` is a text-entry target (an editable input/textarea/select or
 *  a contentEditable element), where typed keys must reach the field rather than
 *  trigger a shortcut. Checkbox/radio/range inputs are not text entry. */
export function isTypingTarget(el: Element | null): boolean {
  if (!el) return false;
  const tag = el.tagName;
  if (tag === 'INPUT') {
    const type = (el as HTMLInputElement).type;
    if (type !== 'checkbox' && type !== 'radio' && type !== 'range') return true;
  }
  if (tag === 'TEXTAREA' || tag === 'SELECT') return true;
  return (el as HTMLElement).isContentEditable;
}

/**
 * True when a page-level keyboard shortcut should be ignored: a modal dialog is
 * open (its ``.modal-backdrop`` is in the DOM) or the focused element is a text
 * field. Callers still apply their own context gates (e.g. the browse popup vs
 * the canvas) on top of this.
 */
export function shortcutsBlocked(): boolean {
  if (document.querySelector('.modal-backdrop')) return true;
  return isTypingTarget(document.activeElement);
}
