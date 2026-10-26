import { DestroyRef, Directive, ElementRef, effect, inject, untracked } from '@angular/core';

import { topOpenModal } from '../components/modal/modal.component';

/**
 * Keeps a page-level overlay usable while a dialog is open: the toast stack and
 * the offline banner, which float over everything else (#5297).
 *
 * `vt-modal` is a modal `<dialog>`. The browser draws it in the top layer, over
 * every z-index, and makes the rest of the document inert. Left where it is, an
 * overlay sits under the dialog's backdrop, takes no click or focus, and drops
 * out of the accessibility tree, so a toast raised from inside a dialog could be
 * neither read nor dismissed. A `popover` does not escape that: shown over a
 * modal dialog it is inert too (checked in chromium). So this moves its host
 * into the topmost open dialog while there is one, and puts it back where it was
 * when the last one closes. Inside the dialog the overlay's own
 * `position: fixed` and z-index place it as before, over the dialog box.
 *
 * Angular never re-inserts a static element, so moving the host does not upset
 * the view it belongs to. A comment marks its place at home while it is away.
 */
@Directive({
  selector: '[vtAboveDialogs]',
  standalone: true,
})
export class AboveDialogsDirective {
  constructor() {
    const host: HTMLElement = inject(ElementRef).nativeElement;
    let placeholder: Comment | null = null;

    const goHome = () => {
      if (!placeholder) return;
      placeholder.replaceWith(host);
      placeholder = null;
    };

    effect(() => {
      const dialog = topOpenModal()?.dialog()?.nativeElement ?? null;
      untracked(() => {
        if (!dialog) {
          goHome();
          return;
        }
        if (host.parentNode === dialog) return;
        if (!placeholder) {
          placeholder = document.createComment('vtAboveDialogs');
          host.replaceWith(placeholder);
        }
        dialog.appendChild(host);
      });
    });

    inject(DestroyRef).onDestroy(goHome);
  }
}
