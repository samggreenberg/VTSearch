import { ChangeDetectionStrategy, Component, HostListener, OnDestroy, effect, input, output, untracked } from '@angular/core';
import { CdkTrapFocus } from '@angular/cdk/a11y';

/**
 * Stack of currently-open modal instances, in the order they opened.
 *
 * Modals stack in real flows (New Detector → media-crop, Settings →
 * importer picker, any modal → dialog-host confirm/prompt), and every
 * instance listens for Escape on `document`. Without the stack, one
 * keypress dismissed every open modal at once — losing the outer form's
 * state. Only the most-recently-opened modal reacts to Escape.
 */
const openModalStack: ModalComponent[] = [];

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-modal',
  standalone: true,
  imports: [CdkTrapFocus],
  templateUrl: './modal.component.html',
  styleUrl: './modal.component.scss',
})
export class ModalComponent implements OnDestroy {
  readonly title = input('');
  readonly open = input(false);
  readonly showCloseButton = input(true);
  readonly closed = output<void>();

  constructor() {
    // Track this instance's place in the open-modal stack. Runs on every
    // open() change; instances created already-open (the common `@if` +
    // `[open]="true"` pattern) register at creation time, which matches
    // their visual stacking order.
    effect(() => {
      const isOpen = this.open();
      untracked(() => {
        const idx = openModalStack.indexOf(this);
        if (isOpen) {
          if (idx === -1) openModalStack.push(this);
        } else if (idx !== -1) {
          openModalStack.splice(idx, 1);
        }
      });
    });
  }

  ngOnDestroy(): void {
    const idx = openModalStack.indexOf(this);
    if (idx !== -1) openModalStack.splice(idx, 1);
  }

  onBackdropClick(event: MouseEvent): void {
    if (event.target === event.currentTarget) {
      this.close();
    }
  }

  close(): void {
    this.closed.emit();
  }

  /**
   * Close on Escape. Listening on `document` (not the backdrop) is required
   * because opening a modal from a button leaves focus on that button, so a
   * `(keydown)` bound to the never-focused backdrop never fired — Esc did
   * nothing until the user clicked into the modal. The `open()` guard ensures
   * only a currently-open modal reacts. Mirrors the `context-menu` /
   * `browse-bin-popup` `document:keydown.escape` pattern.
   *
   * Only the topmost open modal closes: with stacked modals every instance's
   * document listener fires for the same keypress, and closing them all at
   * once destroyed the outer flow's state along with the inner view.
   */
  @HostListener('document:keydown.escape')
  onEscape(): void {
    if (this.open() && openModalStack[openModalStack.length - 1] === this) {
      this.close();
    }
  }
}
