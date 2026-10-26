import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  HostListener,
  OnDestroy,
  effect,
  input,
  output,
  signal,
  untracked,
  viewChild,
} from '@angular/core';

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

/**
 * The dialog on top of the open stack, or `null` while none is open. Toasty's
 * hints read it (#4807): a hint shows only in the topmost dialog, or on the
 * view behind when no dialog is up, so two are never on screen at once.
 */
export const topOpenModal = signal<ModalComponent | null>(null);

function syncTopOpenModal(): void {
  topOpenModal.set(openModalStack[openModalStack.length - 1] ?? null);
}

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-modal',
  standalone: true,
  templateUrl: './modal.component.html',
  styleUrl: './modal.component.scss',
})
export class ModalComponent implements OnDestroy {
  readonly title = input('');
  readonly open = input(false);
  readonly showCloseButton = input(true);
  /**
   * Where focus lands when the dialog opens. `'first-control'` moves it to the
   * first tabbable control, so a form is ready to type into. `'dialog'` puts it
   * on the dialog box itself, so no answer is preselected: for a dialog that
   * pops up mid-task asking the user to choose (#4721). There a focused button
   * draws the keyboard focus ring when the user was working with the keyboard,
   * reads as the recommended answer, and takes the next Enter or Space. Tab
   * still steps into the controls.
   */
  readonly initialFocus = input<'first-control' | 'dialog'>('first-control');
  readonly closed = output<void>();

  /** The dialog box (`.modal-content`), for projected content that places
   *  itself against the box's outline rather than inside it. */
  readonly box = viewChild<ElementRef<HTMLElement>>('box');

  /** The `<dialog>` element, while the dialog is open (it renders only then).
   *  Page-level overlays move into the topmost one (`vtAboveDialogs`). */
  readonly dialog = viewChild<ElementRef<HTMLDialogElement>>('dialog');

  /** What had focus when the dialog opened, to hand focus back to. */
  private returnFocus: HTMLElement | null = null;

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
        syncTopOpenModal();
      });
    });

    // Open the <dialog> modally as soon as it renders, and hand focus back
    // when it goes away. showModal() records the trigger itself, but only
    // close() returns focus to it: removing an open dialog from the DOM, which
    // is how it goes (this template's @if, or the parent's taking the whole
    // component away), drops focus to <body>. So the trigger is kept here.
    effect(() => {
      const el = this.dialog()?.nativeElement;
      untracked(() => {
        if (!el) {
          this.restoreFocus();
        } else if (!el.open && el.isConnected) {
          this.returnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
          el.showModal();
        }
      });
    });
  }

  ngOnDestroy(): void {
    const idx = openModalStack.indexOf(this);
    if (idx !== -1) openModalStack.splice(idx, 1);
    syncTopOpenModal();
    this.restoreFocus();
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
   * The browser closed the dialog itself, which `closedby="none"` rules out
   * where it is supported: a browser without it let Escape through. Keep the
   * element modal and ask the parent to close it, as our own Escape would, so
   * the element and the parent's `[open]` never disagree.
   */
  onNativeClose(): void {
    const el = this.dialog()?.nativeElement;
    if (!el?.isConnected || el.open || !this.open()) return;
    el.showModal();
    this.close();
  }

  /** Hand focus back to what had it when the dialog opened, unless something
   *  has taken focus since (it is on `<body>` only when it was in the dialog). */
  private restoreFocus(): void {
    const target = this.returnFocus;
    this.returnFocus = null;
    const active = document.activeElement;
    if (target?.isConnected && (active === null || active === document.body)) target.focus();
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
   *
   * The keypress is claimed (`preventDefault`), so a browser that does not
   * know `closedby` does not also close the `<dialog>` natively.
   */
  @HostListener('document:keydown.escape', ['$event'])
  onEscape(event: Event): void {
    if (this.open() && openModalStack[openModalStack.length - 1] === this) {
      event.preventDefault();
      this.close();
    }
  }
}
