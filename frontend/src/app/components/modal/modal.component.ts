import { ChangeDetectionStrategy, Component, HostListener, Input, input, output } from '@angular/core';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-modal',
  standalone: true,
  templateUrl: './modal.component.html',
  styleUrl: './modal.component.scss',
})
export class ModalComponent {
  readonly title = input('');
  readonly open = input(false);
  @Input() showCloseButton = true;
  readonly closed = output<void>();

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
   */
  @HostListener('document:keydown.escape')
  onEscape(): void {
    if (this.open()) {
      this.close();
    }
  }
}
