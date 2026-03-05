import { Component, EventEmitter, Input, Output } from '@angular/core';

@Component({
  selector: 'vt-modal',
  standalone: true,
  templateUrl: './modal.component.html',
  styleUrl: './modal.component.scss',
})
export class ModalComponent {
  @Input() title = '';
  @Input() open = false;
  @Output() closed = new EventEmitter<void>();

  onBackdropClick(event: MouseEvent): void {
    if (event.target === event.currentTarget) {
      this.close();
    }
  }

  close(): void {
    this.closed.emit();
  }

  onKeydown(event: KeyboardEvent): void {
    if (event.key === 'Escape') {
      this.close();
    }
  }
}
