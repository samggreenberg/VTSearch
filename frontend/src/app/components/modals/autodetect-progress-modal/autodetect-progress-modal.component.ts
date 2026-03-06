import { Component, EventEmitter, Input, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ModalComponent } from '../../modal/modal.component';
import { ProgressBarComponent } from '../../progress-bar/progress-bar.component';

@Component({
  selector: 'vt-autodetect-progress-modal',
  standalone: true,
  imports: [CommonModule, ModalComponent, ProgressBarComponent],
  templateUrl: './autodetect-progress-modal.component.html',
  styleUrl: './autodetect-progress-modal.component.scss',
})
export class AutoDetectProgressModalComponent {
  @Input() progress = 0;
  @Input() statusText = 'Initializing...';
  @Output() closed = new EventEmitter<void>();
  @Output() cancelled = new EventEmitter<void>();

  onCancel(): void {
    this.cancelled.emit();
  }

  close(): void {
    this.closed.emit();
  }
}
