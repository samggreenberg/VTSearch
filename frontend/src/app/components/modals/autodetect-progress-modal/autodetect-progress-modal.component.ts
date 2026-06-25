import { ChangeDetectionStrategy, Component, Input, output } from '@angular/core';
import { ModalComponent } from '../../modal/modal.component';
import { ProgressBarComponent } from '../../progress-bar/progress-bar.component';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-autodetect-progress-modal',
  standalone: true,
  imports: [ModalComponent, ProgressBarComponent],
  templateUrl: './autodetect-progress-modal.component.html',
  styleUrl: './autodetect-progress-modal.component.scss',
})
export class AutoDetectProgressModalComponent {
  @Input() progress = 0;
  @Input() statusText = 'Initializing...';
  readonly closed = output<void>();
  readonly cancelled = output<void>();

  onCancel(): void {
    this.cancelled.emit();
  }

  close(): void {
    this.closed.emit();
  }
}
