import { Component, inject } from '@angular/core';

import { FormsModule } from '@angular/forms';
import { ModalComponent } from '../modal/modal.component';
import { IconComponent } from '../icon/icon.component';
import { VtDialogService } from '../../services/dialog.service';

@Component({
  selector: 'vt-dialog-host',
  standalone: true,
  imports: [FormsModule, ModalComponent, IconComponent],
  templateUrl: './dialog-host.component.html',
  styleUrl: './dialog-host.component.scss',
})
export class DialogHostComponent {
  dialog = inject(VtDialogService);


  onButtonClick(value: unknown): void {
    this.dialog.resolve(value);
  }

  onClosed(): void {
    this.dialog.resolve(false);
  }
}
