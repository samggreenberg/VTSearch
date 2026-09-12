import { ChangeDetectionStrategy, Component, inject } from '@angular/core';

import { ModalComponent } from '../modal/modal.component';
import { IconComponent } from '../icon/icon.component';
import { VtDialogService } from '../../services/dialog.service';

// Deliberately does NOT import FormsModule.  This host is rendered
// unconditionally by AppComponent, so it sits on the EAGER path: importing it
// here pulled all of @angular/forms (~37kB) into the initial bundle for the
// single prompt() text field.  `[value]` + `(input)` is equivalent here; see
// the bundle-budget note in angular.json and #3811.
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-dialog-host',
  standalone: true,
  imports: [ModalComponent, IconComponent],
  templateUrl: './dialog-host.component.html',
  styleUrl: './dialog-host.component.scss',
})
export class DialogHostComponent {
  dialog = inject(VtDialogService);


  onButtonClick(value: unknown): void {
    this.dialog.resolve(value);
  }

  onClosed(): void {
    // Escape / backdrop = the dialog kind's own cancel value (null for
    // prompt(), false for confirm()), not a hard-coded false.
    this.dialog.cancel();
  }
}
