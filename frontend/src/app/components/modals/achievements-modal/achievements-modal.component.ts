import { Component, output } from '@angular/core';
import { ModalComponent } from '../../modal/modal.component';
import { AchievementsTabComponent } from '../../achievements-tab/achievements-tab.component';

@Component({
  selector: 'vt-achievements-modal',
  standalone: true,
  imports: [ModalComponent, AchievementsTabComponent],
  templateUrl: './achievements-modal.component.html',
})
export class AchievementsModalComponent {
  readonly closed = output<void>();

  close(): void {
    this.closed.emit();
  }
}
