import { Component, EventEmitter, Output } from '@angular/core';
import { ModalComponent } from '../../modal/modal.component';
import { AchievementsTabComponent } from '../../achievements-tab/achievements-tab.component';

@Component({
  selector: 'vt-achievements-modal',
  standalone: true,
  imports: [ModalComponent, AchievementsTabComponent],
  templateUrl: './achievements-modal.component.html',
})
export class AchievementsModalComponent {
  @Output() closed = new EventEmitter<void>();

  close(): void {
    this.closed.emit();
  }
}
