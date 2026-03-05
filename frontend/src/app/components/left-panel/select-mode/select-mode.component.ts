import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SelectMode } from '../left-panel.component';

@Component({
  selector: 'vt-select-mode',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './select-mode.component.html',
  styleUrl: './select-mode.component.scss',
})
export class SelectModeComponent {
  @Input() selectMode: SelectMode = 'top';

  @Output() selectModeChange = new EventEmitter<SelectMode>();

  onChange(mode: SelectMode): void {
    this.selectModeChange.emit(mode);
  }
}
