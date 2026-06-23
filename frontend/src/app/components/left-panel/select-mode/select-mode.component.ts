import { ChangeDetectionStrategy, Component, Input, output } from '@angular/core';

import { SelectMode } from '../left-panel.component';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-select-mode',
  standalone: true,
  imports: [],
  templateUrl: './select-mode.component.html',
  styleUrl: './select-mode.component.scss',
})
export class SelectModeComponent {
  @Input() selectMode: SelectMode = 'top';

  readonly selectModeChange = output<SelectMode>();

  onChange(mode: SelectMode): void {
    this.selectModeChange.emit(mode);
  }
}
