import { Component, Input, output } from '@angular/core';
import { FormsModule } from '@angular/forms';

export type LabelSortMode =
  | 'time-desc'
  | 'time-asc'
  | 'name-asc'
  | 'name-desc'
  | 'confidence-desc'
  | 'confidence-asc'
  | 'id-asc';

@Component({
  selector: 'vt-label-sort',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './label-sort.component.html',
  styleUrl: './label-sort.component.scss',
})
export class LabelSortComponent {
  @Input() mode: LabelSortMode = 'time-desc';
  readonly modeChange = output<LabelSortMode>();

  onSortChange(value: LabelSortMode): void {
    this.mode = value;
    this.modeChange.emit(value);
  }
}
