import { ChangeDetectionStrategy, Component, Input, output } from '@angular/core';
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
  changeDetection: ChangeDetectionStrategy.OnPush,
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
