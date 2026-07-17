import { ChangeDetectionStrategy, Component, model } from '@angular/core';
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
  // Two-way bindable: `[mode]` seeds it and the implicit `modeChange` output
  // fires on selection, so `this.mode.set` both updates the local value and
  // notifies the parent.
  readonly mode = model<LabelSortMode>('time-desc');

  onSortChange(value: LabelSortMode): void {
    this.mode.set(value);
  }
}
