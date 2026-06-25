import { ChangeDetectionStrategy, Component, input, output } from '@angular/core';


@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-inclusion-slider',
  standalone: true,
  imports: [],
  templateUrl: './inclusion-slider.component.html',
  styleUrl: './inclusion-slider.component.scss',
})
export class InclusionSliderComponent {
  readonly value = input(0);

  readonly valueChange = output<number>();

  onInput(event: Event): void {
    const target = event.target as HTMLInputElement;
    const val = parseInt(target.value, 10);
    if (!isNaN(val) && val >= -10 && val <= 10) {
      this.valueChange.emit(val);
    }
  }
}
