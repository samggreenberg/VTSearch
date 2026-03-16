import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'vt-inclusion-slider',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './inclusion-slider.component.html',
  styleUrl: './inclusion-slider.component.scss',
})
export class InclusionSliderComponent {
  @Input() value = 0;

  @Output() valueChange = new EventEmitter<number>();

  onInput(event: Event): void {
    const target = event.target as HTMLInputElement;
    const val = parseInt(target.value, 10);
    if (!isNaN(val) && val >= -10 && val <= 10) {
      this.valueChange.emit(val);
    }
  }
}
