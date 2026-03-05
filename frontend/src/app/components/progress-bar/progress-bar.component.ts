import { Component, Input } from '@angular/core';

@Component({
  selector: 'vt-progress-bar',
  standalone: true,
  templateUrl: './progress-bar.component.html',
  styleUrl: './progress-bar.component.scss',
})
export class ProgressBarComponent {
  @Input() value = 0;
  @Input() max = 100;
  @Input() indeterminate = false;

  get percentage(): number {
    if (this.max <= 0) return 0;
    return Math.min(100, (this.value / this.max) * 100);
  }
}
