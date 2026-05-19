import { Component, Input } from '@angular/core';
import { NgStyle } from '@angular/common';

@Component({
  selector: 'vt-skeleton',
  standalone: true,
  imports: [NgStyle],
  templateUrl: './skeleton.component.html',
  styleUrl: './skeleton.component.scss',
})
export class SkeletonComponent {
  @Input() width = '100%';
  @Input() height = '12px';
  @Input() borderRadius = 'var(--radius-sm)';

  get boxStyle(): Record<string, string> {
    return {
      width: this.width,
      height: this.height,
      'border-radius': this.borderRadius,
    };
  }
}
