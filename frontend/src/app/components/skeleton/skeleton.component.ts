import { Component, input } from '@angular/core';
import { NgStyle } from '@angular/common';

@Component({
  selector: 'vt-skeleton',
  standalone: true,
  imports: [NgStyle],
  templateUrl: './skeleton.component.html',
  styleUrl: './skeleton.component.scss',
})
export class SkeletonComponent {
  readonly width = input('100%');
  readonly height = input('12px');
  readonly borderRadius = input('var(--radius-sm)');

  get boxStyle(): Record<string, string> {
    return {
      width: this.width(),
      height: this.height(),
      'border-radius': this.borderRadius(),
    };
  }
}
