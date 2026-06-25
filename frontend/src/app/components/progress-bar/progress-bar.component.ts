import { ChangeDetectionStrategy, Component, Input } from '@angular/core';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-progress-bar',
  standalone: true,
  templateUrl: './progress-bar.component.html',
  styleUrl: './progress-bar.component.scss',
})
export class ProgressBarComponent {
  @Input() value = 0;
  @Input() max = 100;
  @Input() indeterminate = false;
  /**
   * Opt-in for multi-stage jobs whose `value` is a single whole-job fraction
   * stitched from several phases (the dataset-load bar). It swaps the snappy
   * default fill transition for a longer ease so the unavoidable between-phase
   * jumps (an un-measurable phase filling its slice in one step) glide instead
   * of snapping, keeping the illusion of one continuous process. The fill still
   * only ever eases *toward* the real reported value, so it never overstates
   * progress.
   */
  @Input() smooth = false;

  get percentage(): number {
    if (this.max <= 0) return 0;
    return Math.min(100, (this.value / this.max) * 100);
  }
}
