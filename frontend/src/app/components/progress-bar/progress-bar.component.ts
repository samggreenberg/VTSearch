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

  /**
   * Continuous red -> yellow -> green fill color, matching the CMD progress
   * bar gradient: reddest at 0%, yellowest at 50%, greenest at 100%. Built
   * from the theme's `--color-bad` / `--text-warning` / `--color-good`
   * variables via `color-mix` so it stays correct in every theme (light,
   * dark, high-visibility) without hardcoding RGB values here.
   *
   * Returns `null` while indeterminate so the element falls back to the
   * SCSS default (`--accent`); there is no percentage to color by.
   */
  get fillColor(): string | null {
    if (this.indeterminate) return null;
    const pct = this.percentage;
    if (pct <= 50) {
      const t = (pct / 50) * 100;
      return `color-mix(in srgb, var(--text-warning) ${t}%, var(--color-bad))`;
    }
    const t = ((pct - 50) / 50) * 100;
    return `color-mix(in srgb, var(--color-good) ${t}%, var(--text-warning))`;
  }
}
