import { ChangeDetectionStrategy, Component, input } from '@angular/core';

/**
 * Which direction of a progress bar reads as "good", picking the fill gradient.
 * - `'high-good'`: a fuller bar is better (progress toward a goal).
 * - `'high-bad'`: a fuller bar is worse (a consumed resource like disk or RAM).
 */
export type ProgressPolarity = 'high-good' | 'high-bad';

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-progress-bar',
  standalone: true,
  templateUrl: './progress-bar.component.html',
  styleUrl: './progress-bar.component.scss',
})
export class ProgressBarComponent {
  readonly value = input(0);
  readonly max = input(100);
  readonly indeterminate = input(false);
  /**
   * Opt-in for multi-stage jobs whose `value` is a single whole-job fraction
   * stitched from several phases (the dataset-load bar). It swaps the snappy
   * default fill transition for a longer ease so the unavoidable between-phase
   * jumps (an un-measurable phase filling its slice in one step) glide instead
   * of snapping, keeping the illusion of one continuous process. The fill still
   * only ever eases *toward* the real reported value, so it never overstates
   * progress.
   */
  readonly smooth = input(false);
  /**
   * Whether a fuller bar is good or bad, which picks the direction of the
   * red -> yellow -> green fill gradient:
   * - `'high-good'` (default): reddest empty, greenest full. Use it for
   *   "progress toward a goal" bars (the achievements bar), where more is
   *   better and green rewards it.
   * - `'high-bad'`: greenest empty, reddest full. Use it for consumed-resource
   *   gauges (disk / RAM), where a fuller bar is *worse* so it should redden as
   *   it fills.
   * The two are mirror images: `'high-bad'` colors by `100 - percentage`.
   */
  readonly polarity = input<ProgressPolarity>('high-good');

  get percentage(): number {
    if (this.max() <= 0) return 0;
    return Math.min(100, (this.value() / this.max()) * 100);
  }

  /**
   * Continuous red -> yellow -> green fill color, matching the CMD progress
   * bar gradient. For `'high-good'` (the default) it is reddest at 0%,
   * yellowest at 50%, greenest at 100%; for `'high-bad'` the ramp is mirrored
   * (greenest empty, reddest full) by coloring off `100 - percentage`. Built
   * from the theme's `--color-bad` / `--text-warning` / `--color-good`
   * variables via `color-mix` so it stays correct in every theme (light,
   * dark, high-visibility) without hardcoding RGB values here.
   *
   * Returns `null` while indeterminate so the element falls back to the
   * SCSS default (`--accent`); there is no percentage to color by.
   */
  get fillColor(): string | null {
    if (this.indeterminate()) return null;
    const pct = this.polarity() === 'high-bad' ? 100 - this.percentage : this.percentage;
    if (pct <= 50) {
      const t = (pct / 50) * 100;
      return `color-mix(in srgb, var(--text-warning) ${t}%, var(--color-bad))`;
    }
    const t = ((pct - 50) / 50) * 100;
    return `color-mix(in srgb, var(--color-good) ${t}%, var(--text-warning))`;
  }
}
