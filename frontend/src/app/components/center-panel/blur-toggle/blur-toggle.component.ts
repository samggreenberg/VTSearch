import { ChangeDetectionStrategy, Component, Signal, computed, input } from '@angular/core';

/** What a {@link BlurToggleComponent} flips: a focused viewer of one dataset item. */
export interface BlurToggleTarget {
  /** Whether the item is drawn blurred now. */
  readonly blurred: Signal<boolean>;
  /** Flip the item's state for good, in the dataset it is in. */
  toggleBlur(): void;
}

/**
 * The **Blur** / **Unblur** button (#5689): one press flips the focused item's
 * state for good, everywhere it appears and for everyone, in either direction.
 * The image viewer's toolbar ends with it, and the centre panel puts it under
 * a video (#5698); both answer `b` as well. It reads and writes nothing
 * itself: the viewer it is given knows which dataset its item is in.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-blur-toggle',
  standalone: true,
  template: `
    <button
      type="button"
      class="btn btn--toolbar blur-toggle"
      [attr.data-metric]="metric()"
      (click)="target().toggleBlur()"
      [title]="title()"
      [attr.aria-pressed]="target().blurred()"
    >{{ target().blurred() ? 'Unblur' : 'Blur' }}</button>
  `,
  styleUrl: './blur-toggle.component.scss',
})
export class BlurToggleComponent {
  /** The viewer whose item the button flips. */
  readonly target = input.required<BlurToggleTarget>();
  /** What the item is, for the button's tooltip and its usage metric. */
  readonly noun = input<'image' | 'video'>('image');

  readonly metric = computed(() => `${this.noun()}Blur`);
  readonly title = computed(() =>
    this.target().blurred()
      ? `Unblur: show this ${this.noun()} sharp, everywhere it appears and for everyone (B)`
      : `Blur: hide this ${this.noun()} behind a blur, everywhere it appears and for everyone (B)`,
  );
}
