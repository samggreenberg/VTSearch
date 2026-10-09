import { ChangeDetectionStrategy, Component, computed, input, output } from '@angular/core';

/**
 * The fold control for a side panel of Train and Test (#4673).
 *
 * Folded, it is the whole strip the panel leaves behind: one button the height
 * of the pane, with an arrow pointing back into the view and the panel's name
 * written down it, so the target is large and says what it opens. Open, it is
 * a slim row holding one small arrow on the edge nearest the centre, the same
 * control `vt-autopilot-panel` puts at the head of Train's Autopilot tab.
 *
 * The view owns the state (`PanelHideStateService`); this draws it and reports
 * the click.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-side-panel-toggle',
  standalone: true,
  templateUrl: './side-panel-toggle.component.html',
  styleUrl: './side-panel-toggle.component.scss',
  host: { '[class.folded]': 'collapsed()' },
})
export class SidePanelToggleComponent {
  readonly side = input.required<'left' | 'right'>();
  readonly collapsed = input(false);
  /** The panel's name: written down the strip, and in the button's tooltip. */
  readonly label = input.required<string>();

  readonly toggle = output<void>();

  /** Points the way the panel's edge moves on a click. */
  readonly arrow = computed(() => ((this.side() === 'left') === this.collapsed() ? '▶' : '◀'));
  readonly title = computed(
    () => `${this.collapsed() ? 'Show' : 'Hide'} ${this.label().toLowerCase()} panel`,
  );
}
