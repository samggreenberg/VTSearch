import { ChangeDetectionStrategy, Component, computed, effect, input, output, untracked } from '@angular/core';
import { FieldHintIconComponent } from '../../field-hint-icon/field-hint-icon.component';
import {
  FLOOR_PRESETS,
  DEFAULT_MIN_PRECISION,
  floorExplanation,
  floorSummary,
  isFloorPreset,
  nearestFloorPreset,
  type LineFloor,
} from '../../../utils/line-floor';

/** The dot colour for each state, in `.labeling-indicator[data-status]` terms. */
type Dot = 'green' | 'yellow' | 'red' | 'none';

/**
 * The precision floor: "show me what's at least X% right" (#4246). Replaced the
 * Inclusion stepper; mounted in the Find row and the Manual tab.
 *
 * A `<select>` of three named floors (`FLOOR_PRESETS`): "Lean: Complete /
 * Centered / Correct" (#4298), rather than a free number or a percentage (the
 * spot check rarely delivers a floor exactly, so the control promises only a
 * direction). A stored floor off the list (a pick from before #4298, or one set
 * through the CLI or the API) is snapped to the nearest preset, through the
 * same `valueChange` a pick goes out on, once no sort is running (see the
 * constructor).
 *
 * It is also not a range slider on purpose: `KeyboardService.isTyping()` lets
 * ArrowLeft/Right through from a focused `type="range"` input and casts a vote
 * with them, while a focused `<select>` keeps its keys to itself - and gives
 * focus back once a floor is picked (see {@link onChange}).
 *
 * Under the picker, one line says what the floor does to the current line -
 * the floor and its state, never the estimate behind it (owner, 2026-09-28):
 * the promise kept with the count, can't reach the floor (showing the default
 * cut), or not enough evidence yet (showing the default cut, with the Good
 * votes it has). There is no "off": every detector has a floor (#4269).
 *
 * Content marked `floorActions` is projected onto the picker's line, so a host
 * can seat controls beside the picker (Find's work-queue actions) while the
 * state line keeps the full width underneath.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-precision-floor',
  standalone: true,
  imports: [FieldHintIconComponent],
  templateUrl: './precision-floor.component.html',
  styleUrl: './precision-floor.component.scss',
})
export class PrecisionFloorComponent {
  /** The detector's floor, a fraction. */
  readonly value = input<number>(DEFAULT_MIN_PRECISION);
  /** The floor's verdict on the line the list draws; null when there is no line from the detector. */
  readonly floor = input<LineFloor | null>(null);
  /** How many items the line returns; null when unknown. */
  readonly returned = input<number | null>(null);
  /** True while a sort or Find pass runs; a snap to a preset waits for it. */
  readonly busy = input(false);

  /** A floor the user picked, as a fraction. */
  readonly valueChange = output<number>();

  readonly hint =
    'Lean the line toward Complete, to return as much as it can at the cost of more wrong ones, or toward ' +
    'Correct, to return less with little of it wrong; Centered sits between them. The line can only promise ' +
    'its lean once it has enough evidence. Until then, or when no line on this dataset gets there, it stays ' +
    'at the default cut and is marked unpromised.';

  readonly options = FLOOR_PRESETS.map((p) => ({ value: String(p.value), label: p.name }));

  /** The `<select>`'s current value: a floor off the list shows as the preset it will snap to. */
  readonly selected = computed(() => String(nearestFloorPreset(this.value()).value));

  readonly summary = computed(() => floorSummary(this.floor(), this.returned()));
  readonly explanation = computed(() => floorExplanation(this.floor(), this.returned()));

  readonly dot = computed<Dot>(() => {
    switch (this.floor()?.status) {
      case 'promised':
        return 'green';
      case 'insufficient_evidence':
        return 'yellow';
      case 'unreachable':
        return 'red';
      default:
        return 'none';
    }
  });

  constructor() {
    // Snap a stored floor off the list to the nearest preset, so the line is
    // cut at the floor the picker names. It goes out as a pick would, and waits
    // for any running sort: Find drops a floor change mid-pass, since the POST
    // would re-cut a line the pass is still drawing.
    effect(() => {
      const v = this.value();
      if (this.busy() || isFloorPreset(v)) return;
      const snapped = nearestFloorPreset(v).value;
      untracked(() => this.valueChange.emit(snapped));
    });
  }

  /**
   * Emit the picked floor, and hand focus back to the document.
   *
   * The blur is not a nicety: a focused `<select>` keeps the arrow keys, and
   * Chrome (off macOS) steps a closed one through its options on
   * ArrowLeft/Right. Left focused, the picker would turn the user's next
   * "vote Good" (→) into a floor change. Picking ends the task, as submitting
   * does for the text sort (`SortBarComponent.submitTextSort`).
   */
  onChange(event: Event): void {
    const select = event.target as HTMLSelectElement;
    const raw = select.value;
    const val = Number(raw);
    select.blur();
    if (!Number.isFinite(val) || val <= 0 || val > 1) return;
    this.valueChange.emit(val);
  }
}
