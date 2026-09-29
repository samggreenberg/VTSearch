import { ChangeDetectionStrategy, Component, computed, input, output } from '@angular/core';
import { FieldHintIconComponent } from '../../field-hint-icon/field-hint-icon.component';
import {
  FLOOR_PRESETS,
  DEFAULT_MIN_PRECISION,
  floorExplanation,
  floorPercent,
  floorSummary,
  type LineFloor,
} from '../../../utils/line-floor';

/** The dot colour for each state, in `.labeling-indicator[data-status]` terms. */
type Dot = 'green' | 'yellow' | 'red' | 'none';

/**
 * The precision floor: "show me what's at least X% right" (#4246). Replaced the
 * Inclusion stepper; mounted in the Find row and the Manual tab.
 *
 * A `<select>` of the four floors #4220 priced, rather than a free number:
 * every choice is a measured one (owner, 2026-09-28). It is also not a range
 * slider on purpose: `KeyboardService.isTyping()` lets ArrowLeft/Right through
 * from a focused `type="range"` input and casts a vote with them, while a
 * focused `<select>` keeps its keys to itself - and gives focus back once a
 * floor is picked (see {@link onChange}).
 *
 * Under the picker, one line says what the floor does to the current line -
 * the floor and its state, never the estimate behind it (owner, 2026-09-28):
 * at least X% right with the count, can't reach X% (showing the default cut),
 * or not enough evidence yet (showing the default cut, with the Good votes it
 * has). There is no "off": every detector has a floor (#4269).
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

  /** A floor the user picked, as a fraction. */
  readonly valueChange = output<number>();

  readonly hint =
    'Pick how much of what the detector returns should be right. The line then returns as much as it can ' +
    'while at least that share of it is estimated right. It can only promise that once it has enough ' +
    'evidence; until then, or when no line on this dataset gets there, the line stays at the default cut ' +
    'and is marked unpromised.';

  /** The presets, plus the current value when it is not one of them, in order. */
  readonly options = computed(() => {
    const v = this.value();
    const all = FLOOR_PRESETS.includes(v) ? [...FLOOR_PRESETS] : [...FLOOR_PRESETS, v];
    return all.sort((a, b) => a - b).map((p) => ({ value: String(p), label: floorPercent(p) }));
  });

  /** The `<select>`'s current value. */
  readonly selected = computed(() => String(this.value()));

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
