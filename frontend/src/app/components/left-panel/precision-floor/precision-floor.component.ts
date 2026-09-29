import { ChangeDetectionStrategy, Component, computed, effect, input, output, untracked } from '@angular/core';
import { FieldHintIconComponent } from '../../field-hint-icon/field-hint-icon.component';
import {
  FLOOR_PRESETS,
  DEFAULT_MIN_PRECISION,
  checkLabel,
  checkTitle,
  floorExplanation,
  floorSummary,
  isFloorPreset,
  nearestFloorPreset,
  type LineFloor,
} from '../../../utils/line-floor';

/** The dot colour for each state, in `.labeling-indicator[data-status]` terms. */
type Dot = 'green' | 'yellow' | 'red' | 'none';

/**
 * The Threshold control (#4317): the precision floor, "show me what's at least
 * X% right" (#4246). Mounted in the Find row and the Manual tab.
 *
 * A horizontal spectrum from False Positives to False Negatives, with three
 * radios under it (`FLOOR_PRESETS`), one centred under each third. No radio
 * carries a word or a number: where it sits on the spectrum is the whole
 * message, and its tooltip says it in words. The spot check rarely delivers a
 * floor exactly, so the control promises only a direction (#4298). A stored
 * floor off the list (one set through the CLI or the API) is snapped to the
 * nearest preset, through the same `valueChange` a pick goes out on, once no
 * sort is running (see the constructor).
 *
 * The radios show the host's floor, never the click: a pick cancels the
 * click's default and emits, and `[checked]` follows `value()`. A host that
 * drops the pick (Find, mid-pass) leaves the radios where they were. It is not
 * a range slider on purpose: a focused `type="range"` input would move on the
 * very arrow keys that cast votes. A picked radio gives focus back (see
 * {@link onPick}).
 *
 * Under the spectrum, one line says what the floor does to the current line -
 * the set the line keeps and how close the spot check got, never an estimate
 * from the model (owner, 2026-09-29, #4272): confirmed with the check's likely
 * range and the count kept, the range a short check found, or the top N kept
 * unchecked. There is no "off": every detector has a floor (#4269).
 *
 * In Train, the check affordance sits beside that line, "Check 5 picks"
 * (#4273), in every state: it opens the spot check (`vt-floor-check-modal`,
 * hosted by the view), and after a finished check it runs a fresh one. Find
 * sets `offerCheck` false: it tests the threshold Train set, and labelling
 * more to set one is too late there (#4317). A range that later votes have
 * left stale reads exactly as before; only its tooltip says so.
 *
 * Content marked `floorActions` is projected onto the Threshold heading's
 * line, so a host can seat controls there (Find's work-queue actions).
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
  private static nextId = 0;

  /** The detector's floor, a fraction. */
  readonly value = input<number>(DEFAULT_MIN_PRECISION);
  /** The floor's verdict on the line the list draws; null when there is no line from the detector. */
  readonly floor = input<LineFloor | null>(null);
  /** How many items the line returns; the state line reads the kept count off the floor instead. */
  readonly returned = input<number | null>(null);
  /** True while a sort or Find pass runs; a snap to a preset waits for it. */
  readonly busy = input(false);
  /** False while the host can't run a check (a sort or a scoring pass in flight, or the panel disabled). */
  readonly checkable = input(true);
  /** False where the host offers no spot check at all: Find (#4317). */
  readonly offerCheck = input(true);

  /** A floor the user picked, as a fraction. */
  readonly valueChange = output<number>();
  /** The user asked for a spot check of the line. */
  readonly check = output<void>();

  readonly hint =
    'Where to draw the line. Toward False Positives returns more, with more wrong ones in it; ' +
    'toward False Negatives returns only the surest, and misses more.';

  readonly options = FLOOR_PRESETS;

  /** One radio group, and one heading, per control. */
  readonly groupName = `precision-floor-${PrecisionFloorComponent.nextId++}`;
  readonly labelId = `${this.groupName}-label`;

  /** The radio the floor shows: a floor off the list shows as the preset it will snap to. */
  readonly selected = computed(() => nearestFloorPreset(this.value()).value);

  readonly summary = computed(() => floorSummary(this.floor()));
  readonly explanation = computed(() => floorExplanation(this.floor()));
  readonly checkText = computed(() => (this.offerCheck() ? checkLabel(this.floor()) : null));
  readonly checkHint = computed(() => checkTitle(this.floor()));

  readonly dot = computed<Dot>(() => {
    switch (this.floor()?.status) {
      case 'confirmed':
        return 'green';
      case 'unchecked':
        return 'yellow';
      case 'short':
        return 'red';
      default:
        return 'none';
    }
  });

  constructor() {
    // Snap a stored floor off the list to the nearest preset, so the line is
    // cut at the floor the radios show. It goes out as a pick would, and waits
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
   * The click's default is cancelled, so the radio doesn't check itself:
   * `[checked]` follows `value()`, and a pick the host drops leaves the radios
   * showing the floor it kept. The blur ends the task, as submitting does for
   * the text sort (`SortBarComponent.submitTextSort`), so the next arrow key
   * is a vote with nothing focused.
   */
  onPick(event: Event, value: number): void {
    event.preventDefault();
    (event.target as HTMLElement).blur();
    if (value === this.selected()) return;
    this.valueChange.emit(value);
  }
}
