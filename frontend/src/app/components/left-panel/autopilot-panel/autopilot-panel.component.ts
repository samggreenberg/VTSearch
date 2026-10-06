import {
  ChangeDetectionStrategy,
  Component,
  effect,
  inject,
  input,
  OnInit,
  output,
  signal,
} from '@angular/core';
import { Router } from '@angular/router';

import type { LabelingStatusResponse } from '../../../generated/api-client/models/labeling-status-response';
import {
  AutopilotStateService,
  AutopilotPhase,
  AutopilotState,
} from '../../../services/autopilot-state.service';
import { IconComponent } from '../../icon/icon.component';
import { AutopilotCompleteModalComponent } from '../../modals/autopilot-complete-modal/autopilot-complete-modal.component';

export type { AutopilotPhase, AutopilotState };

/**
 * A step's one light (#4319). It climbs red -> yellow -> green as the step
 * nears its end, and the step hands over to the next one at green. Red and
 * yellow render as a circle, green as the check a finished step keeps.
 */
type LightColor = 'red' | 'yellow' | 'green';

interface StepLight {
  color: LightColor;
  ariaLabel: string;
  title: string;
}

/** Ordered so the lower of two lights is the one with the lower rank. */
const LIGHT_RANK: Record<LightColor, number> = { red: 0, yellow: 1, green: 2 };

/**
 * Light for a step that ends at a target: red for the first half of the way,
 * yellow for the second half, green once the target is met. A target the
 * dataset can't supply (capped to 0) is already met.
 */
function progressLight(count: number, target: number): LightColor {
  if (count >= target) return 'green';
  return count * 2 >= target ? 'yellow' : 'red';
}

/** A backend indicator's status as a light. Unreported (``''``) reads red. */
function indicatorLight(status: string): LightColor {
  return status === 'green' || status === 'yellow' ? status : 'red';
}

/** Leads each light's tooltip, so the color is stated as well as shown. */
const LIGHT_WORDS: Record<LightColor, string> = { red: 'Red.', yellow: 'Yellow.', green: 'Green.' };

/** How a count step's light reads, for its tooltip. */
function countTitle(target: number, kind: 'good' | 'bad'): string {
  return `Red until half of the ${target} ${kind} labels are in, yellow past halfway, green at ${target}.`;
}

export interface StepDisplay {
  phase: AutopilotPhase;
  label: string;
  shortLabel: string;
  stepNumber: number;
  state: 'done' | 'active' | 'future';
  detail: string;
  detailTitle: string;
  /** The active step's light; ``null`` on done and future steps. */
  light: StepLight | null;
  helpText: string;
  intent: string;
}

/** Copy for the completion modal, per terminal phase. */
interface CompletionPrompt {
  /** Dialog title: what just finished. */
  heading: string;
  /** Why autopilot stopped. */
  detail: string;
  /** What the user can do now, whichever button they pick. */
  nextSteps: string;
  /** Label for the "don't go anywhere" button. */
  stayLabel: string;
}

/** What the Done step says when every indicator is green and the pool converged. */
const DONE_HELP = 'All quality indicators are green. You can continue labeling or export your results.';

/**
 * The Stable indicator went green on a plateau: the items still changing
 * calls between retrains sit in an ambiguity the embedding cannot resolve, and
 * their rate has stopped falling (#3831). Said plainly so the user does not
 * read "Done" as "the detector is sure about everything".
 */
const PLATEAU_NOTE =
  'All quality indicators are green, but some items near the cutoff still change calls between retrains and that is no longer improving: the remaining ambiguity looks irreducible in this embedding.';
const DONE_PLATEAU_HELP = `${PLATEAU_NOTE} You can continue labeling or export your results.`;

/**
 * What the Done step says on a document dataset, which stops on the dry run
 * rather than the indicators (#4488).
 */
function dryRunNote(dryRun: number): string {
  return `${dryRun} of the detector's best matches in a row were not good: the documents it can find are likely found.`;
}

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-autopilot-panel',
  standalone: true,
  imports: [IconComponent, AutopilotCompleteModalComponent],
  templateUrl: './autopilot-panel.component.html',
  styleUrl: './autopilot-panel.component.scss',
})
export class AutopilotPanelComponent implements OnInit {
  autopilotState = inject(AutopilotStateService);
  private router = inject(Router);

  readonly goodVotes = input<Set<number>>(new Set());
  readonly badVotes = input<Set<number>>(new Set());
  /**
   * Number of items in the active dataset. Feeds the dataset-size-aware phase
   * targets: on a tiny collection the default 3-good / 4-bad targets are
   * unreachable, so they're capped to what the dataset can supply. ``0`` means
   * unknown (still loading); the service leaves the targets uncapped then.
   */
  readonly datasetSize = input(0);
  /**
   * Total "good" labels in the active detector's saved labelset, across all
   * datasets it has been used with.  When both this and ``labelsetBadCount``
   * are positive at activation time, autopilot enters retrain mode and uses
   * learned sort throughout instead of starting with text/example sort.
   */
  readonly labelsetGoodCount = input(0);
  readonly labelsetBadCount = input(0);
  readonly labelingStatus = input<LabelingStatusResponse | null>(null);
  readonly collapsed = input(false);
  /**
   * True once ``/api/votes`` has answered for the active dataset/detector pair.
   * Until then ``labelsetGoodCount`` / ``labelsetBadCount`` are zeroed defaults
   * rather than facts, and a fully-trained detector is indistinguishable from a
   * brand-new one — which is exactly the reading the completion hand-off turns
   * on (see ``AutopilotStateService.noteInitialLabelset``).
   */
  readonly votesLoaded = input(false);

  readonly started = output<void>();
  readonly stopped = output<void>();
  readonly toggleCollapse = output<void>();
  readonly refocus = output<void>();

  /** Copy for the live completion modal, or ``null`` when none is open. */
  readonly completionPrompt = signal<CompletionPrompt | null>(null);

  constructor() {
    // Signal inputs don't fire ``ngOnChanges``; this effect replaces the old
    // change hook. It re-runs whenever the vote sets, dataset size, or labeling
    // status change, driving the phase transitions and the completion hand-off.
    effect(() => {
      // Read every reactive input so the effect tracks them as dependencies.
      const goodVotes = this.goodVotes();
      const badVotes = this.badVotes();
      const datasetSize = this.datasetSize();
      const labelingStatus = this.labelingStatus();
      const votesLoaded = this.votesLoaded();
      const labelsetGoodCount = this.labelsetGoodCount();
      const labelsetBadCount = this.labelsetBadCount();

      if (!this.running) return;

      // Take the run's one reading of "was this detector already trained when
      // we started?" as soon as the labelset is real rather than zeroed. The
      // service ignores every call after the first, so later votes — which of
      // course push these counts above zero — can't rewrite the answer.
      if (votesLoaded) {
        this.autopilotState.noteInitialLabelset(labelsetGoodCount, labelsetBadCount);
      } else {
        // Labelset un-loaded: either we have not started yet, or the active
        // pair just changed under a running autopilot. Either way any reading
        // we hold describes a detector that is no longer on screen.
        this.autopilotState.forgetInitialLabelset();
      }

      if (labelingStatus) {
        this.autopilotState.updateFromLabelingStatus(labelingStatus);
      }

      const prevPhase = this.autopilotState.state.phase;
      this.autopilotState.checkPhaseTransition(goodVotes.size, badVotes.size, datasetSize);
      const phase = this.autopilotState.state.phase;
      if (prevPhase !== phase && this.autopilotState.shouldAnnounceCompletion) {
        if (phase === 'done') {
          this.announceCompletion({
            heading: 'Detector Trained',
            detail: this.state.dryRunStop
              ? dryRunNote(this.state.moreDryRun)
              : 'Every quality indicator is green: the detector\'s accuracy has settled, its calls '
                + 'have stopped shifting between labeling steps, and your votes span a broad mix of '
                + 'the collection.',
            nextSteps:
              'Nothing here expires. Keep labeling to refine the detector further, or head to the '
              + 'Dashboard to export it, run it over another dataset, or start something new.',
            stayLabel: 'Continue Training',
          });
        } else if (phase === 'exhausted') {
          this.announceCompletion({
            heading: 'Nothing Left to Label',
            detail: 'Autopilot has labeled every item in this dataset.',
            nextSteps:
              'Stay here to review your votes, or head to the Dashboard to export the detector or '
              + 'run it over another dataset.',
            stayLabel: 'Stay Here',
          });
        }
      }
    });
  }

  get state(): AutopilotState {
    return this.autopilotState.state;
  }

  get running(): boolean {
    return this.autopilotState.running;
  }

  /** Terminal state: every item labeled but the indicators never went green
   *  (typical of a tiny dataset that can't reach the good+bad quorum). */
  get exhausted(): boolean {
    return this.state.phase === 'exhausted';
  }

  /** Items still carrying no vote, or ``Infinity`` while the dataset size is
   *  unknown or inconsistent with the vote counts (mirrors the service's
   *  uncapped behavior during load; see ``checkPhaseTransition``). */
  private get remainingUnlabeled(): number {
    const datasetSize = this.datasetSize();
    const raw = datasetSize - this.goodVotes().size - this.badVotes().size;
    if (datasetSize <= 0 || raw < 0) return Infinity;
    return raw;
  }

  /** Good-vote target for the current dataset, capped to what it can supply. */
  get effGoodTarget(): number {
    return Math.min(this.state.goodToStart, this.goodVotes().size + this.remainingUnlabeled);
  }

  /** Bad-vote target for the current dataset, capped to what it can supply. */
  get effBadTarget(): number {
    return Math.min(this.state.badToStart, this.badVotes().size + this.remainingUnlabeled);
  }

  /** The "more" walk's Good target, capped to what the dataset can supply. */
  get effMoreTarget(): number {
    return Math.min(this.state.moreToStart, this.goodVotes().size + this.remainingUnlabeled);
  }

  get steps(): StepDisplay[] {
    // A document dataset stops on the walk's dry run (#4488): no Boundary or
    // Diversity step.
    const phases: AutopilotPhase[] = this.state.dryRunStop
      ? ['good', 'bad', 'more', 'done']
      : ['good', 'bad', 'more', 'hard', 'new', 'done'];
    const phaseIndex = phases.indexOf(this.state.phase);

    return phases.map((phase, i) => {
      let stateStr: 'done' | 'active' | 'future';
      if (i < phaseIndex) stateStr = 'done';
      else if (i === phaseIndex) stateStr = 'active';
      else stateStr = 'future';

      return {
        phase,
        label: this.phaseLabel(phase),
        shortLabel: this.phaseShortLabel(phase),
        stepNumber: i + 1,
        state: stateStr,
        detail: stateStr === 'active' ? this.phaseDetail(phase) : '',
        detailTitle: stateStr === 'active' ? this.phaseDetailTitle(phase) : '',
        light: stateStr === 'active' ? this.phaseLight(phase) : null,
        helpText: this.phaseHelpText(phase),
        intent: this.phaseIntent(phase, i + 1),
      };
    });
  }

  ngOnInit(): void {
    this.activate();
  }

  /**
   * Open the completion hand-off for a terminal autopilot phase.
   *
   * Both ways out are plain buttons and nothing happens on its own: the old
   * version auto-returned to the Dashboard on a countdown, which made *staying*
   * the outcome the user had to fight for — every re-entry to the Train window
   * re-armed it, so someone who thought the detector needed more work had to
   * cancel the same redirect over and over (#3201).
   *
   * Shown at most once per detector: ``shouldAnnounceCompletion`` requires that
   * this run started from an untrained detector, which is only ever true of the
   * run that actually did the training.
   */
  private announceCompletion(prompt: CompletionPrompt): void {
    this.autopilotState.markCompletionAnnounced();
    this.completionPrompt.set(prompt);
  }

  /** Dismiss the hand-off and stay in the Train window. */
  onStay(): void {
    this.completionPrompt.set(null);
  }

  /** Take the hand-off: close the modal and leave for the Dashboard. */
  onGoToDashboard(): void {
    this.completionPrompt.set(null);
    void this.router.navigate(['/dashboard']);
  }

  activate(): void {
    if (this.running) return;
    this.completionPrompt.set(null);
    // Retrain mode: the detector already has good+bad labels (carried over
    // from a previous dataset), so learned sort is available immediately and
    // autopilot should skip the initial text-mode phase.
    //
    // This is the *guess*, not the answer: on entry to the Train window the
    // counts below are still the zeroed defaults (the panel mounts before
    // `/api/votes` answers), and only the tab-switch path reaches here with
    // real ones. `noteInitialLabelset` corrects it from the run's first real
    // reading of the labelset — see #3535.
    const retrainMode = this.labelsetGoodCount() > 0 && this.labelsetBadCount() > 0;
    this.autopilotState.activate(retrainMode);
    // Immediately check whether existing votes already satisfy early phases
    // (e.g. user labeled 23 goods in Manual mode before switching to Autopilot).
    // The phase-transition effect only acts once `running` is true, so seed the
    // check here so the phase cascades (good→bad→hard) before we emit.
    this.autopilotState.checkPhaseTransition(
      this.goodVotes().size, this.badVotes().size, this.datasetSize(),
    );
    this.started.emit();
  }

  deactivate(): void {
    // Turning autopilot off answers the hand-off's question by itself.
    this.completionPrompt.set(null);
    this.autopilotState.deactivate();
    this.stopped.emit();
  }

  private phaseLabel(phase: AutopilotPhase): string {
    switch (phase) {
      case 'good': return 'Find Initial Goods.';
      case 'bad': return 'Find Initial Bads.';
      case 'more': return 'Find More Goods.';
      case 'hard': return 'Refine Boundary.';
      case 'new': return 'Explore Diversity.';
      case 'done': return 'Done!';
      default: return '';
    }
  }

  private phaseShortLabel(phase: AutopilotPhase): string {
    switch (phase) {
      case 'good': return 'Good';
      case 'bad': return 'Bad';
      case 'more': return 'More';
      case 'hard': return 'Boundary';
      case 'new': return 'Diversity';
      case 'done': return 'Done';
      default: return '';
    }
  }

  /**
   * The active step's one light (#4319): red, then yellow, then green, at
   * which point autopilot moves on to the next step.
   *
   * - The count steps (initial goods, initial bads, more goods) go yellow at
   *   half their target and green at the target. The "more" walk can also end
   *   early on a run of misses; the light keeps tracking goods, since a miss
   *   is not progress toward the target and a hit must never dim the light.
   * - The boundary step shows the lower of Smart and Stable: it ends when both
   *   are green, so the one further behind is the one holding it.
   * - The diversity step paces the coverage level against the Span
   *   indicator's own green target, split at half like the count steps. The
   *   level is a tricky metric (consecutive covered atlas nodes, so it stalls
   *   at a gap and then jumps), but the diversity sort always picks from the
   *   first uncovered node, so each vote in this step raises it by at least
   *   one: the plain halfway split already tracks votes, and a jump only ever
   *   lands closer to green. Green itself is the indicator's own call.
   */
  private phaseLight(phase: AutopilotPhase): StepLight {
    const st = this.state;
    const light = (color: LightColor, title: string): StepLight => ({
      color,
      ariaLabel: `Step progress: ${color}`,
      title: `${LIGHT_WORDS[color]} ${title}`,
    });
    switch (phase) {
      case 'good': {
        const target = this.effGoodTarget;
        return light(
          progressLight(this.goodVotes().size, target),
          countTitle(target, 'good'),
        );
      }
      case 'bad': {
        const target = this.effBadTarget;
        return light(
          progressLight(this.badVotes().size, target),
          countTitle(target, 'bad'),
        );
      }
      case 'more': {
        if (st.dryRunStop) {
          // On a document dataset the run of misses is the stop (#4488), so the
          // light tracks it, and a Good, which restarts the run, dims it.
          return light(
            progressLight(st.moreMisses, st.moreDryRun),
            `Counts the detector's best matches in a row that are not good: ${st.moreMisses} so far. `
            + `Red until ${st.moreDryRun / 2}, yellow past halfway, green at ${st.moreDryRun}, when the step ends. `
            + 'A good match starts the count again.',
          );
        }
        const target = this.effMoreTarget;
        return light(
          progressLight(this.goodVotes().size, target),
          `${countTitle(target, 'good')} The step can also end early, after ${st.moreDryRun} matches in a row that are not good.`,
        );
      }
      case 'hard': {
        const smart = indicatorLight(st.smartStatus);
        const stable = indicatorLight(st.stableStatus);
        const color = LIGHT_RANK[smart] <= LIGHT_RANK[stable] ? smart : stable;
        return light(
          color,
          `Shows the lower of two indicators; the step ends when both are green. `
          + `Smart (${smart}) tracks the detector's accuracy: green once it has settled and stopped improving. `
          + `Stable (${stable}) tracks whether the detector keeps changing its mind: green once its calls stop shifting between labeling steps.`,
        );
      }
      case 'new': {
        const target = st.spanTarget;
        const level = Math.round(st.fracDiversity);
        let color: LightColor;
        if (st.spanStatus === 'green') color = 'green';
        else if (target > 0) color = progressLight(level, target) === 'red' ? 'red' : 'yellow';
        else color = 'red';
        const coverage = target > 0
          ? `Your votes reach ${level} of the ${target} groups of your collection this step asks for.`
          : 'Waiting for the first coverage reading.';
        return light(
          color,
          `Tracks how much of your collection your votes cover. ${coverage} Yellow past halfway, green once they span a broad mix of items.`,
        );
      }
      case 'done':
        return light('green', st.dryRunStop ? dryRunNote(st.moreDryRun) : 'All quality indicators are green.');
      default:
        return light('red', '');
    }
  }

  private phaseHelpText(phase: AutopilotPhase): string {
    switch (phase) {
      case 'good': return 'Label a few examples of what you are looking for so the system can learn what "good" looks like.';
      case 'bad': return 'Label examples that are not what you want, helping the system learn the good/bad cutoff.';
      case 'more':
        return this.state.dryRunStop
          ? `Keep labeling the detector's best matches. Stops after ${this.state.moreDryRun} in a row that are not good.`
          : 'Keep labeling the best matches for your search. Stops once the matches stop turning up goods.';
      case 'hard': return 'The system shows you items near the good/bad cutoff. Labeling these improves accuracy where it matters most.';
      case 'new': return 'Explore a broad mix of items the system is less certain about, ensuring nothing important is missed.';
      case 'done':
        if (this.state.dryRunStop) return `${dryRunNote(this.state.moreDryRun)} You can continue labeling or export your results.`;
        return this.state.stablePlateau ? DONE_PLATEAU_HELP : DONE_HELP;
      default: return '';
    }
  }

  /**
   * Headline phase intent shown as a hover tooltip on the collapsed dots
   * (where the only visible affordance is a number/letter) and as a richer
   * tooltip on the expanded step label. Format: "Phase N: Short name.
   * What the user is doing and why."
   */
  private phaseIntent(phase: AutopilotPhase, stepNumber: number): string {
    switch (phase) {
      case 'good':
        return `Phase ${stepNumber}: Find initial goods. Label a few positives so the detector knows what "good" looks like.`;
      case 'bad':
        return `Phase ${stepNumber}: Find initial bads. Label a few negatives so the detector has both sides of the good/bad cutoff.`;
      case 'more':
        return this.state.dryRunStop
          ? `Phase ${stepNumber}: Find more goods. Keep labeling the detector's best matches until ${this.state.moreDryRun} in a row are not good; then the detector is trained.`
          : `Phase ${stepNumber}: Find more goods. Keep labeling the best matches for your search while they keep turning up goods; more goods early make a better detector later.`;
      case 'hard':
        return `Phase ${stepNumber}: Refine the cutoff. Votes on uncertain items train the detector fastest.`;
      case 'new':
        return `Phase ${stepNumber}: Cover a broad mix. Items from parts of your collection you haven't seen catch edge cases the cutoff phase missed.`;
      case 'done':
        if (this.state.dryRunStop) {
          return `Done. ${dryRunNote(this.state.moreDryRun)} Keep labeling for more, or export your results.`;
        }
        return this.state.stablePlateau
          ? `Done. ${PLATEAU_NOTE} Keep labeling if you want, but more votes are unlikely to change the result; or export your results.`
          : 'Done. All quality indicators are green. Keep labeling for more accuracy, or export your results.';
      default:
        return '';
    }
  }

  private phaseDetail(phase: AutopilotPhase): string {
    const st = this.state;
    switch (phase) {
      case 'good':
        return `${this.goodVotes().size}/${this.effGoodTarget} good labels`;
      case 'bad':
        return `${this.badVotes().size}/${this.effBadTarget} bad labels`;
      case 'more':
        return st.dryRunStop
          ? `${st.moreMisses}/${st.moreDryRun} in a row not good`
          : `${this.goodVotes().size}/${this.effMoreTarget} good labels`;
      case 'hard': {
        // No count target here — the phase ends when the smart and stable
        // indicators (the dots rendered right after this text) both go green.
        // That explanation lives in the tooltip (phaseDetailTitle); the visible
        // text stays a bare count so it never overflows the panel.
        const total = this.goodVotes().size + this.badVotes().size;
        return `${total} labels`;
      }
      case 'new':
        return st.spanTarget > 0
          ? `Diversity: ${Math.round(st.fracDiversity)}/${st.spanTarget}`
          : `Diversity: ${Math.round(st.fracDiversity)}`;
      case 'done':
        return st.dryRunStop ? 'Best matches ran dry' : 'All indicators green';
      default:
        return '';
    }
  }

  /**
   * Tooltip for the active step's detail text. Used to carry explanatory
   * copy that would overflow the panel if rendered inline — currently just
   * the "boundary" phase's end condition, which is otherwise invisible.
   */
  private phaseDetailTitle(phase: AutopilotPhase): string {
    switch (phase) {
      case 'more':
        return this.state.dryRunStop
          ? `Ends after ${this.state.moreDryRun} of the detector's best matches in a row are not good. ${this.goodVotes().size} good labels so far.`
          : `Ends at ${this.effMoreTarget} good labels, or after ${this.state.moreDryRun} matches in a row that are not good.`;
      case 'hard':
        return 'Ends when both indicators turn green.';
      case 'new':
        return 'Ends when the diversity indicator turns green.';
      default:
        return '';
    }
  }
}
