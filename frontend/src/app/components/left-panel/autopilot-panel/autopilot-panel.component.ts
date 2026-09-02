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

interface StatusIcon {
  color: 'green' | 'yellow';
  ariaLabel: string;
  title: string;
}

export interface StepDisplay {
  phase: AutopilotPhase;
  label: string;
  shortLabel: string;
  stepNumber: number;
  state: 'done' | 'active' | 'future';
  detail: string;
  detailTitle: string;
  statusIcons: StatusIcon[];
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
            detail:
              'Every quality indicator is green: the detector\'s accuracy has settled, its calls '
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

  get steps(): StepDisplay[] {
    const phases: AutopilotPhase[] = ['good', 'bad', 'hard', 'new', 'done'];
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
        statusIcons: stateStr === 'active' ? this.phaseStatusIcons(phase) : [],
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
      case 'hard': return 'Boundary';
      case 'new': return 'Diversity';
      case 'done': return 'Done';
      default: return '';
    }
  }

  private phaseStatusIcons(phase: AutopilotPhase): StatusIcon[] {
    const st = this.state;
    // The boundary phase is gated by the smart + stable indicators, so it
    // shows both dots. The diversity phase runs *after* smart + stable are
    // already green (that is the condition for leaving the boundary phase),
    // so showing those two dots here would just be two stale greens. The
    // indicator that actually gates the diversity phase is span coverage, so
    // the diversity row shows a single span dot instead.
    if (phase === 'hard') {
      const smartState = st.smartStatus === 'green' ? 'green' : 'pending';
      const stableState = st.stableStatus === 'green' ? 'green' : 'pending';
      return [
        {
          color: st.smartStatus === 'green' ? 'green' : 'yellow',
          ariaLabel: `Smart: ${smartState}`,
          title: `Smart: ${smartState}. Tracks the detector's accuracy as you label. Green when its accuracy has settled and stopped improving. Yellow when it's still getting better.`,
        },
        {
          color: st.stableStatus === 'green' ? 'green' : 'yellow',
          ariaLabel: `Stable: ${stableState}`,
          title: `Stable: ${stableState}. Tracks whether the detector keeps changing its mind. Green when it has stopped changing its calls between labeling steps. Yellow when its calls are still shifting.`,
        },
      ];
    }
    if (phase === 'new') {
      const spanState = st.spanStatus === 'green' ? 'green' : 'pending';
      return [
        {
          color: st.spanStatus === 'green' ? 'green' : 'yellow',
          ariaLabel: `Diversity: ${spanState}`,
          title: `Diverse: ${spanState}. Tracks how much of your collection your votes cover. Green when your votes span a broad mix of items. Yellow when they're still bunched together.`,
        },
      ];
    }
    return [];
  }

  private phaseHelpText(phase: AutopilotPhase): string {
    switch (phase) {
      case 'good': return 'Label a few examples of what you are looking for so the system can learn what "good" looks like.';
      case 'bad': return 'Label examples that are not what you want, helping the system learn the good/bad cutoff.';
      case 'hard': return 'The system shows you items near the good/bad cutoff. Labeling these improves accuracy where it matters most.';
      case 'new': return 'Explore a broad mix of items the system is less certain about, ensuring nothing important is missed.';
      case 'done': return 'All quality indicators are green. You can continue labeling or export your results.';
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
      case 'hard':
        return `Phase ${stepNumber}: Refine the cutoff. Votes on uncertain items train the detector fastest.`;
      case 'new':
        return `Phase ${stepNumber}: Cover a broad mix. Items from parts of your collection you haven't seen catch edge cases the cutoff phase missed.`;
      case 'done':
        return 'Done. All quality indicators are green. Keep labeling for more accuracy, or export your results.';
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
      case 'hard': {
        // No count target here — the phase ends when the smart and stable
        // indicators (the dots rendered right after this text) both go green.
        // That explanation lives in the tooltip (phaseDetailTitle); the visible
        // text stays a bare count so it never overflows the panel.
        const total = this.goodVotes().size + this.badVotes().size;
        return `${total} labels`;
      }
      case 'new':
        return `Diversity: ${Math.round(st.fracDiversity)}`;
      case 'done':
        return 'All indicators green';
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
      case 'hard':
        return 'Ends when both indicators turn green.';
      case 'new':
        return 'Ends when the diversity indicator turns green.';
      default:
        return '';
    }
  }
}
