import {
  ChangeDetectionStrategy,
  Component,
  effect,
  inject,
  input,
  OnDestroy,
  OnInit,
  output,
} from '@angular/core';
import { Router } from '@angular/router';

import type { LabelingStatusResponse } from '../../../generated/api-client/models/labeling-status-response';
import {
  AutopilotStateService,
  AutopilotPhase,
  AutopilotState,
} from '../../../services/autopilot-state.service';
import { ToastService } from '../../../services/toast.service';
import { IconComponent } from '../../icon/icon.component';

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

/**
 * Seconds the completion toast counts down before returning the user to the
 * Dashboard. Long enough to read the headline and reach the "Stay here"
 * button, short enough that a user who is done doesn't sit waiting.
 */
const RETURN_COUNTDOWN_SECONDS = 5;

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'vt-autopilot-panel',
  standalone: true,
  imports: [IconComponent],
  templateUrl: './autopilot-panel.component.html',
  styleUrl: './autopilot-panel.component.scss',
})
export class AutopilotPanelComponent implements OnInit, OnDestroy {
  autopilotState = inject(AutopilotStateService);
  private toastService = inject(ToastService);
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

  readonly started = output<void>();
  readonly stopped = output<void>();
  readonly toggleCollapse = output<void>();
  readonly refocus = output<void>();

  private completionAlerted = false;
  /** Id of the live completion toast, or ``null`` when none is showing. Kept
   *  so leaving the Train window cancels the pending auto-return. */
  private completionToastId: number | null = null;

  constructor() {
    // Signal inputs don't fire ``ngOnChanges``; this effect replaces the old
    // change hook. It re-runs whenever the vote sets, dataset size, or labeling
    // status change, driving the same phase-transition + completion-toast logic.
    effect(() => {
      // Read every reactive input so the effect tracks them as dependencies.
      const goodVotes = this.goodVotes();
      const badVotes = this.badVotes();
      const datasetSize = this.datasetSize();
      const labelingStatus = this.labelingStatus();

      if (!this.running) return;

      if (labelingStatus) {
        this.autopilotState.updateFromLabelingStatus(labelingStatus);
      }

      const prevPhase = this.autopilotState.state.phase;
      this.autopilotState.checkPhaseTransition(goodVotes.size, badVotes.size, datasetSize);
      const phase = this.autopilotState.state.phase;
      if (prevPhase !== phase && !this.completionAlerted) {
        if (phase === 'done') {
          this.announceCompletion('Done! Your detector is trained.');
        } else if (phase === 'exhausted') {
          this.announceCompletion(
            'Done! Nothing left to label.',
            'Autopilot has labeled every item in this dataset.',
          );
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

  ngOnDestroy(): void {
    // Leaving the Train window (tab switch, autopilot stopped, view change)
    // calls off any pending auto-return: dismissing the toast cancels its
    // countdown without running the navigation.
    this.cancelCompletionToast();
  }

  /**
   * Announce a terminal autopilot phase and start the auto-return countdown.
   *
   * A bare "Done!" leaves the user parked in the Train window with no next
   * step, so the toast says where they are being taken and when — and carries
   * a "Stay here" button for the user who wants to keep labeling instead.
   * Dismissing the toast by any means cancels the return.
   */
  private announceCompletion(message: string, detail?: string): void {
    this.completionAlerted = true;
    this.completionToastId = this.toastService.success({
      message,
      detail,
      countdown: {
        label: 'Taking you back to the Dashboard in',
        seconds: RETURN_COUNTDOWN_SECONDS,
        onExpire: () => this.returnToDashboard(),
      },
      action: {
        label: 'Stay here',
        title: 'Stay in the Train window and keep labeling',
        onClick: () => {
          this.completionToastId = null;
        },
      },
    });
  }

  private returnToDashboard(): void {
    // Cleared first so the ``ngOnDestroy`` that the navigation triggers does
    // not try to dismiss an already-dismissed toast.
    this.completionToastId = null;
    void this.router.navigate(['/dashboard']);
  }

  private cancelCompletionToast(): void {
    if (this.completionToastId === null) return;
    this.toastService.dismiss(this.completionToastId);
    this.completionToastId = null;
  }

  activate(): void {
    if (this.running) return;
    this.completionAlerted = false;
    this.cancelCompletionToast();
    // Retrain mode: the detector already has good+bad labels (carried over
    // from a previous dataset), so learned sort is available immediately and
    // autopilot should skip the initial text-mode phase.
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
    // Turning autopilot off is itself a "stay here" — drop any pending return.
    this.cancelCompletionToast();
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
