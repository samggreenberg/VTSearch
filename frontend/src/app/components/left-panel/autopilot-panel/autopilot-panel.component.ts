import { Component, Input, Output, EventEmitter, OnChanges, OnInit, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import type { LabelingStatusResponse } from '../../../generated/api-client/models/labeling-status-response';
import {
  AutopilotStateService,
  AutopilotPhase,
  AutopilotState,
} from '../../../services/autopilot-state.service';
import { ToastService } from '../../../services/toast.service';

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
  statusIcons: StatusIcon[];
  helpText: string;
  intent: string;
}

@Component({
  selector: 'vt-autopilot-panel',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './autopilot-panel.component.html',
  styleUrl: './autopilot-panel.component.scss',
})
export class AutopilotPanelComponent implements OnInit, OnChanges {
  @Input() goodVotes: Set<number> = new Set();
  @Input() badVotes: Set<number> = new Set();
  /**
   * Total "good" labels in the active detector's saved labelset, across all
   * datasets it has been used with.  When both this and ``labelsetBadCount``
   * are positive at activation time, autopilot enters retrain mode and uses
   * learned sort throughout instead of starting with text/example sort.
   */
  @Input() labelsetGoodCount = 0;
  @Input() labelsetBadCount = 0;
  @Input() labelingStatus: LabelingStatusResponse | null = null;
  @Input() collapsed = false;

  @Output() started = new EventEmitter<void>();
  @Output() stopped = new EventEmitter<void>();
  @Output() toggleCollapse = new EventEmitter<void>();
  @Output() refocus = new EventEmitter<void>();

  private completionAlerted = false;

  constructor(
    public autopilotState: AutopilotStateService,
    private toastService: ToastService,
  ) {}

  get state(): AutopilotState {
    return this.autopilotState.state;
  }

  get running(): boolean {
    return this.autopilotState.running;
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
        statusIcons: stateStr === 'active' ? this.phaseStatusIcons(phase) : [],
        helpText: this.phaseHelpText(phase),
        intent: this.phaseIntent(phase, i + 1),
      };
    });
  }

  ngOnInit(): void {
    this.activate();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (!this.running) return;

    if (changes['labelingStatus'] && this.labelingStatus) {
      this.autopilotState.updateFromLabelingStatus(this.labelingStatus);
    }

    if (changes['goodVotes'] || changes['badVotes'] || changes['labelingStatus']) {
      const prevPhase = this.autopilotState.state.phase;
      this.autopilotState.checkPhaseTransition(this.goodVotes.size, this.badVotes.size);
      if (prevPhase !== 'done' && this.autopilotState.state.phase === 'done' && !this.completionAlerted) {
        this.completionAlerted = true;
        this.toastService.success({
          message: 'Autopilot complete',
          detail: 'All quality indicators are green. You can continue labeling or export your results.',
        });
      }
    }
  }

  activate(): void {
    if (this.running) return;
    this.completionAlerted = false;
    // Retrain mode: the detector already has good+bad labels (carried over
    // from a previous dataset), so learned sort is available immediately and
    // autopilot should skip the initial text-mode phase.
    const retrainMode = this.labelsetGoodCount > 0 && this.labelsetBadCount > 0;
    this.autopilotState.activate(retrainMode);
    // Immediately check whether existing votes already satisfy early phases
    // (e.g. user labeled 23 goods in Manual mode before switching to Autopilot).
    // ngOnChanges ran before ngOnInit so `running` was false and the check was
    // skipped; do it now so the phase cascades (good→bad→hard) before we emit.
    this.autopilotState.checkPhaseTransition(this.goodVotes.size, this.badVotes.size);
    this.started.emit();
  }

  deactivate(): void {
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
    if (phase !== 'hard' && phase !== 'new') return [];
    const st = this.state;
    const smartState = st.smartStatus === 'green' ? 'green' : 'pending';
    const stableState = st.stableStatus === 'green' ? 'green' : 'pending';
    return [
      {
        color: st.smartStatus === 'green' ? 'green' : 'yellow',
        ariaLabel: `Smart: ${smartState}`,
        title: `Smart: ${smartState}. Tracks detector accuracy over labeling steps. Green when error cost has leveled off (detector has converged). Yellow when still improving.`,
      },
      {
        color: st.stableStatus === 'green' ? 'green' : 'yellow',
        ariaLabel: `Stable: ${stableState}`,
        title: `Stable: ${stableState}. Tracks prediction stability. Green when predictions stop changing between labeling steps (detector is confident). Yellow when predictions are still shifting.`,
      },
    ];
  }

  private phaseHelpText(phase: AutopilotPhase): string {
    switch (phase) {
      case 'good': return 'Label a few examples of what you are looking for so the system can learn what "good" looks like.';
      case 'bad': return 'Label examples that are not what you want, helping the system learn the boundary between good and bad.';
      case 'hard': return 'The system shows you items near the decision boundary. Labeling these improves accuracy where it matters most.';
      case 'new': return 'Explore diverse items the system is less certain about, ensuring nothing important is missed.';
      case 'done': return 'All quality indicators are green. You can continue labeling or export your results.';
      default: return '';
    }
  }

  /**
   * Headline phase intent shown as a hover tooltip on the collapsed dots
   * (where the only visible affordance is a number/letter) and as a richer
   * tooltip on the expanded step label. Format: "Phase N: Short name —
   * what the user is doing and why."
   */
  private phaseIntent(phase: AutopilotPhase, stepNumber: number): string {
    switch (phase) {
      case 'good':
        return `Phase ${stepNumber}: Find initial goods — label a few positives so the detector knows what "good" looks like.`;
      case 'bad':
        return `Phase ${stepNumber}: Find initial bads — label a few negatives so the detector has both sides of the boundary.`;
      case 'hard':
        return `Phase ${stepNumber}: Boundary refinement — votes on uncertain items train the model fastest.`;
      case 'new':
        return `Phase ${stepNumber}: Diversity exploration — items from unseen parts of the dataset catch edge cases the boundary phase missed.`;
      case 'done':
        return 'Done — all quality indicators are green. Keep labeling for more accuracy, or export your results.';
      default:
        return '';
    }
  }

  private phaseDetail(phase: AutopilotPhase): string {
    const st = this.state;
    switch (phase) {
      case 'good':
        return `${this.goodVotes.size}/${st.goodToStart} good labels`;
      case 'bad':
        return `${this.badVotes.size}/${st.badToStart} bad labels`;
      case 'hard': {
        const total = this.goodVotes.size + this.badVotes.size;
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
}
