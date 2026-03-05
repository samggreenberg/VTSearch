import { Component, Input, Output, EventEmitter, OnChanges, OnInit, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { LabelingStatusResponse } from '../../../models/api.models';
import {
  AutopilotStateService,
  AutopilotPhase,
  AutopilotState,
} from '../../../services/autopilot-state.service';

export type { AutopilotPhase, AutopilotState };

interface StatusIcon {
  color: 'green' | 'yellow';
  ariaLabel: string;
}

interface StepDisplay {
  phase: AutopilotPhase;
  label: string;
  state: 'done' | 'active' | 'future';
  detail: string;
  statusIcons: StatusIcon[];
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
  @Input() labelingStatus: LabelingStatusResponse | null = null;

  @Output() started = new EventEmitter<void>();
  @Output() stopped = new EventEmitter<void>();

  constructor(public autopilotState: AutopilotStateService) {}

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
        state: stateStr,
        detail: stateStr === 'active' ? this.phaseDetail(phase) : '',
        statusIcons: stateStr === 'active' ? this.phaseStatusIcons(phase) : [],
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
      this.autopilotState.checkPhaseTransition(this.goodVotes.size, this.badVotes.size);
    }
  }

  activate(): void {
    if (this.running) return;
    this.autopilotState.activate();
    this.started.emit();
  }

  deactivate(): void {
    this.autopilotState.deactivate();
    this.stopped.emit();
  }

  private phaseLabel(phase: AutopilotPhase): string {
    switch (phase) {
      case 'good': return 'Label Good Examples';
      case 'bad': return 'Label Bad Examples';
      case 'hard': return 'Refine Boundary';
      case 'new': return 'Explore Diversity';
      case 'done': return 'Done';
      default: return '';
    }
  }

  private phaseStatusIcons(phase: AutopilotPhase): StatusIcon[] {
    if (phase !== 'hard') return [];
    const st = this.state;
    return [
      { color: st.smartStatus === 'green' ? 'green' : 'yellow', ariaLabel: `Smart: ${st.smartStatus === 'green' ? 'green' : 'pending'}` },
      { color: st.stableStatus === 'green' ? 'green' : 'yellow', ariaLabel: `Stable: ${st.stableStatus === 'green' ? 'green' : 'pending'}` },
    ];
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
        return `Diversity: ${st.fracDiversity.toFixed(1)}`;
      case 'done':
        return 'All indicators green';
      default:
        return '';
    }
  }
}
