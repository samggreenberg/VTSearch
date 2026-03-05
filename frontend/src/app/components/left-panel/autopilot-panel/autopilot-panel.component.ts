import { Component, Input, Output, EventEmitter, OnChanges, OnInit, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { LabelingStatusResponse } from '../../../models/api.models';

export type AutopilotPhase = 'idle' | 'good' | 'bad' | 'hard' | 'new' | 'done';

export interface AutopilotState {
  phase: AutopilotPhase;
  goodToStart: number;
  badToStart: number;
  smartStatus: string;
  stableStatus: string;
  spanStatus: string;
  fracDiversity: number;
}

interface StepDisplay {
  phase: AutopilotPhase;
  label: string;
  state: 'done' | 'active' | 'future';
  detail: string;
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

  state: AutopilotState = {
    phase: 'idle',
    goodToStart: 3,
    badToStart: 4,
    smartStatus: '',
    stableStatus: '',
    spanStatus: '',
    fracDiversity: 0,
  };

  get running(): boolean {
    return this.state.phase !== 'idle';
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
      };
    });
  }

  ngOnInit(): void {
    this.activate();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (!this.running) return;

    if (changes['labelingStatus'] && this.labelingStatus) {
      this.state.smartStatus = (this.labelingStatus.smart?.status as string) || '';
      this.state.stableStatus = (this.labelingStatus.stable?.status as string) || '';
      this.state.spanStatus = (this.labelingStatus.span?.status as string) || '';
      if (this.labelingStatus.span?.['diversity_level'] != null) {
        this.state.fracDiversity = this.labelingStatus.span['diversity_level'] as number;
      }
    }

    if (changes['goodVotes'] || changes['badVotes'] || changes['labelingStatus']) {
      this.checkPhaseTransition();
    }
  }

  activate(): void {
    if (this.running) return;
    this.state = {
      ...this.state,
      phase: 'good',
      smartStatus: '',
      stableStatus: '',
      spanStatus: '',
      fracDiversity: 0,
    };
    this.started.emit();
  }

  deactivate(): void {
    this.state = { ...this.state, phase: 'idle' };
    this.stopped.emit();
  }

  private checkPhaseTransition(): void {
    const st = this.state;

    if (st.phase === 'good' && this.goodVotes.size >= st.goodToStart) {
      this.state = { ...st, phase: 'bad' };
    } else if (st.phase === 'bad' && this.badVotes.size >= st.badToStart) {
      this.state = { ...st, phase: 'hard' };
    } else if (st.phase === 'hard' && st.smartStatus === 'green' && st.stableStatus === 'green') {
      this.state = { ...st, phase: 'new' };
    } else if (st.phase === 'new' && st.spanStatus === 'green') {
      this.state = { ...st, phase: 'done' };
    }
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

  private phaseDetail(phase: AutopilotPhase): string {
    const st = this.state;
    switch (phase) {
      case 'good':
        return `${this.goodVotes.size}/${st.goodToStart} good labels`;
      case 'bad':
        return `${this.badVotes.size}/${st.badToStart} bad labels`;
      case 'hard': {
        const total = this.goodVotes.size + this.badVotes.size;
        const smart = st.smartStatus === 'green' ? 'green' : 'pending';
        const stable = st.stableStatus === 'green' ? 'green' : 'pending';
        return `${total} labels | Smart: ${smart}, Stable: ${stable}`;
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
