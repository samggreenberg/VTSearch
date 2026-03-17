import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { LabelingStatusResponse } from '../models/api.models';

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

const INITIAL_STATE: AutopilotState = {
  phase: 'idle',
  goodToStart: 3,
  badToStart: 4,
  smartStatus: '',
  stableStatus: '',
  spanStatus: '',
  fracDiversity: 0,
};

@Injectable({ providedIn: 'root' })
export class AutopilotStateService {
  private readonly stateSubject = new BehaviorSubject<AutopilotState>({ ...INITIAL_STATE });

  readonly state$ = this.stateSubject.asObservable();

  get state(): AutopilotState {
    return this.stateSubject.value;
  }

  get running(): boolean {
    return this.stateSubject.value.phase !== 'idle';
  }

  activate(): void {
    if (this.running) return;
    this.stateSubject.next({
      ...this.stateSubject.value,
      phase: 'good',
      smartStatus: '',
      stableStatus: '',
      spanStatus: '',
      fracDiversity: 0,
    });
  }

  deactivate(): void {
    this.stateSubject.next({ ...this.stateSubject.value, phase: 'idle' });
  }

  updateFromLabelingStatus(status: LabelingStatusResponse): void {
    const current = this.stateSubject.value;
    this.stateSubject.next({
      ...current,
      smartStatus: (status.smart?.status as string) || '',
      stableStatus: (status.stable?.status as string) || '',
      spanStatus: (status.span?.status as string) || '',
      fracDiversity:
        status.span?.['diversity_level'] != null
          ? (status.span['diversity_level'] as number)
          : current.fracDiversity,
    });
  }

  checkPhaseTransition(goodCount: number, badCount: number): void {
    const st = this.stateSubject.value;
    let nextPhase = st.phase;

    if (st.phase === 'good' && goodCount >= st.goodToStart) {
      nextPhase = 'bad';
    } else if (st.phase === 'bad' && badCount >= st.badToStart) {
      nextPhase = 'hard';
    } else if (st.phase === 'hard' && st.smartStatus === 'green' && st.stableStatus === 'green') {
      nextPhase = 'new';
    } else if (st.phase === 'new' && (st.smartStatus !== 'green' || st.stableStatus !== 'green')) {
      nextPhase = 'hard';
    } else if (st.phase === 'new' && st.spanStatus === 'green') {
      nextPhase = 'done';
    }

    if (nextPhase !== st.phase) {
      this.stateSubject.next({ ...st, phase: nextPhase });
    }
  }

  clear(): void {
    this.stateSubject.next({ ...INITIAL_STATE });
  }
}
