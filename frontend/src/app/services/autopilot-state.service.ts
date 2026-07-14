import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import type { LabelingStatusResponse } from '../generated/api-client/models/labeling-status-response';

export type AutopilotPhase = 'idle' | 'good' | 'bad' | 'hard' | 'new' | 'done' | 'exhausted';

export interface AutopilotState {
  phase: AutopilotPhase;
  goodToStart: number;
  badToStart: number;
  smartStatus: string;
  stableStatus: string;
  spanStatus: string;
  fracDiversity: number;
  /**
   * True when autopilot started against a detector that already has labels
   * (e.g. trained on DatasetA, now continuing on DatasetB).  In retrain mode
   * every phase uses learned sort; the initial "good"/"bad" phases use
   * Learned-Good and Learned-Hard against the existing model instead of
   * falling back to text/example sort.
   */
  retrainMode: boolean;
}

const INITIAL_STATE: AutopilotState = {
  phase: 'idle',
  goodToStart: 3,
  badToStart: 4,
  smartStatus: '',
  stableStatus: '',
  spanStatus: '',
  fracDiversity: 0,
  retrainMode: false,
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

  activate(retrainMode = false): void {
    if (this.running) return;
    this.stateSubject.next({
      ...this.stateSubject.value,
      phase: 'good',
      smartStatus: '',
      stableStatus: '',
      spanStatus: '',
      fracDiversity: 0,
      retrainMode,
    });
  }

  deactivate(): void {
    this.stateSubject.next({ ...this.stateSubject.value, phase: 'idle' });
  }

  updateFromLabelingStatus(status: LabelingStatusResponse): void {
    const current = this.stateSubject.value;
    this.stateSubject.next({
      ...current,
      smartStatus: status.smart.status || '',
      stableStatus: status.stable.status || '',
      spanStatus: status.span.status || '',
      fracDiversity:
        status.span['diversity_level'] != null
          ? (status.span['diversity_level'] as number)
          : current.fracDiversity,
    });
  }

  updateDiversityLevel(level: number): void {
    const current = this.stateSubject.value;
    if (current.fracDiversity === level) return;
    this.stateSubject.next({ ...current, fracDiversity: level });
  }

  /**
   * Recompute the phase from current vote counts and indicator statuses.
   *
   * ``totalCount`` is the number of items in the active dataset. When it is
   * known (finite and positive) the phase targets are capped to what the
   * dataset can actually supply: on a 1-item (or otherwise tiny) collection
   * the default 3-good / 4-bad targets are unreachable, so gating purely on
   * ``count < target`` would strand autopilot in an early phase forever. Pass
   * ``0`` (the default) when the size is unknown to keep the targets uncapped.
   */
  checkPhaseTransition(goodCount: number, badCount: number, totalCount = 0): void {
    const st = this.stateSubject.value;
    if (st.phase === 'idle') return;

    // How many items still carry no vote. Treat the size as "unknown" — and so
    // leave targets uncapped and never exhaust — unless it is a finite positive
    // number that is at least the current vote count. A total below the vote
    // count means the number is stale/inconsistent (votes loaded before medias,
    // or a labelset spanning several datasets), which we must not mistake for a
    // fully-labeled tiny dataset.
    const rawRemaining = totalCount - goodCount - badCount;
    const sizeKnown = Number.isFinite(totalCount) && totalCount > 0 && rawRemaining >= 0;
    const remainingUnlabeled = sizeKnown ? rawRemaining : Infinity;

    // Cap each phase target at the most votes of that class the dataset could
    // still yield (current votes of that class + everything unlabeled), so a
    // tiny dataset can still satisfy — and advance past — the initial phases.
    const effGoodTarget = Math.min(st.goodToStart, goodCount + remainingUnlabeled);
    const effBadTarget = Math.min(st.badToStart, badCount + remainingUnlabeled);

    // Derive the correct phase from current counts and indicator statuses.
    // This allows both forward and backward transitions (e.g. if votes are
    // cleared or un-toggled, the phase regresses to match).
    let nextPhase: AutopilotPhase;
    if (goodCount < effGoodTarget) {
      nextPhase = 'good';
    } else if (badCount < effBadTarget) {
      nextPhase = 'bad';
    } else if (st.smartStatus === 'green' && st.stableStatus === 'green' && st.spanStatus === 'green') {
      nextPhase = 'done';
    } else if (remainingUnlabeled === 0) {
      // Every item is labeled but the quality indicators never went green
      // (typical of a tiny dataset that can't reach the good+bad quorum).
      // Land in a terminal "exhausted" state so the view can render a clear
      // message instead of a blank pane stuck in 'hard' with nothing to select.
      nextPhase = 'exhausted';
    } else if (st.smartStatus === 'green' && st.stableStatus === 'green') {
      nextPhase = 'new';
    } else {
      nextPhase = 'hard';
    }

    if (nextPhase !== st.phase) {
      this.stateSubject.next({ ...st, phase: nextPhase });
    }
  }

  clear(): void {
    this.stateSubject.next({ ...INITIAL_STATE });
  }
}
