import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import type { LabelingStatusResponse } from '../generated/api-client/models/labeling-status-response';

export type AutopilotPhase = 'idle' | 'good' | 'bad' | 'more' | 'hard' | 'new' | 'done' | 'exhausted';

export interface AutopilotState {
  phase: AutopilotPhase;
  goodToStart: number;
  badToStart: number;
  /**
   * The "more" walk (#4282): after the Good and Bad quorum, go back to the top
   * of the seed sort until the labelset holds ``moreToStart`` positives, or
   * until ``moreDryRun`` walk picks in a row held none. #4222's study measured
   * it against the old two-round opening on COCO Better: +0.05 AP by vote 150,
   * with the same share of sessions left without a detector.
   */
  moreToStart: number;
  moreDryRun: number;
  /** Walk picks in a row without a positive, toward ``moreDryRun``. */
  moreMisses: number;
  /** The walk has ended (target met, ran dry, or skipped) and never resumes. */
  moreDone: boolean;
  /**
   * The dataset's labeling stops on the dry run, not the lights (#4488): a
   * document (tiled structural) dataset, whose labeling status says
   * ``stop_rule: 'dry_run'``. There the walk draws off the detector's own
   * ranking, has no Good target, and its run of ``moreDryRun`` misses is the
   * stop: Good, Bad, More, then Done, with no Boundary or Diversity step.
   */
  dryRunStop: boolean;
  smartStatus: string;
  stableStatus: string;
  /**
   * True when Stable is green only because the flips that remain are
   * boundary wobble that has stopped falling: the detector has stopped
   * improving but a fringe of the pool sits in an ambiguity the embedding
   * cannot resolve (#3831). The Done step says so instead of implying the
   * pool converged.
   */
  stablePlateau: boolean;
  spanStatus: string;
  fracDiversity: number;
  /**
   * The coverage level that turns the Span indicator green (the span payload's
   * ``target``: the diversity goal, capped at the atlas's node count). The
   * Diversity step paces its light against it (#4319); ``0`` until the first
   * labeling status reports one.
   */
  spanTarget: number;
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
  moreToStart: 20,
  moreDryRun: 16,
  moreMisses: 0,
  moreDone: false,
  dryRunStop: false,
  smartStatus: '',
  stableStatus: '',
  stablePlateau: false,
  spanStatus: '',
  fracDiversity: 0,
  spanTarget: 0,
  retrainMode: false,
};

@Injectable({ providedIn: 'root' })
export class AutopilotStateService {
  private readonly stateSubject = new BehaviorSubject<AutopilotState>({ ...INITIAL_STATE });

  readonly state$ = this.stateSubject.asObservable();

  /**
   * Whether the terminal-phase hand-off (the "trained" modal) has already been
   * shown for the current autopilot run.
   *
   * This lives on the service rather than on the panel component on purpose:
   * the panel is destroyed and rebuilt every time the user switches the
   * left-panel tab, so a component-scoped flag would re-open the modal on each
   * return.
   */
  private completionAnnounced = false;

  /**
   * Whether the detector already carried labels when this autopilot run began.
   *
   * The completion hand-off is for the moment a user *finishes training a
   * detector*, which happens exactly once per detector. Every later run —
   * continuing after autopilot already finished, or picking up a detector
   * trained on another dataset — starts from a detector that is already
   * trained, and announcing "Done!" there is the nag reported in #3201: the
   * user re-enters the Train window on purpose and is immediately told to
   * leave it again.
   *
   * ``completionAnnounced`` cannot carry that on its own, because it is
   * per-run and every re-entry to the Train window is a new run. The labelset
   * is what remembers across runs, so the run's first look at it decides.
   */
  private startedTrained = false;

  /** Whether {@link noteInitialLabelset} has taken its one reading this run. */
  private initialLabelsetKnown = false;

  /**
   * The vote counts at the last {@link checkPhaseTransition}. The phase check
   * sees only counts, so the "more" walk reads each vote's outcome off which
   * count rose (``vtscore/eval/autopilot_flow.py`` reads it the same way).
   */
  private lastCounts: { good: number; bad: number } | null = null;

  get state(): AutopilotState {
    return this.stateSubject.value;
  }

  get running(): boolean {
    return this.stateSubject.value.phase !== 'idle';
  }

  /**
   * Record the run's first real reading of the detector's labelset.
   *
   * Takes effect on the first call after {@link activate} and ignores every
   * later one: once the user starts voting the labelset is no longer evidence
   * of anything, so only the first reading — taken as soon as ``/api/votes``
   * has landed, before the user can have voted — is meaningful.
   *
   * Two things read it. {@link startedTrained} asks whether the detector held
   * *any* labels, and gates the completion hand-off. ``retrainMode`` asks the
   * stricter question — both classes, so learned sort is available from the
   * first moment — and it is corrected here rather than left at whatever
   * {@link activate} guessed, because on entry to the Train window that guess
   * is always ``false``: the panel mounts on the first render, two round trips
   * before ``/api/votes`` answers, so a fully trained detector activated as
   * untrained and spent its first sort on the text hint (#3535).
   */
  noteInitialLabelset(goodCount: number, badCount: number): void {
    if (this.initialLabelsetKnown) return;
    this.initialLabelsetKnown = true;
    this.startedTrained = goodCount > 0 || badCount > 0;
    const retrainMode = goodCount > 0 && badCount > 0;
    if (retrainMode !== this.stateSubject.value.retrainMode) {
      this.stateSubject.next({
        ...this.stateSubject.value,
        retrainMode,
        moreDone: this.stateSubject.value.moreDone || retrainMode,
      });
    }
  }

  /**
   * Discard the reading, so the next {@link noteInitialLabelset} takes a fresh
   * one. Called while the labelset is un-loaded — the dataset/detector pair
   * changed under a running autopilot — because the reading described the
   * *previous* detector and says nothing about the one now in front of the user.
   */
  forgetInitialLabelset(): void {
    this.initialLabelsetKnown = false;
    this.startedTrained = false;
  }

  /**
   * Whether reaching a terminal phase right now should be announced to the
   * user: this run has to have started from an untrained detector (see
   * {@link startedTrained}), the labelset must have been read at least once so
   * we are not guessing from not-yet-loaded zeroes, and the hand-off must not
   * already have been offered.
   */
  get shouldAnnounceCompletion(): boolean {
    return this.initialLabelsetKnown && !this.startedTrained && !this.completionAnnounced;
  }

  /** Record that the completion hand-off has been offered for this run. */
  markCompletionAnnounced(): void {
    this.completionAnnounced = true;
  }

  activate(retrainMode = false): void {
    if (this.running) return;
    this.completionAnnounced = false;
    this.startedTrained = false;
    this.initialLabelsetKnown = false;
    this.lastCounts = null;
    this.stateSubject.next({
      ...this.stateSubject.value,
      phase: 'good',
      moreMisses: 0,
      // A detector that already had labels starts in retrain mode, where every
      // phase draws off the learned sort: there is no seed sort to walk.
      moreDone: retrainMode,
      smartStatus: '',
      stableStatus: '',
      stablePlateau: false,
      spanStatus: '',
      fracDiversity: 0,
      spanTarget: 0,
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
      dryRunStop: status.stop_rule === 'dry_run',
      smartStatus: status.smart.status || '',
      stableStatus: status.stable.status || '',
      stablePlateau: status.stable['plateau'] === true,
      spanStatus: status.span.status || '',
      fracDiversity:
        status.span['diversity_level'] != null
          ? (status.span['diversity_level'] as number)
          : current.fracDiversity,
      spanTarget:
        status.span['target'] != null ? (status.span['target'] as number) : current.spanTarget,
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
    let st = this.stateSubject.value;
    if (st.phase === 'idle') return;

    // The "more" walk's run of misses. Only a vote cast *in* the walk counts,
    // and its outcome is read off the counts: a rise in Goods is a hit, a rise
    // in Bads alone is a miss. Repeated checks with unchanged counts are no-ops.
    const prev = this.lastCounts;
    this.lastCounts = { good: goodCount, bad: badCount };
    if (prev && st.phase === 'more' && (st.dryRunStop || !st.moreDone)) {
      if (goodCount > prev.good) {
        st = { ...st, moreMisses: 0 };
      } else if (badCount > prev.bad) {
        const moreMisses = st.moreMisses + 1;
        st = { ...st, moreMisses, moreDone: moreMisses >= st.moreDryRun };
      }
    }

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
    const effMoreTarget = Math.min(st.moreToStart, goodCount + remainingUnlabeled);

    // Derive the correct phase from current counts and indicator statuses.
    // This allows both forward and backward transitions (e.g. if votes are
    // cleared or un-toggled, the phase regresses to match).
    let nextPhase: AutopilotPhase;
    if (goodCount < effGoodTarget) {
      nextPhase = 'good';
    } else if (badCount < effBadTarget) {
      nextPhase = 'bad';
    } else if (st.dryRunStop) {
      // A document dataset (#4488): the walk is the rest of the run, with no
      // Good target, and running dry is Done. It walks in retrain mode too,
      // where the latched ``moreDone`` would otherwise skip it: the run of
      // misses is the stop, so it is the only history that counts here.
      if (st.moreMisses >= st.moreDryRun) nextPhase = 'done';
      else if (remainingUnlabeled === 0) nextPhase = 'exhausted';
      else nextPhase = 'more';
    } else if (!st.moreDone && goodCount < effMoreTarget) {
      nextPhase = 'more';
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

    // Once the machine has moved past the walk it is spent, as a schedule
    // round is in the harness: un-voting a positive later does not resume it.
    const moreDone = st.moreDone || !['good', 'bad', 'more'].includes(nextPhase);
    if (nextPhase !== st.phase || st !== this.stateSubject.value || moreDone !== st.moreDone) {
      this.stateSubject.next({ ...st, phase: nextPhase, moreDone });
    }
  }

  clear(): void {
    this.completionAnnounced = false;
    this.startedTrained = false;
    this.initialLabelsetKnown = false;
    this.lastCounts = null;
    this.stateSubject.next({ ...INITIAL_STATE });
  }
}
