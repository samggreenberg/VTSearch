import { Injectable, computed, signal } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import type { LabelingStatusResponse } from '../generated/api-client/models/labeling-status-response';

export type AutopilotPhase = 'idle' | 'good' | 'bad' | 'more' | 'hard' | 'new' | 'spot' | 'done' | 'exhausted';

/**
 * Why a spot check is running inside a run: `weak` when the labels separate
 * weakly (#4496), `done` for the check Autopilot runs once every light is green,
 * before it reports Done.
 */
export type SpotCheckKind = 'weak' | 'done';

export interface AutopilotState {
  phase: AutopilotPhase;
  goodToStart: number;
  badToStart: number;
  /**
   * The "more" walk (#4282): after the Good and Bad quorum, go back to the top
   * of the seed sort until the labelset holds ``moreToStart`` positives, or
   * until ``moreDryRun`` walk picks in a row held none. Off on photos since
   * #4740 (``moreToStart`` 0, so Bad goes straight to Boundary): #4671 scored it
   * against the opening without it and the walk lost from vote 23 on, at every
   * preset. A document dataset's walk (``dryRunStop``) does not read it.
   */
  moreToStart: number;
  moreDryRun: number;
  /** Walk picks in a row without a positive, toward ``moreDryRun``. */
  moreMisses: number;
  /** The walk has ended (target met, ran dry, or skipped) and never resumes. */
  moreDone: boolean;
  /**
   * Good-phase picks in a row without a positive, counted once a Good is in
   * hand, toward ``moreDryRun`` (#4731).
   */
  goodMisses: number;
  /**
   * The Good walk ran dry (#4731): ``moreDryRun`` picks in a row held no
   * Good, with a Good in hand. The Goods found so far then meet the Good
   * target, and the More walk down the same sort is spent. A target with one
   * or two Goods the sort can reach used to walk it for the whole session.
   * Latched for the run, like ``moreDone``.
   */
  goodRanDry: boolean;
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
   * The labels line's own expected F-beta at the detector's balance has reached
   * Done's level (#4878: the labeling status's ``level.met``). Done comes once
   * every light is green or this holds: priced on 2,160 sessions and A/B'd, it
   * ends a session a few clicks sooner for the same F-beta.
   */
  levelMet: boolean;
  /** That expected F-beta, for the panel to quote when the level is what said Done; null without a line. */
  expectedFbeta: number | null;
  /**
   * The detector holds {@link STOP_CAP_VOTES} votes (#4903): Done comes here
   * whatever the lights and the level say, so no session goes unstopped.
   */
  atCap: boolean;
  /**
   * True when autopilot started against a detector that already has labels
   * (e.g. trained on DatasetA, now continuing on DatasetB).  In retrain mode
   * every phase uses learned sort; the initial "good"/"bad" phases use
   * Learned-Good and Learned-Hard against the existing model instead of
   * falling back to text/example sort.
   */
  retrainMode: boolean;
  /**
   * This run has reached Done (#4621). The phase keeps following the
   * indicators after that, because it decides what autopilot picks: a vote
   * that knocks Stable off green sends it back to ``hard`` for boundary items.
   * But the user was told the detector is trained, so the panel reads this
   * instead of the phase, keeping Done checked and showing further votes as
   * an open-ended last step (Keep Improving) rather than bouncing back to
   * Boundary. Display
   * only: it never feeds the phase. Cleared only by falling back into the
   * opening (votes un-done below its targets) or by a new run.
   */
  doneReached: boolean;
  /**
   * The spot check running inside this run, or null. While one runs the phase
   * is `spot`: the check deals the picks, not the phase's sort, and the view
   * shows them in the centre panel as the next items. Only Autopilot starts a
   * check.
   */
  spotCheck: SpotCheckKind | null;
  /**
   * This run's check before Done has closed: it ran (finished, or stopped by
   * the user), or was skipped because a check had just ended or the line has
   * none to run. Latched for the run, like `moreDone`: Done follows from here
   * on, and a later return of every light to green does not check again.
   */
  doneChecked: boolean;
}

/**
 * Toasty's hand-off when a run that trained the detector ends (#4680): `done`
 * when the run reaches Done, whatever said so (#5143: the hint says what to do
 * next, not why), `all-labeled` when nothing is left to label, Done or not
 * (#5151: "keep clicking" with nothing to click is wrong). `votes` is the
 * vote total it was raised at; any vote or undo after that drops it.
 */
export interface AutopilotHandoff {
  kind: 'done' | 'all-labeled';
  votes: number;
}

/** Every quality light green: what said Done before the level could (#4878). */
export function allLightsGreen(st: AutopilotState): boolean {
  return st.smartStatus === 'green' && st.stableStatus === 'green' && st.spanStatus === 'green';
}

/**
 * The click cap (#4903): Done once the detector holds this many votes, whatever
 * the lights and the level say. Every stop rule leaves some sessions unstopped,
 * and a user Autopilot never tells to stop either quits when they like or labels
 * on through the collection. The harness mirrors it as `STOP_CAP`
 * (`vtscore/eval/autopilot_flow.py`).
 */
export const STOP_CAP_VOTES = 200;

/** The phases before the first learned sort: no trained detector yet. */
const OPENING_PHASES: readonly AutopilotPhase[] = ['good', 'bad', 'more'];

/**
 * The run reached the click cap still in its opening (#4903): {@link STOP_CAP_VOTES}
 * votes without the Goods a detector needs, so there is no line to check and
 * no Done to give. Autopilot tells the user it has stopped and offers a
 * different sort. Display only: the phase, and so what Autopilot picks if the
 * user votes on, is unchanged, and finding the Goods later moves the run to
 * the check before Done and Done at once.
 */
export function cappedInOpening(st: AutopilotState): boolean {
  return st.atCap && OPENING_PHASES.includes(st.phase);
}

const INITIAL_STATE: AutopilotState = {
  phase: 'idle',
  goodToStart: 3,
  badToStart: 4,
  moreToStart: 0,
  moreDryRun: 16,
  moreMisses: 0,
  moreDone: false,
  goodMisses: 0,
  goodRanDry: false,
  dryRunStop: false,
  smartStatus: '',
  stableStatus: '',
  stablePlateau: false,
  spanStatus: '',
  fracDiversity: 0,
  spanTarget: 0,
  levelMet: false,
  expectedFbeta: null,
  atCap: false,
  retrainMode: false,
  doneReached: false,
  spotCheck: null,
  doneChecked: false,
};

@Injectable({ providedIn: 'root' })
export class AutopilotStateService {
  private readonly stateSubject = new BehaviorSubject<AutopilotState>({ ...INITIAL_STATE });

  readonly state$ = this.stateSubject.asObservable();

  /**
   * The hand-off hint on show, or `null` (#4680). Toasty gives it under the
   * top bar's Dashboard button, which is in `AppComponent`, so it lives here
   * rather than on the panel that raises it. Cleared by any vote after it
   * went up, by a new run or a stop, and when the Train view goes away.
   */
  readonly handoff = signal<AutopilotHandoff | null>(null);

  /**
   * Whether the terminal-phase hand-off (Toasty's "trained" hint) has already been
   * shown for the current autopilot run.
   *
   * This lives on the service rather than on the panel component on purpose:
   * the panel is destroyed and rebuilt every time the user switches the
   * left-panel tab, so a component-scoped flag would raise the hint again on each
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
  private readonly startedTrained = signal(false);

  /** Whether {@link noteInitialLabelset} has taken its one reading this run. */
  private readonly initialLabelsetKnown = signal(false);

  /**
   * Whether this run started on a detector with no labels (#4807): the run
   * that trains it, which Toasty's phase hints are for, as the hand-off is.
   * False until the labelset has been read, so a run never reads as training
   * from not-yet-loaded zeroes.
   */
  readonly startedUntrained = computed(() => this.initialLabelsetKnown() && !this.startedTrained());

  /**
   * The vote counts at the last {@link checkPhaseTransition}. The phase check
   * sees only counts, so the "more" walk reads each vote's outcome off which
   * count rose (``vtscore/eval/autopilot_flow.py`` reads it the same way).
   */
  private lastCounts: { good: number; bad: number } | null = null;
  /** The dataset size at the last {@link checkPhaseTransition}, for re-deriving the phase when a check starts or ends. */
  private lastTotal = 0;

  /**
   * Whether this run has taken its first labeling-status reading. A run whose
   * first reading is already all green opened on a finished detector (the
   * Train view re-entered after Done): it latches `doneChecked`, so the check
   * before Done belongs to the session that brings the detector there, and
   * re-entry after a server restart, which forgets the last check, runs none.
   */
  private statusRead = false;

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
    if (this.initialLabelsetKnown()) return;
    this.initialLabelsetKnown.set(true);
    this.startedTrained.set(goodCount > 0 || badCount > 0);
    const retrainMode = goodCount > 0 && badCount > 0;
    // A detector already at the cap (#4903) is past Done: like one whose first
    // status reads Done, it has nothing to check before Done.
    const atCap = goodCount + badCount >= STOP_CAP_VOTES;
    const current = this.stateSubject.value;
    if (retrainMode !== current.retrainMode || (atCap && !current.doneChecked)) {
      this.stateSubject.next({
        ...current,
        retrainMode,
        moreDone: current.moreDone || retrainMode,
        doneChecked: current.doneChecked || atCap,
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
    this.initialLabelsetKnown.set(false);
    this.startedTrained.set(false);
  }

  /**
   * Whether reaching a terminal phase right now should be announced to the
   * user: this run has to have started from an untrained detector (see
   * {@link startedTrained}), the labelset must have been read at least once so
   * we are not guessing from not-yet-loaded zeroes, and the hand-off must not
   * already have been offered.
   */
  get shouldAnnounceCompletion(): boolean {
    return this.startedUntrained() && !this.completionAnnounced;
  }

  /** Record that the completion hand-off has been offered for this run. */
  markCompletionAnnounced(): void {
    this.completionAnnounced = true;
  }

  activate(retrainMode = false): void {
    if (this.running) return;
    this.handoff.set(null);
    this.completionAnnounced = false;
    this.startedTrained.set(false);
    this.initialLabelsetKnown.set(false);
    this.lastCounts = null;
    this.lastTotal = 0;
    this.statusRead = false;
    this.stateSubject.next({
      ...this.stateSubject.value,
      phase: 'good',
      moreMisses: 0,
      goodMisses: 0,
      goodRanDry: false,
      // A detector that already had labels starts in retrain mode, where every
      // phase draws off the learned sort: there is no seed sort to walk.
      moreDone: retrainMode,
      smartStatus: '',
      stableStatus: '',
      stablePlateau: false,
      spanStatus: '',
      fracDiversity: 0,
      spanTarget: 0,
      atCap: false,
      retrainMode,
      doneReached: false,
      spotCheck: null,
      doneChecked: false,
    });
  }

  deactivate(): void {
    this.handoff.set(null);
    this.stateSubject.next({ ...this.stateSubject.value, phase: 'idle', spotCheck: null });
  }

  /**
   * A spot check starts inside the run: the phase is `spot` until
   * {@link endSpotCheck}. The view calls it when the server says a weak check
   * is due, and when the phase machine reached `spot` for the check before Done.
   */
  beginSpotCheck(kind: SpotCheckKind): void {
    const st = this.stateSubject.value;
    if (st.phase === 'idle' || st.spotCheck !== null) return;
    this.stateSubject.next({ ...st, spotCheck: kind, phase: 'spot' });
  }

  /**
   * The running check closed, however it ended: finished, stopped, or refused
   * by the server. Closing the check before Done latches it for the run.
   */
  endSpotCheck(): void {
    const st = this.stateSubject.value;
    if (st.spotCheck === null) return;
    this.stateSubject.next({ ...st, spotCheck: null, doneChecked: st.doneChecked || st.spotCheck === 'done' });
    this.rederive();
  }

  /**
   * The check before Done does not run (`done_check_due` false: a check ended
   * moments ago, or the line has none to run): latch it, and the phase goes on
   * to Done.
   */
  skipDoneCheck(): void {
    const st = this.stateSubject.value;
    if (st.phase === 'idle' || st.doneChecked) return;
    this.stateSubject.next({ ...st, doneChecked: true });
    this.rederive();
  }

  /** Re-run the phase check on the last counts, after a change that is not a vote. */
  private rederive(): void {
    const counts = this.lastCounts;
    if (counts) this.checkPhaseTransition(counts.good, counts.bad, this.lastTotal);
  }

  updateFromLabelingStatus(status: LabelingStatusResponse): void {
    const current = this.stateSubject.value;
    const allGreen =
      status.smart.status === 'green' && status.stable.status === 'green' && status.span.status === 'green';
    const levelMet = status.level?.met === true;
    const firstRead = current.phase !== 'idle' && !this.statusRead;
    if (current.phase !== 'idle') this.statusRead = true;
    this.stateSubject.next({
      ...current,
      // A detector that already reads Done on its first status (the Train view
      // re-entered on a finished detector) has nothing to check before Done.
      doneChecked: current.doneChecked || (firstRead && (allGreen || levelMet)),
      levelMet,
      expectedFbeta: status.level?.expected_fbeta ?? null,
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
    this.lastTotal = totalCount;
    // The Good walk's run of misses (#4731), read the same way. A miss counts
    // only with a Good already in hand, so the run is the misses since the
    // latest Good, and a walk that has found nothing never runs dry.
    if (prev && st.phase === 'good' && !st.goodRanDry) {
      if (goodCount > prev.good) {
        st = { ...st, goodMisses: 0 };
      } else if (badCount > prev.bad && prev.good >= 1) {
        const goodMisses = st.goodMisses + 1;
        const goodRanDry = goodMisses >= st.moreDryRun;
        st = { ...st, goodMisses, goodRanDry, moreDone: st.moreDone || goodRanDry };
      }
    }
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
    const atCap = goodCount + badCount >= STOP_CAP_VOTES;

    // Cap each phase target at the most votes of that class the dataset could
    // still yield (current votes of that class + everything unlabeled), so a
    // tiny dataset can still satisfy — and advance past — the initial phases.
    let effGoodTarget = Math.min(st.goodToStart, goodCount + remainingUnlabeled);
    // A Good walk that ran dry has met its target with the Goods it found (#4731).
    if (st.goodRanDry && goodCount >= 1) effGoodTarget = Math.min(effGoodTarget, goodCount);
    const effBadTarget = Math.min(st.badToStart, badCount + remainingUnlabeled);
    const effMoreTarget = Math.min(st.moreToStart, goodCount + remainingUnlabeled);

    // Derive the correct phase from current counts and indicator statuses.
    // This allows both forward and backward transitions (e.g. if votes are
    // cleared or un-toggled, the phase regresses to match).
    let nextPhase: AutopilotPhase;
    if (st.spotCheck !== null) {
      // A spot check is running: its picks are on screen, whatever the counts.
      nextPhase = 'spot';
    } else if (goodCount < effGoodTarget) {
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
    } else if (allLightsGreen(st) || st.levelMet || atCap) {
      // Every light is green, or the labels line expects Done's level of F-beta
      // at the detector's balance (#4878), or the votes reached the cap (#4903):
      // check the line once before reporting Done. With nothing left unlabeled
      // there is nothing to check.
      nextPhase = st.doneChecked || remainingUnlabeled === 0 ? 'done' : 'spot';
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
    const opening = OPENING_PHASES.includes(nextPhase);
    const moreDone = st.moreDone || !opening;
    // Display only (#4621): never read above, so the phase is the same with or
    // without it.
    const doneReached = nextPhase === 'done' || (st.doneReached && !opening);
    if (
      nextPhase !== st.phase
      || st !== this.stateSubject.value
      || moreDone !== st.moreDone
      || doneReached !== st.doneReached
      || atCap !== st.atCap
    ) {
      this.stateSubject.next({ ...st, phase: nextPhase, moreDone, doneReached, atCap });
    }
  }

  clear(): void {
    this.handoff.set(null);
    this.completionAnnounced = false;
    this.startedTrained.set(false);
    this.initialLabelsetKnown.set(false);
    this.lastCounts = null;
    this.lastTotal = 0;
    this.statusRead = false;
    this.stateSubject.next({ ...INITIAL_STATE });
  }
}
