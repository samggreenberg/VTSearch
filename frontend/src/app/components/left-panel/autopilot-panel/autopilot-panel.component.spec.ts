import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter, Router } from '@angular/router';
import { AutopilotPanelComponent } from './autopilot-panel.component';
import { AutopilotStateService } from '../../../services/autopilot-state.service';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

/** Green-across-the-board status payload: drives the phase machine to 'done'. */
const ALL_GREEN = {
  good_count: 0,
  bad_count: 0,
  total_count: 0,
  smart: { status: 'green' },
  stable: { status: 'green' },
  span: { status: 'green' },
};

/** Good vote ids 1..n: 20 of them carry a run past the "more" walk (#4282). */
function goods(n: number): Set<number> {
  return new Set(Array.from({ length: n }, (_, i) => i + 1));
}

/** Bad vote ids, numbered clear of {@link goods}. */
function bads(n: number): Set<number> {
  return new Set(Array.from({ length: n }, (_, i) => i + 101));
}

describe('AutopilotPanelComponent', () => {
  let component: AutopilotPanelComponent;
  let fixture: ComponentFixture<AutopilotPanelComponent>;
  let autopilotState: AutopilotStateService;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AutopilotPanelComponent],
      providers: [...provideZoneless(), provideRouter([])],
    }).compileComponents();

    fixture = TestBed.createComponent(AutopilotPanelComponent);
    component = fixture.componentInstance;
    autopilotState = TestBed.inject(AutopilotStateService);
    await settleZoneless(fixture);
  });

  afterEach(() => {
    autopilotState.clear();
  });

  /**
   * Drive the phase machine to 'done' (every indicator green, votes past the
   * initial targets) — the state that offers the completion hand-off.
   */
  function reachDone(): void {
    // votesLoaded with a 0/0 labelset is "brand-new detector, votes have
    // arrived" — the run that actually did the training.
    fixture.componentRef.setInput('votesLoaded', true);
    fixture.componentRef.setInput('goodVotes', goods(20));
    fixture.componentRef.setInput('badVotes', bads(5));
    fixture.componentRef.setInput('labelingStatus', ALL_GREEN);
    TestBed.tick();
  }

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should auto-start on init in good phase', () => {
    expect(component.state.phase).toBe('good');
    expect(component.running).toBe(true);
  });

  it('should emit started on init', async () => {
    autopilotState.clear();
    const fresh = TestBed.createComponent(AutopilotPanelComponent);
    const comp = fresh.componentInstance;
    vi.spyOn(comp.started, 'emit');
    await settleZoneless(fresh);
    expect(comp.started.emit).toHaveBeenCalled();
  });

  it('should show steps immediately', () => {
    const steps = fixture.nativeElement.querySelectorAll('.ap-step');
    expect(steps.length).toBe(6);
  });

  it('should transition from good to bad phase', async () => {
    fixture.componentRef.setInput('goodVotes', new Set([1, 2, 3]));
    await settleZoneless(fixture);
    expect(component.state.phase).toBe('bad');
  });

  it('should transition from bad to the more walk, then to hard at its target', async () => {
    fixture.componentRef.setInput('goodVotes', goods(3));
    await settleZoneless(fixture);
    expect(component.state.phase).toBe('bad');

    fixture.componentRef.setInput('badVotes', bads(4));
    await settleZoneless(fixture);
    expect(component.state.phase).toBe('more');
    const more = component.steps.find((st: any) => st.phase === 'more');
    expect(more!.label).toBe('Find More Goods.');
    expect(more!.detail).toBe('3/20 good labels');

    fixture.componentRef.setInput('goodVotes', goods(20));
    await settleZoneless(fixture);
    expect(component.state.phase).toBe('hard');
  });

  it('should transition from hard to new when smart+stable are green', async () => {
    // Advance to hard phase
    fixture.componentRef.setInput('goodVotes', goods(20));
    fixture.componentRef.setInput('badVotes', bads(4));
    await settleZoneless(fixture);
    expect(component.state.phase).toBe('hard');

    fixture.componentRef.setInput('labelingStatus', {
      good_count: 0,
      bad_count: 0,
      total_count: 0,
      smart: { status: 'green' },
      stable: { status: 'green' },
      span: { status: '' },
    });
    await settleZoneless(fixture);
    expect(component.state.phase).toBe('new');
  });

  it('should transition from new to done when span is green', async () => {
    // Advance to new phase
    fixture.componentRef.setInput('goodVotes', goods(20));
    fixture.componentRef.setInput('badVotes', bads(10));
    fixture.componentRef.setInput('labelingStatus', {
      good_count: 0,
      bad_count: 0,
      total_count: 0,
      smart: { status: 'green' },
      stable: { status: 'green' },
      span: { status: '' },
    });
    await settleZoneless(fixture);
    expect(component.state.phase).toBe('new');

    fixture.componentRef.setInput('labelingStatus', {
      good_count: 0,
      bad_count: 0,
      total_count: 0,
      smart: { status: 'green' },
      stable: { status: 'green' },
      span: { status: 'green' },
    });
    await settleZoneless(fixture);
    expect(component.state.phase).toBe('done');
  });

  it('should bounce from new back to hard when smart drops to yellow', async () => {
    // Advance to new phase
    fixture.componentRef.setInput('goodVotes', goods(20));
    fixture.componentRef.setInput('badVotes', bads(10));
    fixture.componentRef.setInput('labelingStatus', {
      good_count: 0,
      bad_count: 0,
      total_count: 0,
      smart: { status: 'green' },
      stable: { status: 'green' },
      span: { status: '' },
    });
    await settleZoneless(fixture);
    expect(component.state.phase).toBe('new');

    // A surprise vote causes smart to drop
    fixture.componentRef.setInput('labelingStatus', {
      good_count: 0,
      bad_count: 0,
      total_count: 0,
      smart: { status: 'yellow' },
      stable: { status: 'green' },
      span: { status: 'yellow' },
    });
    await settleZoneless(fixture);
    expect(component.state.phase).toBe('hard');
  });

  describe('step light (#4319)', () => {
    /** The active step's light colour, straight off the step model. */
    function activeLight(): string | undefined {
      return component.steps.find((st) => st.state === 'active')?.light?.color;
    }

    /** A labeling-status payload carrying the given indicator readings. */
    function status(smart: string, stable: string, span: Record<string, unknown>) {
      return { good_count: 0, bad_count: 0, total_count: 0, smart: { status: smart }, stable: { status: stable }, span };
    }

    it('gives the active step exactly one light and no other step a light', async () => {
      await settleZoneless(fixture);
      const lit = component.steps.filter((st) => st.light !== null);
      expect(lit.map((st) => st.phase)).toEqual(['good']);
      expect(fixture.nativeElement.querySelectorAll('.ap-light').length).toBe(1);
      expect(fixture.nativeElement.querySelectorAll('.ap-step.active .ap-light').length).toBe(1);
    });

    it('shows the light in the collapsed rail too', async () => {
      fixture.componentRef.setInput('collapsed', true);
      await settleZoneless(fixture);
      const lights = fixture.nativeElement.querySelectorAll('.collapsed-step.active .ap-light');
      expect(lights.length).toBe(1);
      expect(lights[0].getAttribute('data-color')).toBe('red');
    });

    it('count steps go red, then yellow at half the target', async () => {
      // Initial goods: target 3.
      expect(activeLight()).toBe('red');
      fixture.componentRef.setInput('goodVotes', goods(1));
      await settleZoneless(fixture);
      expect(activeLight()).toBe('red');
      fixture.componentRef.setInput('goodVotes', goods(2));
      await settleZoneless(fixture);
      expect(activeLight()).toBe('yellow');
      const rendered = fixture.nativeElement.querySelector('.ap-step.active .ap-light');
      expect(rendered.getAttribute('data-color')).toBe('yellow');
      expect(rendered.title).toContain('Yellow.');

      // Initial bads: target 4, a fresh light that starts red again.
      fixture.componentRef.setInput('goodVotes', goods(3));
      fixture.componentRef.setInput('badVotes', bads(1));
      await settleZoneless(fixture);
      expect(component.state.phase).toBe('bad');
      expect(activeLight()).toBe('red');
      fixture.componentRef.setInput('badVotes', bads(2));
      await settleZoneless(fixture);
      expect(activeLight()).toBe('yellow');

      // More goods: target 20.
      fixture.componentRef.setInput('badVotes', bads(4));
      await settleZoneless(fixture);
      expect(component.state.phase).toBe('more');
      expect(activeLight()).toBe('red');
      fixture.componentRef.setInput('goodVotes', goods(10));
      await settleZoneless(fixture);
      expect(activeLight()).toBe('yellow');
    });

    it('a count step capped to what a tiny dataset holds is paced against the cap', async () => {
      // 2 items: at most 2 goods, so one good is already halfway.
      fixture.componentRef.setInput('datasetSize', 2);
      fixture.componentRef.setInput('goodVotes', goods(1));
      await settleZoneless(fixture);
      expect(component.state.phase).toBe('good');
      expect(component.effGoodTarget).toBe(2);
      expect(activeLight()).toBe('yellow');
    });

    it('the boundary step shows the lower of Smart and Stable', async () => {
      fixture.componentRef.setInput('goodVotes', goods(20));
      fixture.componentRef.setInput('badVotes', bads(4));
      await settleZoneless(fixture);
      expect(component.state.phase).toBe('hard');
      // Nothing reported yet reads red.
      expect(activeLight()).toBe('red');

      fixture.componentRef.setInput('labelingStatus', status('yellow', 'red', { status: '' }));
      await settleZoneless(fixture);
      expect(activeLight()).toBe('red');

      fixture.componentRef.setInput('labelingStatus', status('green', 'yellow', { status: '' }));
      await settleZoneless(fixture);
      expect(component.state.phase).toBe('hard');
      expect(activeLight()).toBe('yellow');
      const light = component.steps.find((st) => st.phase === 'hard')!.light!;
      expect(light.title).toContain('Smart (green)');
      expect(light.title).toContain('Stable (yellow)');
    });

    it('the diversity step is paced against the Span green target', async () => {
      fixture.componentRef.setInput('goodVotes', goods(20));
      fixture.componentRef.setInput('badVotes', bads(10));
      fixture.componentRef.setInput(
        'labelingStatus',
        status('green', 'green', { status: 'yellow', diversity_level: 13, target: 40 }),
      );
      await settleZoneless(fixture);
      expect(component.state.phase).toBe('new');
      // Past the backend's own yellow cutoff (10), but not yet halfway to 40.
      expect(activeLight()).toBe('red');
      const step = component.steps.find((st) => st.phase === 'new')!;
      expect(step.detail).toBe('Diversity: 13/40');
      expect(step.light!.title).toContain('cover');
      expect(step.light!.title).toContain('13 of the 40');

      fixture.componentRef.setInput(
        'labelingStatus',
        status('green', 'green', { status: 'yellow', diversity_level: 20, target: 40 }),
      );
      await settleZoneless(fixture);
      expect(activeLight()).toBe('yellow');

      // Green is the indicator's call, never the count's: the level can sit at
      // the target a poll ahead of the status that confirms it.
      fixture.componentRef.setInput(
        'labelingStatus',
        status('green', 'green', { status: 'yellow', diversity_level: 40, target: 40 }),
      );
      await settleZoneless(fixture);
      expect(component.state.phase).toBe('new');
      expect(activeLight()).toBe('yellow');
    });

    it('the diversity step reads red until a target is reported', async () => {
      fixture.componentRef.setInput('goodVotes', goods(20));
      fixture.componentRef.setInput('badVotes', bads(10));
      fixture.componentRef.setInput('labelingStatus', status('green', 'green', { status: 'yellow' }));
      await settleZoneless(fixture);
      expect(component.state.phase).toBe('new');
      expect(activeLight()).toBe('red');
    });

    it('at Done the active step is green, drawn as the check a finished step keeps', async () => {
      fixture.componentRef.setInput('goodVotes', goods(20));
      fixture.componentRef.setInput('badVotes', bads(5));
      fixture.componentRef.setInput('labelingStatus', ALL_GREEN);
      await settleZoneless(fixture);
      expect(component.state.phase).toBe('done');
      // Past Done the active step is Keep Improving (#4621), green while every
      // indicator is.
      expect(component.steps.find((st) => st.state === 'active')?.phase).toBe('improve');
      expect(activeLight()).toBe('green');

      // Red circle, yellow circle, green check: no green circle is ever drawn.
      const el: HTMLElement = fixture.nativeElement;
      expect(el.querySelectorAll('.ap-light').length).toBe(0);
      const activeCheck = el.querySelector('.ap-step.active .ap-check')!;
      expect(activeCheck).toBeTruthy();
      expect(activeCheck.getAttribute('aria-label')).toBe('Step progress: green');
      // ...and every finished step before it, Done included, carries the same check.
      expect(el.querySelectorAll('.ap-step.done .ap-check').length).toBe(6);

      fixture.componentRef.setInput('collapsed', true);
      await settleZoneless(fixture);
      expect(el.querySelectorAll('.collapsed-step.active .ap-check').length).toBe(1);
      expect(el.querySelectorAll('.ap-light').length).toBe(0);
    });
  });

  describe('past Done (#4621)', () => {
    /** A labeling-status payload carrying the given indicator readings. */
    function status(smart: string, stable: string, span: string) {
      return { good_count: 0, bad_count: 0, total_count: 0, smart: { status: smart }, stable: { status: stable }, span: { status: span } };
    }

    function active() {
      return component.steps.find((st) => st.state === 'active')!;
    }

    it('keeps Done checked and makes Keep Improving the active step', async () => {
      reachDone();
      await settleZoneless(fixture);

      expect(component.steps.map((st) => st.phase)).toEqual(['good', 'bad', 'more', 'hard', 'new', 'done', 'improve']);
      expect(component.steps.slice(0, 6).every((st) => st.state === 'done')).toBe(true);
      const step = active();
      expect(step.phase).toBe('improve');
      expect(step.label).toBe('Keep Improving.');
      expect(step.stepNumber).toBe(7);
      expect(step.detail).toBe('All indicators green');
      expect(step.intent).toContain('Optional');

      const el: HTMLElement = fixture.nativeElement;
      expect(el.querySelectorAll('.ap-step').length).toBe(7);
      expect(el.querySelector('.ap-step.active .ap-step-label')!.textContent!.trim()).toBe('Keep Improving.');
    });

    it('does not fall back to Refine Boundary when a vote knocks an indicator off green', async () => {
      reachDone();
      await settleZoneless(fixture);

      fixture.componentRef.setInput('goodVotes', goods(21));
      fixture.componentRef.setInput('labelingStatus', status('green', 'yellow', 'green'));
      await settleZoneless(fixture);
      // The phase underneath still follows the indicators: it decides the picks.
      expect(component.state.phase).toBe('hard');
      // ...but the panel stays put, Done still checked.
      expect(component.steps.find((st) => st.phase === 'done')!.state).toBe('done');
      expect(component.steps.find((st) => st.phase === 'hard')!.state).toBe('done');
      const step = active();
      expect(step.phase).toBe('improve');
      expect(step.detail).toBe('Showing boundary items');
      expect(step.light!.color).toBe('yellow');
      expect(step.light!.title).toContain('Stable (yellow)');
      expect(fixture.nativeElement.querySelectorAll('.ap-step').length).toBe(7);

      // Back to green: the same step, its light green again.
      fixture.componentRef.setInput('labelingStatus', ALL_GREEN);
      await settleZoneless(fixture);
      expect(component.state.phase).toBe('done');
      expect(active().phase).toBe('improve');
      expect(active().light!.color).toBe('green');
    });

    it('says when it is offering diverse items while only Span lags', async () => {
      reachDone();
      await settleZoneless(fixture);

      fixture.componentRef.setInput('labelingStatus', status('green', 'green', 'red'));
      await settleZoneless(fixture);
      expect(component.state.phase).toBe('new');
      expect(active().phase).toBe('improve');
      expect(active().detail).toBe('Showing diverse items');
      expect(active().light!.color).toBe('red');
    });

    it('shows Improve in the collapsed rail', async () => {
      reachDone();
      fixture.componentRef.setInput('collapsed', true);
      await settleZoneless(fixture);
      const label = fixture.nativeElement.querySelector('.collapsed-step.active .collapsed-step-label');
      expect(label.textContent.trim()).toBe('Improve');
      expect(fixture.nativeElement.querySelectorAll('.collapsed-step').length).toBe(7);
    });

    it('gives the full step list back when votes fall back into the opening', async () => {
      reachDone();
      await settleZoneless(fixture);

      fixture.componentRef.setInput('goodVotes', goods(1));
      await settleZoneless(fixture);
      expect(component.state.phase).toBe('good');
      expect(component.steps.map((st) => st.phase)).toEqual(['good', 'bad', 'more', 'hard', 'new', 'done']);
      expect(active().phase).toBe('good');
    });

    it('starts each new run without it', async () => {
      reachDone();
      await settleZoneless(fixture);

      component.deactivate();
      component.activate();
      fixture.componentRef.setInput('labelingStatus', status('yellow', 'yellow', 'yellow'));
      await settleZoneless(fixture);
      expect(component.state.phase).toBe('hard');
      expect(component.steps.map((st) => st.phase)).toEqual(['good', 'bad', 'more', 'hard', 'new', 'done']);
      expect(active().phase).toBe('hard');
    });
  });

  it('should deactivate autopilot', () => {
    vi.spyOn(component.stopped, 'emit');
    component.deactivate();
    expect(component.state.phase).toBe('idle');
    expect(component.running).toBe(false);
    expect(component.stopped.emit).toHaveBeenCalled();
  });

  it('should re-activate after deactivate', () => {
    component.deactivate();
    vi.spyOn(component.started, 'emit');
    component.activate();
    expect(component.state.phase).toBe('good');
    expect(component.running).toBe(true);
    expect(component.started.emit).toHaveBeenCalled();
  });

  it('should not re-activate if already running', () => {
    vi.spyOn(component.started, 'emit');
    component.activate();
    expect(component.started.emit).not.toHaveBeenCalled();
  });

  it('should regress from hard to good when vote counts drop to zero', async () => {
    // Advance to hard phase with sufficient votes
    fixture.componentRef.setInput('goodVotes', goods(20));
    fixture.componentRef.setInput('badVotes', bads(4));
    await settleZoneless(fixture);
    expect(component.state.phase).toBe('hard');

    // Votes are cleared (e.g. new detector session); phase should regress
    fixture.componentRef.setInput('goodVotes', new Set());
    fixture.componentRef.setInput('badVotes', new Set());
    await settleZoneless(fixture);
    expect(component.state.phase).toBe('good');
  });

  it('should regress from hard to bad when good count drops below threshold', async () => {
    fixture.componentRef.setInput('goodVotes', goods(20));
    fixture.componentRef.setInput('badVotes', bads(4));
    await settleZoneless(fixture);
    expect(component.state.phase).toBe('hard');

    // Good votes drop below threshold but bad are still sufficient
    fixture.componentRef.setInput('goodVotes', goods(1));
    fixture.componentRef.setInput('badVotes', bads(4));
    await settleZoneless(fixture);
    expect(component.state.phase).toBe('good');
  });

  it('should show tooltip on each step label via title attribute', () => {
    const stepLabels = fixture.nativeElement.querySelectorAll('.ap-step-label');
    expect(stepLabels.length).toBe(6);
    // Active step (phase 1) leads with phase intent and ends with reselect hint
    expect(stepLabels[0].title).toContain('Phase 1');
    expect(stepLabels[0].title).toContain('Find initial goods');
    expect(stepLabels[0].title).toContain('reselect');
    // Future steps show phase intent only
    expect(stepLabels[1].title).toContain('Phase 2');
    expect(stepLabels[1].title).toContain('Find initial bads');
    expect(stepLabels[2].title).toContain('Phase 3');
    expect(stepLabels[2].title).toContain('Find more goods');
    expect(stepLabels[3].title).toContain('Refine the cutoff');
    expect(stepLabels[4].title).toContain('Cover a broad mix');
  });

  it('should show phase intent tooltip on each collapsed-step dot', async () => {
    fixture.componentRef.setInput('collapsed', true);
    await settleZoneless(fixture);
    const dots = fixture.nativeElement.querySelectorAll('.collapsed-step');
    expect(dots.length).toBe(6);
    expect(dots[0].title).toContain('Phase 1');
    expect(dots[0].title).toContain('Find initial goods');
    expect(dots[0].title).toContain('reselect');
    expect(dots[2].title).toContain('Phase 3');
    expect(dots[2].title).toContain('Find more goods');
    expect(dots[3].title).toContain('Phase 4');
    expect(dots[3].title).toContain('Refine the cutoff');
    expect(dots[3].title).toContain('uncertain items');
  });

  it('should emit refocus when clicking the active step', async () => {
    vi.spyOn(component.refocus, 'emit');
    await settleZoneless(fixture);
    const activeStep = fixture.nativeElement.querySelector('.ap-step.active');
    activeStep.click();
    expect(component.refocus.emit).toHaveBeenCalled();
  });

  it('should not emit refocus when clicking a future step', async () => {
    vi.spyOn(component.refocus, 'emit');
    await settleZoneless(fixture);
    const futureSteps = fixture.nativeElement.querySelectorAll('.ap-step.future');
    futureSteps[0].click();
    expect(component.refocus.emit).not.toHaveBeenCalled();
  });

  it('activate without labelset labels should not enter retrain mode', async () => {
    autopilotState.clear();
    const fresh = TestBed.createComponent(AutopilotPanelComponent);
    fresh.componentRef.setInput('labelsetGoodCount', 0);
    fresh.componentRef.setInput('labelsetBadCount', 0);
    await settleZoneless(fresh);
    expect(fresh.componentInstance.state.retrainMode).toBe(false);
  });

  it('activate with detector labels from another dataset should enter retrain mode', async () => {
    autopilotState.clear();
    const fresh = TestBed.createComponent(AutopilotPanelComponent);
    // Simulate "trained on DatasetA, switched to DatasetB with 0 votes here":
    // current-dataset goodVotes/badVotes empty, but labelset counts positive.
    fresh.componentRef.setInput('labelsetGoodCount', 5);
    fresh.componentRef.setInput('labelsetBadCount', 4);
    fresh.componentRef.setInput('goodVotes', new Set());
    fresh.componentRef.setInput('badVotes', new Set());
    await settleZoneless(fresh);
    expect(fresh.componentInstance.state.retrainMode).toBe(true);
    // Still in 'good' phase since current-dataset votes are below threshold.
    expect(fresh.componentInstance.state.phase).toBe('good');
  });

  it('should mark current step as active and future steps as future', () => {
    const steps = component.steps;
    expect(steps[0].state).toBe('active');
    expect(steps[1].state).toBe('future');
    expect(steps[2].state).toBe('future');
  });

  it('caps the good-phase target for display on a tiny dataset', () => {
    fixture.componentRef.setInput('datasetSize', 1);
    fixture.componentRef.setInput('goodVotes', new Set());
    fixture.componentRef.setInput('badVotes', new Set());
    // Default target is 3, but a 1-item dataset can supply at most 1 good.
    expect(component.effGoodTarget).toBe(1);
  });

  it('reaches the exhausted state and renders a note when a 1-item dataset is labeled', async () => {
    // Drive the inputs the way the template binding does, so OnPush re-renders
    // and the phase-transition effect fires the dataset-size-aware phase check.
    fixture.componentRef.setInput('datasetSize', 1);
    fixture.componentRef.setInput('goodVotes', new Set([1]));
    fixture.componentRef.setInput('badVotes', new Set());
    await settleZoneless(fixture);
    expect(component.state.phase).toBe('exhausted');
    expect(component.exhausted).toBe(true);
    const note = fixture.nativeElement.querySelector('.autopilot-exhausted');
    expect(note).toBeTruthy();
    expect(note.textContent).toContain('Nothing left to label');
    // The five-step list is replaced by the terminal note.
    expect(fixture.nativeElement.querySelectorAll('.ap-step').length).toBe(0);
  });

  describe('completion hand-off', () => {
    it("raises Toasty's hand-off and moves nobody (#4680)", async () => {
      const router = TestBed.inject(Router);
      const navigate = vi.spyOn(router, 'navigate').mockResolvedValue(true);
      reachDone();
      await settleZoneless(fixture);

      expect(component.state.phase).toBe('done');
      // 20 Good + 5 Bad: the vote total it went up at.
      expect(autopilotState.handoff()).toEqual({ kind: 'done', votes: 25, dryRun: undefined });
      // No dialog, and nothing on a timer: the user is never moved without asking.
      expect(fixture.nativeElement.querySelector('vt-modal')).toBeNull();
      expect(navigate).not.toHaveBeenCalled();
    });

    it('drops the hand-off on the next vote: voting on is the answer', async () => {
      reachDone();
      await settleZoneless(fixture);
      expect(autopilotState.handoff()).toBeTruthy();

      fixture.componentRef.setInput('goodVotes', goods(21));
      await settleZoneless(fixture);
      expect(autopilotState.handoff()).toBeNull();
    });

    it('drops the hand-off when Autopilot is stopped', async () => {
      reachDone();
      await settleZoneless(fixture);
      component.deactivate();
      expect(autopilotState.handoff()).toBeNull();
    });

    it('announces the exhausted state too', async () => {
      fixture.componentRef.setInput('labelsetGoodCount', 0);
      fixture.componentRef.setInput('labelsetBadCount', 0);
      fixture.componentRef.setInput('votesLoaded', true);
      fixture.componentRef.setInput('datasetSize', 1);
      fixture.componentRef.setInput('goodVotes', new Set([1]));
      await settleZoneless(fixture);

      expect(component.state.phase).toBe('exhausted');
      expect(autopilotState.handoff()?.kind).toBe('all-labeled');
    });

    it('does not announce again when the user returns to the Train window', async () => {
      reachDone();
      await settleZoneless(fixture);
      expect(autopilotState.handoff()).toBeTruthy();

      // Leaving and coming back rebuilds the panel and (via label-view) clears
      // the service, so this is a brand-new run — but the detector it finds is
      // already trained, which is the whole point: the user came back to keep
      // working, not to be told to leave again (#3201).
      fixture.destroy();
      autopilotState.clear();

      const revisit = TestBed.createComponent(AutopilotPanelComponent);
      revisit.componentRef.setInput('labelsetGoodCount', 5);
      revisit.componentRef.setInput('labelsetBadCount', 5);
      revisit.componentRef.setInput('votesLoaded', true);
      revisit.componentRef.setInput('goodVotes', goods(20));
      revisit.componentRef.setInput('badVotes', bads(5));
      revisit.componentRef.setInput('labelingStatus', ALL_GREEN);
      await settleZoneless(revisit);

      expect(revisit.componentInstance.state.phase).toBe('done');
      expect(autopilotState.handoff()).toBeNull();
    });

    it('does not announce for a detector that started partially trained', async () => {
      autopilotState.clear();
      const fresh = TestBed.createComponent(AutopilotPanelComponent);
      // Trained on DatasetA, now continuing on DatasetB: no votes here yet, but
      // the labelset carries the earlier ones.
      fresh.componentRef.setInput('labelsetGoodCount', 7);
      fresh.componentRef.setInput('labelsetBadCount', 6);
      fresh.componentRef.setInput('votesLoaded', true);
      await settleZoneless(fresh);
      expect(fresh.componentInstance.state.retrainMode).toBe(true);

      // ...and it finishes here too. Still no hand-off: this detector was
      // already trained when the run started.
      fresh.componentRef.setInput('goodVotes', goods(20));
      fresh.componentRef.setInput('badVotes', bads(5));
      fresh.componentRef.setInput('labelingStatus', ALL_GREEN);
      await settleZoneless(fresh);

      expect(fresh.componentInstance.state.phase).toBe('done');
      expect(autopilotState.handoff()).toBeNull();
    });

    it('re-reads the labelset when the active pair changes mid-run', async () => {
      // Switching detector clears the votes (votesLoaded → false) without
      // stopping autopilot, so the reading taken for the previous detector must
      // not decide the hand-off for the new one.
      fixture.componentRef.setInput('votesLoaded', true);
      fixture.componentRef.setInput('labelsetGoodCount', 0);
      fixture.componentRef.setInput('labelsetBadCount', 0);
      fixture.componentRef.setInput('goodVotes', new Set([1, 2, 3]));
      await settleZoneless(fixture);
      expect(autopilotState.shouldAnnounceCompletion).toBe(true);

      fixture.componentRef.setInput('votesLoaded', false);
      fixture.componentRef.setInput('goodVotes', new Set());
      await settleZoneless(fixture);

      // The new pair's detector is already trained: no hand-off is owed here.
      fixture.componentRef.setInput('votesLoaded', true);
      fixture.componentRef.setInput('labelsetGoodCount', 9);
      fixture.componentRef.setInput('labelsetBadCount', 9);
      fixture.componentRef.setInput('goodVotes', goods(20));
      fixture.componentRef.setInput('badVotes', bads(5));
      fixture.componentRef.setInput('labelingStatus', ALL_GREEN);
      await settleZoneless(fixture);

      expect(component.state.phase).toBe('done');
      expect(autopilotState.handoff()).toBeNull();
    });

    it('holds the hand-off until the labelset has actually loaded', async () => {
      autopilotState.clear();
      const pending = TestBed.createComponent(AutopilotPanelComponent);
      // votesLoaded still false: the 0/0 labelset counts are defaults, not
      // facts, so we must not read them as "brand-new detector" and announce.
      pending.componentRef.setInput('votesLoaded', false);
      pending.componentRef.setInput('goodVotes', goods(20));
      pending.componentRef.setInput('badVotes', bads(5));
      pending.componentRef.setInput('labelingStatus', ALL_GREEN);
      await settleZoneless(pending);

      expect(pending.componentInstance.state.phase).toBe('done');
      expect(autopilotState.handoff()).toBeNull();
    });
  });

  describe('on a document dataset (#4488)', () => {
    const off = { status: 'off' };
    const DOCUMENT = {
      good_count: 0,
      bad_count: 0,
      total_count: 0,
      smart: off,
      stable: off,
      span: off,
      stop_rule: 'dry_run',
      dry_run: { status: 'red', run: 0, target: 16 },
    };

    async function enterWalk(): Promise<void> {
      fixture.componentRef.setInput('votesLoaded', true);
      fixture.componentRef.setInput('labelingStatus', DOCUMENT);
      fixture.componentRef.setInput('goodVotes', goods(3));
      fixture.componentRef.setInput('badVotes', bads(4));
      await settleZoneless(fixture);
      expect(component.state.phase).toBe('more');
    }

    it('has no Boundary or Diversity step', async () => {
      await enterWalk();
      expect(component.steps.map((st) => st.phase)).toEqual(['good', 'bad', 'more', 'done']);
    });

    it('lights the walk by its run of misses', async () => {
      await enterWalk();
      for (let n = 5; n <= 12; n++) {
        fixture.componentRef.setInput('badVotes', bads(n));
        await settleZoneless(fixture);
      }
      const more = component.steps.find((st) => st.phase === 'more');
      expect(more?.detail).toBe('8/16 in a row not good');
      expect(more?.light?.color).toBe('yellow');
    });

    it('hands off as trained when the walk runs dry', async () => {
      await enterWalk();
      for (let n = 5; n <= 20; n++) {
        fixture.componentRef.setInput('badVotes', bads(n));
        await settleZoneless(fixture);
      }
      expect(component.state.phase).toBe('done');
      expect(autopilotState.handoff()).toMatchObject({ kind: 'done', dryRun: 16 });
      // Nothing undoes a dry run, so there is no Keep Improving step (#4621).
      expect(component.steps.map((st) => st.phase)).toEqual(['good', 'bad', 'more', 'done']);
      expect(component.steps.find((st) => st.state === 'active')?.phase).toBe('done');
    });
  });

  it('does not exhaust while the dataset size is unknown (datasetSize 0)', async () => {
    fixture.componentRef.setInput('datasetSize', 0);
    fixture.componentRef.setInput('goodVotes', new Set([1]));
    fixture.componentRef.setInput('badVotes', new Set());
    await settleZoneless(fixture);
    // Unknown size → uncapped → 1 good is not enough, still in good phase.
    expect(component.state.phase).toBe('good');
  });
});
