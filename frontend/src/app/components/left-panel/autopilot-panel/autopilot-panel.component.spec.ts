import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter, Router } from '@angular/router';
import { AutopilotPanelComponent } from './autopilot-panel.component';
import { AutopilotStateService } from '../../../services/autopilot-state.service';
import { ToastService } from '../../../services/toast.service';
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

describe('AutopilotPanelComponent', () => {
  let component: AutopilotPanelComponent;
  let fixture: ComponentFixture<AutopilotPanelComponent>;
  let autopilotState: AutopilotStateService;
  let toasts: ToastService;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AutopilotPanelComponent],
      providers: [...provideZoneless(), provideRouter([])],
    }).compileComponents();

    fixture = TestBed.createComponent(AutopilotPanelComponent);
    component = fixture.componentInstance;
    autopilotState = TestBed.inject(AutopilotStateService);
    toasts = TestBed.inject(ToastService);
    await settleZoneless(fixture);
  });

  afterEach(() => {
    autopilotState.clear();
    toasts.dismissAll();
    vi.useRealTimers();
  });

  /**
   * Drive the phase machine to 'done' (every indicator green, votes past the
   * initial targets) — the state that fires the completion toast. Uses
   * `TestBed.tick()` rather than `settleZoneless` so it also works under
   * `vi.useFakeTimers()`, where a real `setTimeout` would never resolve.
   */
  function reachDone(): void {
    fixture.componentRef.setInput('goodVotes', new Set([1, 2, 3, 4, 5]));
    fixture.componentRef.setInput('badVotes', new Set([11, 12, 13, 14, 15]));
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
    expect(steps.length).toBe(5);
  });

  it('should transition from good to bad phase', async () => {
    fixture.componentRef.setInput('goodVotes', new Set([1, 2, 3]));
    await settleZoneless(fixture);
    expect(component.state.phase).toBe('bad');
  });

  it('should transition from bad to hard phase', async () => {
    fixture.componentRef.setInput('goodVotes', new Set([1, 2, 3]));
    await settleZoneless(fixture);
    expect(component.state.phase).toBe('bad');

    fixture.componentRef.setInput('badVotes', new Set([4, 5, 6, 7]));
    await settleZoneless(fixture);
    expect(component.state.phase).toBe('hard');
  });

  it('should transition from hard to new when smart+stable are green', async () => {
    // Advance to hard phase
    fixture.componentRef.setInput('goodVotes', new Set([1, 2, 3]));
    fixture.componentRef.setInput('badVotes', new Set([4, 5, 6, 7]));
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
    fixture.componentRef.setInput('goodVotes', new Set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
    fixture.componentRef.setInput('badVotes', new Set([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]));
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
    fixture.componentRef.setInput('goodVotes', new Set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
    fixture.componentRef.setInput('badVotes', new Set([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]));
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

  it('should show the diversity (span) status icon during new phase', () => {
    // Advance to new phase
    autopilotState.checkPhaseTransition(3, 0);
    autopilotState.checkPhaseTransition(3, 4);
    autopilotState.updateFromLabelingStatus({
      good_count: 0,
      bad_count: 0,
      total_count: 0,
      smart: { status: 'green' },
      stable: { status: 'green' },
      span: { status: 'yellow' },
    });
    autopilotState.checkPhaseTransition(10, 10);
    expect(component.state.phase).toBe('new');

    const steps = component.steps;
    const newStep = steps.find((s: any) => s.phase === 'new');
    // The diversity (new) phase runs after smart + stable are already green,
    // so it shows a single span/diversity dot instead of repeating those two.
    expect(newStep!.statusIcons.length).toBe(1);
    // span was reported yellow above, so the diversity dot is yellow.
    expect(newStep!.statusIcons[0].color).toBe('yellow');
    // Tooltip explains the diversity indicator, not just its raw colour.
    expect(newStep!.statusIcons[0].title).toContain('Diverse');
    expect(newStep!.statusIcons[0].title).toContain('cover');
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
    fixture.componentRef.setInput('goodVotes', new Set([1, 2, 3]));
    fixture.componentRef.setInput('badVotes', new Set([4, 5, 6, 7]));
    await settleZoneless(fixture);
    expect(component.state.phase).toBe('hard');

    // Votes are cleared (e.g. new detector session); phase should regress
    fixture.componentRef.setInput('goodVotes', new Set());
    fixture.componentRef.setInput('badVotes', new Set());
    await settleZoneless(fixture);
    expect(component.state.phase).toBe('good');
  });

  it('should regress from hard to bad when good count drops below threshold', async () => {
    fixture.componentRef.setInput('goodVotes', new Set([1, 2, 3]));
    fixture.componentRef.setInput('badVotes', new Set([4, 5, 6, 7]));
    await settleZoneless(fixture);
    expect(component.state.phase).toBe('hard');

    // Good votes drop below threshold but bad are still sufficient
    fixture.componentRef.setInput('goodVotes', new Set([1]));
    fixture.componentRef.setInput('badVotes', new Set([4, 5, 6, 7]));
    await settleZoneless(fixture);
    expect(component.state.phase).toBe('good');
  });

  it('should show tooltip on each step label via title attribute', () => {
    const stepLabels = fixture.nativeElement.querySelectorAll('.ap-step-label');
    expect(stepLabels.length).toBe(5);
    // Active step (phase 1) leads with phase intent and ends with reselect hint
    expect(stepLabels[0].title).toContain('Phase 1');
    expect(stepLabels[0].title).toContain('Find initial goods');
    expect(stepLabels[0].title).toContain('reselect');
    // Future steps show phase intent only
    expect(stepLabels[1].title).toContain('Phase 2');
    expect(stepLabels[1].title).toContain('Find initial bads');
    expect(stepLabels[2].title).toContain('Refine the cutoff');
    expect(stepLabels[3].title).toContain('Cover a broad mix');
  });

  it('should show phase intent tooltip on each collapsed-step dot', async () => {
    fixture.componentRef.setInput('collapsed', true);
    await settleZoneless(fixture);
    const dots = fixture.nativeElement.querySelectorAll('.collapsed-step');
    expect(dots.length).toBe(5);
    expect(dots[0].title).toContain('Phase 1');
    expect(dots[0].title).toContain('Find initial goods');
    expect(dots[0].title).toContain('reselect');
    expect(dots[2].title).toContain('Phase 3');
    expect(dots[2].title).toContain('Refine the cutoff');
    expect(dots[2].title).toContain('uncertain items');
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
    it('announces where the user is being taken, with a countdown and an escape', () => {
      vi.useFakeTimers();
      reachDone();
      expect(component.state.phase).toBe('done');

      const [t] = toasts.toasts;
      expect(t.message).toContain('Done!');
      expect(t.countdown).toBeTruthy();
      expect(t.countdown!.label).toContain('Dashboard');
      expect(t.countdown!.remaining).toBe(10);
      expect(t.action!.label).toBe('Stay here');
    });

    it('ticks the countdown down each second', async () => {
      vi.useFakeTimers();
      reachDone();

      await vi.advanceTimersByTimeAsync(2000);
      expect(toasts.toasts[0].countdown!.remaining).toBe(8);
    });

    it('navigates to the Dashboard when the countdown runs out', async () => {
      vi.useFakeTimers();
      const router = TestBed.inject(Router);
      const navigate = vi.spyOn(router, 'navigate').mockResolvedValue(true);
      reachDone();

      await vi.advanceTimersByTimeAsync(9000);
      expect(navigate).not.toHaveBeenCalled();

      await vi.advanceTimersByTimeAsync(1000);
      expect(navigate).toHaveBeenCalledWith(['/dashboard']);
      // The toast is gone by the time the view changes.
      expect(navigate).toHaveBeenCalledTimes(1);
      expect(toasts.toasts.length).toBe(0);
    });

    it('calls off the return as soon as the user clicks anything', async () => {
      vi.useFakeTimers();
      const router = TestBed.inject(Router);
      const navigate = vi.spyOn(router, 'navigate').mockResolvedValue(true);
      reachDone();

      // The reported bug: the user goes straight on to export their detector
      // and the redirect fires mid-click, looking like a broken export.
      document.body.dispatchEvent(new Event('pointerdown', { bubbles: true }));
      await vi.advanceTimersByTimeAsync(30_000);

      expect(navigate).not.toHaveBeenCalled();
    });

    it('calls off the return on a keystroke too', async () => {
      vi.useFakeTimers();
      const router = TestBed.inject(Router);
      const navigate = vi.spyOn(router, 'navigate').mockResolvedValue(true);
      reachDone();

      document.body.dispatchEvent(new Event('keydown', { bubbles: true }));
      await vi.advanceTimersByTimeAsync(30_000);

      expect(navigate).not.toHaveBeenCalled();
    });

    it('keeps the news on screen when interaction cancels the return', async () => {
      vi.useFakeTimers();
      reachDone();

      document.body.dispatchEvent(new Event('pointerdown', { bubbles: true }));

      // The "Done!" headline survives — just without the countdown, the escape
      // button, and any suggestion the user is about to be moved.
      const [t] = toasts.toasts;
      expect(t.message).toContain('Done!');
      expect(t.countdown).toBeUndefined();
      expect(t.action).toBeUndefined();
      expect(t.detail).toContain('Staying here');

      // ...and it then clears itself on the ordinary success timer.
      await vi.advanceTimersByTimeAsync(5000);
      expect(toasts.toasts.length).toBe(0);
    });

    it('leaves the toast alone when the interaction is with the toast itself', async () => {
      vi.useFakeTimers();
      reachDone();

      // The toast stack lives outside this fixture, so stand in for it with an
      // element carrying the class the guard looks for.
      const stack = document.createElement('div');
      stack.className = 'toast-stack';
      const btn = document.createElement('button');
      stack.appendChild(btn);
      document.body.appendChild(stack);
      try {
        btn.dispatchEvent(new Event('pointerdown', { bubbles: true }));
        expect(toasts.toasts[0].countdown).toBeTruthy();
      } finally {
        document.body.removeChild(stack);
      }
    });

    it('does not re-arm the countdown when the user returns to the Train window', async () => {
      vi.useFakeTimers();
      const router = TestBed.inject(Router);
      const navigate = vi.spyOn(router, 'navigate').mockResolvedValue(true);
      reachDone();
      expect(toasts.toasts.length).toBe(1);

      // Leaving drops the toast; the autopilot run itself keeps going.
      fixture.destroy();
      expect(toasts.toasts.length).toBe(0);

      // Coming back rebuilds the panel. The hand-off was already offered once,
      // so it must not start counting down again — that is what made the
      // redirect feel like it was chasing the user around.
      const revisit = TestBed.createComponent(AutopilotPanelComponent);
      revisit.componentRef.setInput('goodVotes', new Set([1, 2, 3, 4, 5]));
      revisit.componentRef.setInput('badVotes', new Set([11, 12, 13, 14, 15]));
      revisit.componentRef.setInput('labelingStatus', ALL_GREEN);
      TestBed.tick();

      expect(toasts.toasts.length).toBe(0);
      await vi.advanceTimersByTimeAsync(30_000);
      expect(navigate).not.toHaveBeenCalled();
    });

    it('offers the hand-off again on a genuinely new autopilot run', async () => {
      vi.useFakeTimers();
      reachDone();
      expect(toasts.toasts.length).toBe(1);
      toasts.dismissAll();

      // Stopping and restarting autopilot is a new run, not a return visit.
      component.deactivate();
      component.activate();
      reachDone();

      expect(toasts.toasts[0].countdown!.remaining).toBe(10);
    });

    it('stays in the Train window when the escape button is used', async () => {
      vi.useFakeTimers();
      const router = TestBed.inject(Router);
      const navigate = vi.spyOn(router, 'navigate').mockResolvedValue(true);
      reachDone();

      // What the toast's action button does: run the handler, drop the toast.
      const t = toasts.toasts[0];
      t.action!.onClick();
      toasts.dismiss(t.id);

      await vi.advanceTimersByTimeAsync(30_000);
      expect(navigate).not.toHaveBeenCalled();
    });

    it('cancels the pending return when the panel is torn down', async () => {
      vi.useFakeTimers();
      const router = TestBed.inject(Router);
      const navigate = vi.spyOn(router, 'navigate').mockResolvedValue(true);
      reachDone();
      expect(toasts.toasts.length).toBe(1);

      // Leaving the Train window (tab switch / view change) destroys the panel.
      fixture.destroy();
      expect(toasts.toasts.length).toBe(0);

      await vi.advanceTimersByTimeAsync(10_000);
      expect(navigate).not.toHaveBeenCalled();
    });

    it('counts down from the exhausted state too', async () => {
      vi.useFakeTimers();
      const router = TestBed.inject(Router);
      const navigate = vi.spyOn(router, 'navigate').mockResolvedValue(true);

      fixture.componentRef.setInput('datasetSize', 1);
      fixture.componentRef.setInput('goodVotes', new Set([1]));
      TestBed.tick();
      expect(component.state.phase).toBe('exhausted');
      expect(toasts.toasts[0].message).toContain('Done!');
      expect(toasts.toasts[0].countdown!.remaining).toBe(10);

      await vi.advanceTimersByTimeAsync(10_000);
      expect(navigate).toHaveBeenCalledWith(['/dashboard']);
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
