import { ComponentFixture, TestBed } from '@angular/core/testing';
import { AutopilotPanelComponent } from './autopilot-panel.component';
import { AutopilotStateService } from '../../../services/autopilot-state.service';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { settleZoneless } from '../../../testing/settle-resource';

describe('AutopilotPanelComponent', () => {
  let component: AutopilotPanelComponent;
  let fixture: ComponentFixture<AutopilotPanelComponent>;
  let autopilotState: AutopilotStateService;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AutopilotPanelComponent],
      providers: [...provideZoneless()],
    }).compileComponents();

    fixture = TestBed.createComponent(AutopilotPanelComponent);
    component = fixture.componentInstance;
    autopilotState = TestBed.inject(AutopilotStateService);
    await settleZoneless(fixture);
  });

  afterEach(() => {
    autopilotState.clear();
  });

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

  it('should transition from good to bad phase', () => {
    component.goodVotes = new Set([1, 2, 3]);
    component.ngOnChanges({
      goodVotes: { currentValue: component.goodVotes, previousValue: new Set(), firstChange: false, isFirstChange: () => false },
    });
    expect(component.state.phase).toBe('bad');
  });

  it('should transition from bad to hard phase', () => {
    // Advance to bad phase first
    component.goodVotes = new Set([1, 2, 3]);
    autopilotState.checkPhaseTransition(3, 0);
    expect(component.state.phase).toBe('bad');

    component.badVotes = new Set([4, 5, 6, 7]);
    component.ngOnChanges({
      badVotes: { currentValue: component.badVotes, previousValue: new Set(), firstChange: false, isFirstChange: () => false },
    });
    expect(component.state.phase).toBe('hard');
  });

  it('should transition from hard to new when smart+stable are green', () => {
    // Advance to hard phase
    component.goodVotes = new Set([1, 2, 3]);
    component.badVotes = new Set([4, 5, 6, 7]);
    autopilotState.checkPhaseTransition(3, 0);
    autopilotState.checkPhaseTransition(3, 4);
    expect(component.state.phase).toBe('hard');

    component.labelingStatus = {
      good_count: 0,
      bad_count: 0,
      total_count: 0,
      smart: { status: 'green' },
      stable: { status: 'green' },
      span: { status: '' },
    };
    component.ngOnChanges({
      labelingStatus: { currentValue: component.labelingStatus, previousValue: null, firstChange: false, isFirstChange: () => false },
    });
    expect(component.state.phase).toBe('new');
  });

  it('should transition from new to done when span is green', () => {
    // Advance to new phase
    component.goodVotes = new Set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    component.badVotes = new Set([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]);
    autopilotState.checkPhaseTransition(3, 0);
    autopilotState.checkPhaseTransition(3, 4);
    autopilotState.updateFromLabelingStatus({
      good_count: 0,
      bad_count: 0,
      total_count: 0,
      smart: { status: 'green' },
      stable: { status: 'green' },
      span: { status: '' },
    });
    autopilotState.checkPhaseTransition(10, 10);
    expect(component.state.phase).toBe('new');

    component.labelingStatus = {
      good_count: 0,
      bad_count: 0,
      total_count: 0,
      smart: { status: 'green' },
      stable: { status: 'green' },
      span: { status: 'green' },
    };
    component.ngOnChanges({
      labelingStatus: { currentValue: component.labelingStatus, previousValue: null, firstChange: false, isFirstChange: () => false },
    });
    expect(component.state.phase).toBe('done');
  });

  it('should bounce from new back to hard when smart drops to yellow', () => {
    // Advance to new phase
    component.goodVotes = new Set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    component.badVotes = new Set([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]);
    autopilotState.checkPhaseTransition(3, 0);
    autopilotState.checkPhaseTransition(3, 4);
    autopilotState.updateFromLabelingStatus({
      good_count: 0,
      bad_count: 0,
      total_count: 0,
      smart: { status: 'green' },
      stable: { status: 'green' },
      span: { status: '' },
    });
    autopilotState.checkPhaseTransition(10, 10);
    expect(component.state.phase).toBe('new');

    // A surprise vote causes smart to drop
    component.labelingStatus = {
      good_count: 0,
      bad_count: 0,
      total_count: 0,
      smart: { status: 'yellow' },
      stable: { status: 'green' },
      span: { status: 'yellow' },
    };
    component.ngOnChanges({
      labelingStatus: { currentValue: component.labelingStatus, previousValue: null, firstChange: false, isFirstChange: () => false },
    });
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

  it('should regress from hard to good when vote counts drop to zero', () => {
    // Advance to hard phase with sufficient votes
    autopilotState.checkPhaseTransition(3, 4);
    expect(component.state.phase).toBe('hard');

    // Votes are cleared (e.g. new detector session); phase should regress
    component.goodVotes = new Set();
    component.badVotes = new Set();
    component.ngOnChanges({
      goodVotes: { currentValue: component.goodVotes, previousValue: new Set([1, 2, 3]), firstChange: false, isFirstChange: () => false },
    });
    expect(component.state.phase).toBe('good');
  });

  it('should regress from hard to bad when good count drops below threshold', () => {
    autopilotState.checkPhaseTransition(3, 4);
    expect(component.state.phase).toBe('hard');

    // Good votes drop below threshold but bad are still sufficient
    component.goodVotes = new Set([1]);
    component.badVotes = new Set([4, 5, 6, 7]);
    component.ngOnChanges({
      goodVotes: { currentValue: component.goodVotes, previousValue: new Set([1, 2, 3]), firstChange: false, isFirstChange: () => false },
    });
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
    fresh.componentInstance.labelsetGoodCount = 0;
    fresh.componentInstance.labelsetBadCount = 0;
    await settleZoneless(fresh);
    expect(fresh.componentInstance.state.retrainMode).toBe(false);
  });

  it('activate with detector labels from another dataset should enter retrain mode', async () => {
    autopilotState.clear();
    const fresh = TestBed.createComponent(AutopilotPanelComponent);
    // Simulate "trained on DatasetA, switched to DatasetB with 0 votes here":
    // current-dataset goodVotes/badVotes empty, but labelset counts positive.
    fresh.componentInstance.labelsetGoodCount = 5;
    fresh.componentInstance.labelsetBadCount = 4;
    fresh.componentInstance.goodVotes = new Set();
    fresh.componentInstance.badVotes = new Set();
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
});
