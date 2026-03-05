import { ComponentFixture, TestBed } from '@angular/core/testing';
import { AutopilotPanelComponent } from './autopilot-panel.component';
import { AutopilotStateService } from '../../../services/autopilot-state.service';

describe('AutopilotPanelComponent', () => {
  let component: AutopilotPanelComponent;
  let fixture: ComponentFixture<AutopilotPanelComponent>;
  let autopilotState: AutopilotStateService;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AutopilotPanelComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(AutopilotPanelComponent);
    component = fixture.componentInstance;
    autopilotState = TestBed.inject(AutopilotStateService);
    fixture.detectChanges();
  });

  afterEach(() => {
    autopilotState.clear();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should auto-start on init in good phase', () => {
    expect(component.state.phase).toBe('good');
    expect(component.running).toBeTrue();
  });

  it('should emit started on init', () => {
    autopilotState.clear();
    const fresh = TestBed.createComponent(AutopilotPanelComponent);
    const comp = fresh.componentInstance;
    spyOn(comp.started, 'emit');
    fresh.detectChanges();
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
    autopilotState.checkPhaseTransition(3, 0);
    expect(component.state.phase).toBe('bad');

    component.badVotes = new Set([1, 2, 3, 4]);
    component.ngOnChanges({
      badVotes: { currentValue: component.badVotes, previousValue: new Set(), firstChange: false, isFirstChange: () => false },
    });
    expect(component.state.phase).toBe('hard');
  });

  it('should transition from hard to new when smart+stable are green', () => {
    // Advance to hard phase
    autopilotState.checkPhaseTransition(3, 0);
    autopilotState.checkPhaseTransition(3, 4);
    expect(component.state.phase).toBe('hard');

    component.labelingStatus = {
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
    autopilotState.checkPhaseTransition(3, 0);
    autopilotState.checkPhaseTransition(3, 4);
    autopilotState.updateFromLabelingStatus({
      smart: { status: 'green' },
      stable: { status: 'green' },
      span: { status: '' },
    });
    autopilotState.checkPhaseTransition(10, 10);
    expect(component.state.phase).toBe('new');

    component.labelingStatus = {
      smart: { status: 'green' },
      stable: { status: 'green' },
      span: { status: 'green' },
    };
    component.ngOnChanges({
      labelingStatus: { currentValue: component.labelingStatus, previousValue: null, firstChange: false, isFirstChange: () => false },
    });
    expect(component.state.phase).toBe('done');
  });

  it('should deactivate autopilot', () => {
    spyOn(component.stopped, 'emit');
    component.deactivate();
    expect(component.state.phase).toBe('idle');
    expect(component.running).toBeFalse();
    expect(component.stopped.emit).toHaveBeenCalled();
  });

  it('should re-activate after deactivate', () => {
    component.deactivate();
    spyOn(component.started, 'emit');
    component.activate();
    expect(component.state.phase).toBe('good');
    expect(component.running).toBeTrue();
    expect(component.started.emit).toHaveBeenCalled();
  });

  it('should not re-activate if already running', () => {
    spyOn(component.started, 'emit');
    component.activate();
    expect(component.started.emit).not.toHaveBeenCalled();
  });

  it('should show tooltip on each step label via title attribute', () => {
    const stepLabels = fixture.nativeElement.querySelectorAll('.ap-step-label');
    expect(stepLabels.length).toBe(5);
    expect(stepLabels[0].title).toContain('good');
    expect(stepLabels[1].title).toContain('not what you want');
  });

  it('should mark current step as active and future steps as future', () => {
    const steps = component.steps;
    expect(steps[0].state).toBe('active');
    expect(steps[1].state).toBe('future');
    expect(steps[2].state).toBe('future');
  });
});
