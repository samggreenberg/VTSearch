import { ComponentFixture, TestBed } from '@angular/core/testing';
import { AutopilotPanelComponent } from './autopilot-panel.component';

describe('AutopilotPanelComponent', () => {
  let component: AutopilotPanelComponent;
  let fixture: ComponentFixture<AutopilotPanelComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AutopilotPanelComponent],
    }).compileComponents();

    fixture = TestBed.createComponent(AutopilotPanelComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should start in idle state', () => {
    expect(component.state.phase).toBe('idle');
    expect(component.running).toBeFalse();
  });

  it('should show start button when idle', () => {
    expect(fixture.nativeElement.querySelector('.autopilot-idle')).toBeTruthy();
    expect(fixture.nativeElement.querySelector('.autopilot-active')).toBeNull();
  });

  it('should transition to good phase on start', () => {
    spyOn(component.start, 'emit');
    component.onStart();
    expect(component.state.phase).toBe('good');
    expect(component.running).toBeTrue();
    expect(component.start.emit).toHaveBeenCalled();
  });

  it('should show steps when running', () => {
    component.onStart();
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.autopilot-active')).toBeTruthy();
    const steps = fixture.nativeElement.querySelectorAll('.ap-step');
    expect(steps.length).toBe(5);
  });

  it('should transition from good to bad phase', () => {
    component.onStart();
    component.goodVotes = new Set([1, 2, 3]);
    component.ngOnChanges({
      goodVotes: { currentValue: component.goodVotes, previousValue: new Set(), firstChange: false, isFirstChange: () => false },
    });
    expect(component.state.phase).toBe('bad');
  });

  it('should transition from bad to hard phase', () => {
    component.onStart();
    component.state = { ...component.state, phase: 'bad' };
    component.badVotes = new Set([1, 2, 3, 4]);
    component.ngOnChanges({
      badVotes: { currentValue: component.badVotes, previousValue: new Set(), firstChange: false, isFirstChange: () => false },
    });
    expect(component.state.phase).toBe('hard');
  });

  it('should transition from hard to new when smart+stable are green', () => {
    component.onStart();
    component.state = { ...component.state, phase: 'hard' };
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
    component.onStart();
    component.state = { ...component.state, phase: 'new', smartStatus: 'green', stableStatus: 'green' };
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

  it('should stop autopilot', () => {
    spyOn(component.stop, 'emit');
    component.onStart();
    component.onStop();
    expect(component.state.phase).toBe('idle');
    expect(component.running).toBeFalse();
    expect(component.stop.emit).toHaveBeenCalled();
  });

  it('should mark current step as active and future steps as future', () => {
    component.onStart();
    const steps = component.steps;
    expect(steps[0].state).toBe('active');
    expect(steps[1].state).toBe('future');
    expect(steps[2].state).toBe('future');
  });
});
