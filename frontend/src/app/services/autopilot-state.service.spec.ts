import { TestBed } from '@angular/core/testing';
import { AutopilotStateService } from './autopilot-state.service';
import { LabelingStatusResponse } from '../models/api.models';

describe('AutopilotStateService', () => {
  let service: AutopilotStateService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(AutopilotStateService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should start idle', () => {
    expect(service.state.phase).toBe('idle');
    expect(service.running).toBeFalse();
  });

  it('activate should move to good phase', () => {
    service.activate();
    expect(service.state.phase).toBe('good');
    expect(service.running).toBeTrue();
  });

  it('activate when already running should be a no-op', () => {
    service.activate();
    service.activate();
    expect(service.state.phase).toBe('good');
  });

  it('deactivate should move to idle', () => {
    service.activate();
    service.deactivate();
    expect(service.state.phase).toBe('idle');
    expect(service.running).toBeFalse();
  });

  it('should transition from good to bad when enough good votes', () => {
    service.activate();
    service.checkPhaseTransition(3, 0); // goodToStart default is 3
    expect(service.state.phase).toBe('bad');
  });

  it('should transition from bad to hard when enough bad votes', () => {
    service.activate();
    service.checkPhaseTransition(3, 0);
    service.checkPhaseTransition(3, 4); // badToStart default is 4
    expect(service.state.phase).toBe('hard');
  });

  it('should transition from hard to new when smart and stable are green', () => {
    service.activate();
    service.checkPhaseTransition(3, 0);
    service.checkPhaseTransition(3, 4);

    const status: LabelingStatusResponse = {
      smart: { status: 'green' },
      stable: { status: 'green' },
      span: { status: 'yellow' },
    };
    service.updateFromLabelingStatus(status);
    service.checkPhaseTransition(10, 10);
    expect(service.state.phase).toBe('new');
  });

  it('should transition from new to done when span is green', () => {
    service.activate();
    service.checkPhaseTransition(3, 0);
    service.checkPhaseTransition(3, 4);

    service.updateFromLabelingStatus({
      smart: { status: 'green' },
      stable: { status: 'green' },
      span: { status: 'yellow' },
    });
    service.checkPhaseTransition(10, 10);

    service.updateFromLabelingStatus({
      smart: { status: 'green' },
      stable: { status: 'green' },
      span: { status: 'green' },
    });
    service.checkPhaseTransition(15, 15);
    expect(service.state.phase).toBe('done');
  });

  it('updateFromLabelingStatus should update status fields', () => {
    service.activate();
    const status: LabelingStatusResponse = {
      smart: { status: 'yellow' },
      stable: { status: 'green' },
      span: { status: 'red', diversity_level: 0.75 },
    };
    service.updateFromLabelingStatus(status);

    expect(service.state.smartStatus).toBe('yellow');
    expect(service.state.stableStatus).toBe('green');
    expect(service.state.spanStatus).toBe('red');
    expect(service.state.fracDiversity).toBe(0.75);
  });

  it('clear should reset to initial state', () => {
    service.activate();
    service.checkPhaseTransition(3, 0);
    service.clear();

    expect(service.state.phase).toBe('idle');
    expect(service.state.smartStatus).toBe('');
  });

  it('state$ should emit on changes', (done) => {
    const phases: string[] = [];
    service.state$.subscribe((s) => phases.push(s.phase));

    service.activate();

    setTimeout(() => {
      expect(phases).toContain('idle');
      expect(phases).toContain('good');
      done();
    });
  });
});
