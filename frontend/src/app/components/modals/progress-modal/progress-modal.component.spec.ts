import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpRequest } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { Subject } from 'rxjs';
import { ProgressModalComponent } from './progress-modal.component';
import { ProgressEventsService } from '../../../services/progress-events.service';
import { VotingIterationsResponse } from '../../../models/api.models';
import { provideZoneless } from '../../../testing/zoneless-testbed';

describe('ProgressModalComponent', () => {
  let component: ProgressModalComponent;
  let fixture: ComponentFixture<ProgressModalComponent>;
  let httpMock: HttpTestingController;
  // Controllable stand-in for the `eval` SSE channel so a test can push a
  // "done" progress frame at will.
  let votingIterations$: Subject<VotingIterationsResponse>;

  beforeEach(async () => {
    votingIterations$ = new Subject<VotingIterationsResponse>();
    await TestBed.configureTestingModule({
      imports: [ProgressModalComponent],
      providers: [
        ...provideZoneless(),
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: ProgressEventsService, useValue: { votingIterations$ } },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(ProgressModalComponent);
    component = fixture.componentInstance;
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    component.ngOnDestroy();
    httpMock.verify();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should set title based on metric', () => {
    component.metric = 'smart';
    expect(component.title).toContain('Detector Accuracy');

    component.metric = 'stable';
    expect(component.title).toContain('Changes Its Mind');

    component.metric = 'diverse';
    expect(component.title).toContain('Your Votes Cover');
  });

  it('should start in analyzing state', async () => {
    vi.useFakeTimers();
    try {
      component.metric = 'smart';
      // TestBed.tick() runs ngOnInit (kicks off the train POST) under zoneless.
      TestBed.tick();
      expect(component.analyzing).toBe(true);

      // Progress now arrives over the `eval` SSE channel, not via HTTP polling.
      // The only HTTP call is the train-and-score POST, which returns a job
      // envelope; on a cache hit (status=done) the component applies the data
      // inline without polling.
      const trainReq = httpMock.expectOne('/api/eval/train-and-score');
      trainReq.flush({
        job_id: 'abc',
        status: 'done',
        metric: 'smart',
        error_cost: [{ num_labels: 5, error_cost: 0.5 }],
      });

      await vi.advanceTimersByTimeAsync(50);
      expect(component.analyzing).toBe(false);
      expect(component.chartData.length).toBe(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it('should emit closed on close', () => {
    vi.spyOn(component.closed, 'emit');
    component.close();
    expect(component.closed.emit).toHaveBeenCalled();
  });

  it('keeps polling for the result after the SSE reports done', async () => {
    // Regression: the eval `idle/Done` frame lands while the job status
    // endpoint may still say `running`. The progress watcher must not tear
    // down the result poller, or `analyzing` hangs forever.
    vi.useFakeTimers();
    try {
      component.metric = 'smart';
      TestBed.tick();

      // Job started, not a cache hit -> the component polls for the result.
      httpMock.expectOne('/api/eval/train-and-score').flush({
        job_id: 'j1',
        status: 'running',
        current: 0,
        total: 5,
      });
      expect(component.analyzing).toBe(true);

      // SSE says done *before* the result endpoint flips. This used to kill
      // the poller; it must only stop the progress watcher now.
      votingIterations$.next({ progress: 5, total: 5, done: true });

      const isResult = (req: HttpRequest<unknown>) =>
        req.url === '/api/eval/train-and-score/result';

      // First poll still sees `running`.
      await vi.advanceTimersByTimeAsync(200);
      httpMock.expectOne(isResult).flush({ job_id: 'j1', status: 'running', current: 5, total: 5 });
      expect(component.analyzing).toBe(true);

      // Next poll observes completion and applies the data.
      await vi.advanceTimersByTimeAsync(500);
      httpMock.expectOne(isResult).flush({
        job_id: 'j1',
        status: 'done',
        metric: 'smart',
        error_cost: [{ num_labels: 5, error_cost: 0.5 }],
      });

      await vi.advanceTimersByTimeAsync(50);
      expect(component.analyzing).toBe(false);
      expect(component.chartData.length).toBe(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it('cancels the in-flight eval job when dismissed via the modal', async () => {
    // Regression: Escape / X / backdrop route through the modal's `(closed)`
    // to `onCancel()`, so they must cancel the running job like the in-body
    // Cancel button does — not silently orphan it.
    vi.useFakeTimers();
    try {
      component.metric = 'stable';
      TestBed.tick();

      httpMock.expectOne('/api/eval/train-and-score').flush({
        job_id: 'j2',
        status: 'running',
        current: 0,
        total: 3,
      });

      vi.spyOn(component.closed, 'emit');
      component.onCancel();

      httpMock.expectOne('/api/eval/train-and-score/cancel/j2').flush({ ok: true });
      expect(component.closed.emit).toHaveBeenCalled();
      expect(component.analyzing).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });

  it('does not arm a poller when destroyed before the job envelope arrives', async () => {
    // Regression: destroying the modal while the train POST is in flight must
    // not leave a poller running against an already-completed `destroy$`.
    vi.useFakeTimers();
    try {
      component.metric = 'diverse';
      TestBed.tick();

      const trainReq = httpMock.expectOne('/api/eval/train-and-score');
      component.ngOnDestroy();
      // The POST resolves after destroy; `takeUntil(destroy$)` drops it, so
      // `pollEvalJob` never runs and no result poll is issued.
      trainReq.flush({ job_id: 'j3', status: 'running', current: 0, total: 3 });

      await vi.advanceTimersByTimeAsync(1000);
      httpMock.expectNone((req: HttpRequest<unknown>) =>
        req.url === '/api/eval/train-and-score/result',
      );
    } finally {
      vi.useRealTimers();
    }
  });
});
