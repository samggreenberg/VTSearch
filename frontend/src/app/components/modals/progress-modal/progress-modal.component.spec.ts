import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HttpRequest } from '@angular/common/http';
import { HttpTestingController } from '@angular/common/http/testing';
import { Subject } from 'rxjs';
import { ProgressModalComponent } from './progress-modal.component';
import { ProgressEventsService } from '../../../services/progress-events.service';
import { VotingIterationsResponse } from '../../../models/api.models';
import { provideZoneless } from '../../../testing/zoneless-testbed';
import { provideHttpTesting } from '../../../testing/test-providers';

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
        ...provideHttpTesting(),
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

  const isHistory = (req: HttpRequest<unknown>) => req.url === '/api/indicator-score-history';
  const isResult = (req: HttpRequest<unknown>) => req.url === '/api/eval/train-and-score/result';

  /** Flush the cached-history GET that `ngOnInit` always issues first, with a
   *  cache miss so the component falls through to the async job. */
  const missCachedHistory = (metric = 'smart') => {
    httpMock.expectOne(isHistory).flush({ metric, history: [], complete: false });
  };

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should set title based on metric', () => {
    fixture.componentRef.setInput('metric', 'smart');
    expect(component.title).toContain('Detector Accuracy');

    fixture.componentRef.setInput('metric', 'stable');
    expect(component.title).toContain('Changes Its Mind');

    fixture.componentRef.setInput('metric', 'diverse');
    expect(component.title).toContain('Your Votes Cover');
  });

  it('should start in analyzing state', async () => {
    vi.useFakeTimers();
    try {
      fixture.componentRef.setInput('metric', 'smart');
      // TestBed.tick() runs ngOnInit (the cached read) under zoneless.
      TestBed.tick();
      expect(component.analyzing).toBe(true);
      missCachedHistory('smart');

      // Progress now arrives over the `eval` SSE channel, not via HTTP polling.
      // The only other HTTP call is the train-and-score POST, which returns a
      // job envelope; on a cache hit (status=done) the component applies the
      // data inline without polling.
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
      fixture.componentRef.setInput('metric', 'smart');
      TestBed.tick();
      missCachedHistory('smart');

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
      fixture.componentRef.setInput('metric', 'stable');
      TestBed.tick();
      missCachedHistory('stable');

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
      fixture.componentRef.setInput('metric', 'diverse');
      TestBed.tick();
      missCachedHistory('diverse');

      const trainReq = httpMock.expectOne('/api/eval/train-and-score');
      component.ngOnDestroy();
      // `takeUntil(destroy$)` unsubscribes from the in-flight POST, so the
      // request is cancelled: its `next` never runs, `pollEvalJob` is never
      // armed, and no result poll is issued.
      expect(trainReq.cancelled).toBe(true);

      await vi.advanceTimersByTimeAsync(1000);
      httpMock.expectNone((req: HttpRequest<unknown>) =>
        req.url === '/api/eval/train-and-score/result',
      );
    } finally {
      vi.useRealTimers();
    }
  });

  it('renders a complete cached history without starting a job', async () => {
    // The warm-cache fast path: the per-step cache already covers the label
    // history, so the plot paints straight from the cheap GET and no MLP
    // retraining is triggered at all.
    vi.useFakeTimers();
    try {
      fixture.componentRef.setInput('metric', 'smart');
      TestBed.tick();

      httpMock.expectOne(isHistory).flush({
        metric: 'smart',
        history: [{ num_labels: 5, error_cost: 0.5 }],
        complete: true,
      });

      await vi.advanceTimersByTimeAsync(50);
      expect(component.analyzing).toBe(false);
      expect(component.runningJob).toBe(false);
      expect(component.chartData.length).toBe(1);
      httpMock.expectNone('/api/eval/train-and-score');
    } finally {
      vi.useRealTimers();
    }
  });

  it('falls back to the async job when the cached history is incomplete', async () => {
    // Regression for the hang: the GET must never block on a cache advance, so
    // an incomplete cache reports `complete: false` and the modal hands off to
    // the background job — which is what surfaces the progress bar + Cancel.
    vi.useFakeTimers();
    try {
      fixture.componentRef.setInput('metric', 'stable');
      TestBed.tick();

      expect(component.runningJob).toBe(false);
      missCachedHistory('stable');
      expect(component.runningJob).toBe(true);
      expect(component.analyzing).toBe(true);

      httpMock.expectOne('/api/eval/train-and-score').flush({
        job_id: 'j3',
        status: 'done',
        metric: 'stable',
        stability: [{ num_labels: 5, num_flips: 1 }],
      });

      await vi.advanceTimersByTimeAsync(50);
      expect(component.analyzing).toBe(false);
      expect(component.chartData.length).toBe(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it('falls back to the async job when the cached read fails', async () => {
    // A failed cached read is recoverable: the job recomputes the series from
    // scratch, so the modal must not settle on the "no history" empty state.
    vi.useFakeTimers();
    try {
      fixture.componentRef.setInput('metric', 'diverse');
      TestBed.tick();

      httpMock.expectOne(isHistory).flush(
        { message: 'Score history computation failed' },
        { status: 500, statusText: 'Server Error' },
      );
      expect(component.runningJob).toBe(true);
      expect(component.emptyHistory).toBe(false);

      httpMock.expectOne('/api/eval/train-and-score').flush({
        job_id: 'j4',
        status: 'done',
        metric: 'diverse',
        diversity: [{ num_labels: 5, diversity_level: 2 }],
      });

      await vi.advanceTimersByTimeAsync(50);
      expect(component.analyzing).toBe(false);
      expect(component.chartData.length).toBe(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it('shows the empty state when a finished job yields no points', async () => {
    vi.useFakeTimers();
    try {
      fixture.componentRef.setInput('metric', 'smart');
      TestBed.tick();
      missCachedHistory('smart');

      httpMock.expectOne('/api/eval/train-and-score').flush({
        job_id: 'j5',
        status: 'done',
        metric: 'smart',
        error_cost: [],
      });

      await vi.advanceTimersByTimeAsync(50);
      expect(component.analyzing).toBe(false);
      expect(component.emptyHistory).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });
});
